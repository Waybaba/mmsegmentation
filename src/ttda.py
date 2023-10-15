from typing import Optional

import torch
from mmengine.hooks import Hook
from mmengine.runner import Runner

from mmengine.config import Config, DictAction
from copy import deepcopy
from mmseg.registry import HOOKS
from mmengine.model import (MMDistributedDataParallel, convert_sync_batchnorm,
                            is_model_wrapper, revert_sync_batchnorm)
from mmengine.model.efficient_conv_bn_eval import \
    turn_on_efficient_conv_bn_eval

@HOOKS.register_module()
class TTDAHook(Hook):
    """Check invalid loss hook.

    This hook will regularly check whether the loss is valid
    during training.

    Args:
        interval (int): Checking interval (every k iterations).
            Default: 50.
    """

    def __init__(self, turn_on_adapt=False, **kwargs) -> None:
        self.turn_on_adapt = turn_on_adapt
        self.kwargs = Config(kwargs)

    def fake_train_init(self, runner):
        """ Since we need to init something before test
        the original mmseg does not do this since test does not optimizer ...
        we here fetch some train code to init
        mmengine/runner/runner.py -> Runner.train()
        """
        if is_model_wrapper(runner.model):
            ori_model = runner.model.module
        else:
            ori_model = runner.model
        assert hasattr(ori_model, 'train_step'), (
            'If you want to train your model, please make sure your model '
            'has implemented `train_step`.')

        if runner._val_loop is not None:
            assert hasattr(ori_model, 'val_step'), (
                'If you want to validate your model, please make sure your '
                'model has implemented `val_step`.')

        if runner._train_loop is None:
            raise RuntimeError(
                '`self._train_loop` should not be None when calling train '
                'method. Please provide `train_dataloader`, `train_cfg`, '
                '`optimizer` and `param_scheduler` arguments when '
                'initializing runner.')

        # runner._train_loop = runner.build_train_loop(
        #     runner._train_loop)  # type: ignore

        # `build_optimizer` should be called before `build_param_scheduler`
        #  because the latter depends on the former
        runner.optim_wrapper = runner.build_optim_wrapper(runner.optim_wrapper)
        # Automatically scaling lr by linear scaling rule
        runner.scale_lr(runner.optim_wrapper, runner.auto_scale_lr)

        if runner.param_schedulers is not None:
            runner.param_schedulers = runner.build_param_scheduler(  # type: ignore
                runner.param_schedulers)  # type: ignore

        if runner._val_loop is not None:
            runner._val_loop = runner.build_val_loop(
                runner._val_loop)  # type: ignore
        # TODO: add a contextmanager to avoid calling `before_run` many times
        # runner.call_hook('before_run')

        # initialize the model weights
        runner._init_model_weights() # ! note the process when train multiple times

        # try to enable efficient_conv_bn_eval feature
        modules = runner.cfg.get('efficient_conv_bn_eval', None)
        if modules is not None:
            runner.logger.info(f'Enabling the "efficient_conv_bn_eval" feature'
                             f' for sub-modules: {modules}')
            turn_on_efficient_conv_bn_eval(ori_model, modules)

        # make sure checkpoint-related hooks are triggered after `before_run`
        runner.load_or_resume()

        # Initiate inner count of `optim_wrapper`.
        # runner.optim_wrapper.initialize_count_status( # ! TODO how to handle this?
        #     runner.model,
        #     runner._train_loop.iter,  # type: ignore
        #     runner._train_loop.max_iters)  # type: ignore

        # Maybe compile the model according to options in self.cfg.compile
        # This must be called **AFTER** model has been wrapped.
        # runner._maybe_compile('train_step')

        # model = runner.train_loop.run()  # type: ignore
        # runner.call_hook('after_run')
        return None

    @torch.enable_grad()
    def ttda_before_test(self, runner, batch_idx, data_batch):
        """
        input:
            inputs: 
                inputs [(CH, 512, 1024)], data_samples [SegDataSample]
                {
                    gt_sem_seg: (512, 1024)
                    img_path: 
                    metainfo:
                        scale_factor:
                        ori_shape:
                        img_shape:
                        reduce_zero_label:
                    img_shape:
                    ori_shape:
                    pred_sem_seg: # would have value after model.test_step
                    seg_logits: # would have value after model.test_step
                    scale_factor:
                }
        """
        # init
        if not self.turn_on_adapt: return
        USE_PSEUDO_LABEL = True
        inputs, data_samples = data_batch["inputs"], data_batch["data_samples"]

        # inference label
        data_samples_ = runner.model.test_step(data_batch) # data_samples_[0].pred_sem_seg.shape = (512, 1024)

        # set data_batch_for_adapt for train
        data_batch_for_adapt = deepcopy(data_batch)
        if USE_PSEUDO_LABEL:
            for i in range(len(data_samples)):
                data_batch_for_adapt["data_samples"][i].gt_sem_seg = data_samples_[i].pred_sem_seg

        ### adapt
        optim_wrapper = runner.optim_wrapper
        model = runner.model
        with optim_wrapper.optim_context(model):
            data_batch_for_adapt = model.data_preprocessor(data_batch_for_adapt, True)
            losses = model._run_forward(data_batch_for_adapt, mode='loss')  # type: ignore
        parsed_losses, log_vars = model.parse_losses(losses)  # type: ignore
        optim_wrapper.update_params(parsed_losses)

        #
        runner.visualizer.add_scalars({
            "adapt/"+k: v.item() for k, v in log_vars.items()
        })
        dict_log = {k: "{:.2f}".format(v.item()) for k, v in log_vars.items()}
        runner.logger.info(f"log_vars: {dict_log}")
    
    ### hooks
    
    def before_run(self, runner) -> None:
        if not self.turn_on_adapt: return
        # if self.mode == "test":
        runner.logger.info('fake train init')
        self.fake_train_init(runner)
        runner.logger.info('fake train init done')
        # else:
        #     raise ValueError(f"mode {self.mode} not supported, only 'test' for TTDA ")

    def before_test_iter(self,
                         runner: Runner,
                         batch_idx: int,
                         data_batch: Optional[dict] = None) -> None:
        """Regularly check whether the loss is valid every n iterations.

        Args:
            runner (:obj:`Runner`): The runner of the training process.
            batch_idx (int): The index of the current batch in the train loop.
            data_batch (dict, Optional): Data from dataloader.
                Defaults to None.
            outputs (dict, Optional): Outputs from model. Defaults to None.
        """
        self.ttda_before_test(runner, batch_idx, data_batch)

    def after_test_epoch(self, runner, metrics):
        # runner.logger.info('after test epoch')
        # revert_sync_batchnorm(runner.model) # ! ??? shoud we handle?
        # runner.logger.info('after test epoch done')
        runner.visualizer.add_scalars({
            "test/"+k: v for k, v in metrics.items()
        })

    def after_test_iter(self, runner, batch_idx, data_batch, outputs):
        # metrics = runner.test_evaluator.evaluate(len(runner.test_dataloader.dataset))
        # runner.visualizer.add_scalars({
        #     "test_step/"+k: v for k, v in metrics.items()
        # })
        # runner.logger.info(f"test iter {batch_idx} done, metrics: {metrics}")
        pass