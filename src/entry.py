# Copyright (c) OpenMMLab. All rights reserved.
import pyrootutils
root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True, indicator=["configs"])
from mmengine.runner import Runner
import hydra
from mmseg.registry import RUNNERS
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig


def parse_tuple(cfg):
    """
    Recursively instantiate the config.
    ! we mannually do this since hydra can not parse tuple recursively
    e.g.
    ngram_range:
    _target_: builtins.tuple
    _args_:
      - [2, 3]
    https://github.com/facebookresearch/hydra/issues/1432#issuecomment-786943368
    """
    if isinstance(cfg, (DictConfig, dict)):
        if "_target_" in cfg:
            cfg = hydra.utils.instantiate(cfg, _recursive_=False)
            return cfg
        for k, v in cfg.items():
            # if k == "crop_size":  print(1)
            cfg[k] = parse_tuple(v)
    elif isinstance(cfg, (ListConfig, list)):
        for i, v in enumerate(cfg):
            cfg[i] = parse_tuple(v)
    return cfg

def replace_tuples_in_dict(dict1, dict2):
    for key, value in dict1.items():
        if isinstance(value, tuple):
            dict2[key] = value
        elif isinstance(value, dict):
            replace_tuples_in_dict(value, dict2[key])
        elif isinstance(value, list):
            for idx, item in enumerate(value):
                if isinstance(item, dict):
                    replace_tuples_in_dict(item, dict2[key][idx])
                elif isinstance(item, tuple):
                    dict2[key][idx] = item


@hydra.main(version_base=None, config_path=str(root / "configs_hydra"), config_name="entry.yaml")	
def main(cfg):
	# if not os.path.exists(root / ".env"):
		# raise FileNotFoundError("Please create .env file in the root directory. See .env.example for reference.")

    # to pure dict recursively
    cfg = OmegaConf.to_container(cfg)
    cfg_with_parse_tuple = parse_tuple(cfg) # can keep tuple but can not parse path
    cfg_with_parse_path = hydra.utils.instantiate(cfg) # can parse path but can not keep tuple
    cfg_with_parse_path = OmegaConf.to_container(cfg_with_parse_path)
    replace_tuples_in_dict(cfg_with_parse_tuple, cfg_with_parse_path)
    cfg = cfg_with_parse_path


    from mmengine.config import Config, DictAction
    cfg_py = Config.fromfile("configs/segformer/segformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py")


    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    runner.train()

if __name__ == "__main__":
    main()