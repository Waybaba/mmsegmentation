# Copyright (c) OpenMMLab. All rights reserved.
import pyrootutils
root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True, indicator=[".env"])
from mmengine.runner import Runner
import hydra
from mmseg.registry import RUNNERS
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from mmengine.config import Config, DictAction
import src.ttda
import wandb

def parse_tuple(cfg):
    """
    Recursively instantiate the config to support tuple parsing.

    Hydra by default cannot recursively parse tuples, and would also convert
    tuples into lists. This function creates a new dictionary or list to
    prevent modifications on the original config and to retain tuple format.

    Example:
        ngram_range:
        _target_: builtins.tuple
        _args_:
          - [2, 3]
    Return:
        A pure dict
    
    Reference:
    https://github.com/facebookresearch/hydra/issues/1432#issuecomment-786943368
    """
    # For dictionaries or DictConfig
    if isinstance(cfg, (DictConfig, dict)):
        # Instantiate if "_target_" key is present, else recurse
        return hydra.utils.instantiate(cfg, _recursive_=False) if "_target_" in cfg else {k: parse_tuple(v) for k, v in cfg.items()}

    # For lists or ListConfig
    elif isinstance(cfg, (ListConfig, list)):
        return [parse_tuple(v) for v in cfg]

    # Return original value for non-list or non-dict types
    return cfg

def link_output(cfg):
	"""
		link the output_dir to ./debug/latest/output/ for easier access
		if there exists a previous link, delete it first
		note that the parent dir could miss, so create it if necessary
	"""
	import os
	from pathlib import Path

	output_dir = Path(cfg.paths.output_dir)
	root = Path.cwd()
	latest_dir = root / "debug" / "latest" / "output"
	
	latest_dir.parent.mkdir(parents=True, exist_ok=True)

	if latest_dir.is_symlink():
		latest_dir.unlink()
	
	os.symlink(output_dir, latest_dir)

	print(f"Linked output_dir to {latest_dir}")

@hydra.main(version_base=None, config_path=str(root / "configs_hydra"), config_name="entry.yaml")	
def main(cfg):

    cfg = parse_tuple(cfg) # to pure dict recursively
    cfg = Config(cfg) # mmcv cfg
    
    link_output(cfg)

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    if cfg.train:  runner.train()
    if cfg.test:  metrics = runner.test()
    
    wandb.finish()

if __name__ == "__main__":
    main()