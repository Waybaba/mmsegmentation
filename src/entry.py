# Copyright (c) OpenMMLab. All rights reserved.
import pyrootutils
root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True, indicator=["configs"])
from mmengine.runner import Runner

import hydra

from mmseg.registry import RUNNERS



@hydra.main(version_base=None, config_path=str(root / "configs"), config_name="entry.yaml")	
def main(cfg):
	# if not os.path.exists(root / ".env"):
		# raise FileNotFoundError("Please create .env file in the root directory. See .env.example for reference.")
	
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
