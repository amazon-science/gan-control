# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import os
from pathlib import Path
_PWD = Path(__file__).absolute().parent
sys.path.append(str(_PWD.parent))
sys.path.append(os.path.join(str(_PWD.parent.parent), 'face-alignment'))

import argparse
from gan_control.trainers.controller_trainer import ControllerTrainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True)

    config_path = parser.parse_args().config_path
    trainer = ControllerTrainer(config_path)
    trainer.train()

# python -m train_controller --config_path configs512/id_orientation_expression_512.json