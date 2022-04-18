# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path
_PWD = Path(__file__).absolute().parent
sys.path.append(str(_PWD.parent))

import argparse
from gan_control.trainers.generator_trainer import GeneratorTrainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True)

    config_path = parser.parse_args().config_path
    trainer = GeneratorTrainer(config_path)
    trainer.dry_run()
    trainer.train()

