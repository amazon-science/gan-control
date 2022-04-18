# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from datetime import datetime
import json


class DefaultObj(object):
    def __init__(self, dict):
        self.__dict__ = dict


def read_json(path, return_obj=False):
    with open(path) as json_file:
        data = json.load(json_file)
    if return_obj:
        data = DefaultObj(data)
    return data


def write_json(data_dict, path):
    with open(path, 'w') as outfile:
        json.dump(data_dict, outfile)


def setup_logging_from_args(args):
    """
    Calls setup_logging, exports args and creates a ResultsLog class.
    Can resume training/logging if args.resume is set
    """
    def set_args_default(field_name, value):
        if hasattr(args, field_name):
            return eval('args.' + field_name)
        else:
            return value

    # Set default args in case they don't exist in args
    # resume = set_args_default('resume', False)
    args.save_name = f"{args.save_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    results_dir = set_args_default('results_dir', './results')

    save_path = os.path.join(results_dir, args.save_name)
    os.makedirs(save_path, exist_ok=True)
    # log_file = os.path.join(save_path, 'log.txt')

    export_args(args, save_path)
    return save_path


def export_args(args, save_path):
    """
    args: argparse.Namespace
        arguments to save
    save_path: string
        path to directory to save at
    """
    os.makedirs(save_path, exist_ok=True)
    json_file_name = os.path.join(save_path, 'args.json')
    with open(json_file_name, 'w') as fp:
        json.dump(args.__dict__, fp, sort_keys=True, indent=4)

