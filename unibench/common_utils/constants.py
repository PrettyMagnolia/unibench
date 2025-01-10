"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from os.path import join as pjoin
from pathlib import Path
import os
import argparse


##################################################################
# DIRECTORIES
##################################################################
PROJ_DIR = Path(__file__).parent.parent.absolute()
CURRENT_DIR = Path(os.getcwd())
HUB_CACHE_DIR = Path.home().joinpath(".cache").joinpath("torch").joinpath("hub")
# CACHE_DIR = Path.home().joinpath(".cache").joinpath("unibench")
CACHE_DIR = Path("/mnt/shared/unibench")

DATA_DIR = CACHE_DIR.joinpath("data").joinpath('raw')
MASK_DIR = CACHE_DIR.joinpath("data").joinpath('DINO')
OUTPUT_DIR = CACHE_DIR.joinpath("outputs")
LOCK_DIR = CACHE_DIR.joinpath("locks")
DS_CACHE_DIR = CACHE_DIR.joinpath("cache")

parser = argparse.ArgumentParser(description='Evaluate models on benchmarks.')
parser.add_argument('--use_mask', action='store_true', help='Use mask for evaluation')
parser.add_argument('--model_ids', nargs='+', help='List of model names to evaluate')
parser.add_argument('--dataset_names', nargs='+', help='List of dataset names to process')

args = parser.parse_args()
USE_MASK = args.use_mask
