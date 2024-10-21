"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import os
import pickle
from pathlib import Path
import random
from typing import Optional
import numpy as np
import shutil
from huggingface_hub import hf_hub_download, snapshot_download
import torch
from torchvision import transforms

from ..benchmarks_zoo.registry import get_benchmark_info
from ..benchmarks_zoo import benchmarks, list_benchmarks
from ..models_zoo.registry import get_model_info
from ..models_zoo import models, list_models

import pandas as pd
from rich.table import Table
from rich import box

from pycocotools import mask as maskUtils
from scipy.ndimage import convolve


def df_to_table(
        pandas_dataframe: pd.DataFrame,
        rich_table: Table,
        show_index: bool = True,
        index_name: Optional[str] = None,
) -> Table:
    """Convert a pandas.DataFrame obj into a rich.Table obj.
    Args:
        pandas_dataframe (DataFrame): A Pandas DataFrame to be converted to a rich Table.
        rich_table (Table): A rich Table that should be populated by the DataFrame values.
        show_index (bool): Add a column with a row count to the table. Defaults to True.
        index_name (str, optional): The column name to give to the index column. Defaults to None, showing no value.
    Returns:
        Table: The rich Table instance passed, populated with the DataFrame values."""

    if show_index:
        index_name = str(index_name) if index_name else ""
        rich_table.add_column(index_name)

    for column in pandas_dataframe.columns:
        rich_table.add_column(str(column))

    for index, value_list in enumerate(pandas_dataframe.values.tolist()):
        row = [str(pandas_dataframe.index.to_list()[index])] if show_index else []
        row += [str(round(float(x), 2)) for x in value_list]
        rich_table.add_row(*row)

    rich_table.row_styles = ["none", "dim"]
    rich_table.box = box.SIMPLE_HEAD

    return rich_table


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_benchmark_mappings(axis, benchmarks=None):
    if benchmarks is None:
        benchmarks = list_benchmarks()
    benchmark_mappings = {}
    for benchmark in benchmarks:
        if axis is None:
            benchmark_mappings[benchmark] = get_benchmark_info(benchmark)
        else:
            benchmark_mappings[benchmark] = get_benchmark_info(benchmark)[axis]
    return benchmark_mappings


def get_model_mappings(axis, models=None):
    if models is None:
        models = list_models()
    model_mappings = {}
    for model in models:
        if axis is None:
            model_mappings[model] = get_model_info(model)
        else:
            model_mappings[model] = get_model_info(model)[axis]
    return model_mappings


def download_only_aggregate(output_dir):
    print(f"Downloading only aggregate results...{output_dir}")
    hf_hub_download(
        repo_id="haideraltahan/unibench",
        cache_dir=output_dir,
        local_dir=output_dir,
        local_dir_use_symlinks=False,
        repo_type="dataset",
        filename="aggregate.f",
    )


def download_all_results(output_dir):
    print(f"Downloading all results...{output_dir}")
    snapshot_download(
        repo_id="haideraltahan/unibench",
        cache_dir=output_dir,
        local_dir=output_dir,
        local_dir_use_symlinks=False,
        repo_type="dataset",
    )


def rle_to_mask(rle):
    return maskUtils.decode(rle)


def binary_mask_edges(binary_mask):
    kernel = np.array([[1, 1, 1],
                       [1, -8, 1],
                       [1, 1, 1]])

    edges = convolve(binary_mask.astype(np.float32), kernel)
    edges = np.clip(edges, 0, 1)
    return edges.astype(np.uint8)


def load_mask(pkl_file_path, image_shape):
    if os.path.exists(pkl_file_path):
        with open(pkl_file_path, 'rb') as f:
            masks = pickle.load(f)

        shape = masks[0]['segmentation']['size']
        combined_edges = np.zeros(shape, dtype=np.uint8)
        sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)

        for mask_data in sorted_masks:
            segmentation = mask_data['segmentation']
            rle_counts = segmentation['counts']
            binary_mask = rle_to_mask({'size': shape, 'counts': rle_counts})
            # binary_mask = np.clip(binary_mask, 0, 1)
            # edges = cv2.Canny(binary_mask.astype(np.uint8), 1, 1)
            edges = binary_mask_edges(binary_mask)
            combined_edges = np.maximum(combined_edges, edges)
        return combined_edges
    else:
        return np.ones(image_shape[:2], dtype=np.uint8)


def load_DINO_mask(edge_path, image_shape):
    if os.path.exists(edge_path):
        with open(edge_path, 'rb') as f:
            combined_edges = pickle.load(f)
        rle = {'size': combined_edges['size'], 'counts': combined_edges['counts']}
        mask = rle_to_mask(rle)
        return mask
    else:
        return np.ones(image_shape[:2], dtype=np.uint8)


def get_mask_transform(transform):
    resize_size = None
    for t in transform.transforms:
        if isinstance(t, transforms.Resize):
            resize_size = t.size

    mask_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((resize_size, resize_size)) if resize_size else transforms.Resize((224, 224)),
        transforms.Normalize(0.5, 0.26)
    ])

    return mask_transform
