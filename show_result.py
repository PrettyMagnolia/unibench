import os

import pandas as pd
from unibench.benchmarks_zoo.registry import get_benchmark_info, list_benchmarks
from collections import defaultdict

base_dir = '/mnt/shared/unibench/outputs'

names = [
    # 'clip_vit_b_16',
    # 'clip_vit_l_14',
    # 'clip_vit_l_14_336',
    # 'open_clip-vit_b_16',
    # 'open_clip_vit_l_14',
    # 'alpha_clip_vit_b_16',
    # 'alpha_clip_vit_l_14',
    # 'alpha_clip_vit_l_14_336',
    'semantic_clip_vit_b_16'

]


def get_benchmark_type_mapping():
    benchmark_mapping = {}
    for benchmark in list_benchmarks():
        benchmark_mapping[benchmark] = get_benchmark_info(benchmark)['benchmark_type']
    return benchmark_mapping


def get_benchmark_acc_mapping(file_dir):
    benchmark_acc_mapping = {}
    for files in os.listdir(file_dir):
        if files.endswith('.f'):
            content = pd.read_feather(os.path.join(file_dir, files))
            acc = content['correctness'].mean() * 100
            benchmark = files.split('.')[0]
            benchmark_acc_mapping[benchmark] = acc
    return benchmark_acc_mapping


def get_aggregated_result(benchmark_acc_mapping, benchmark_type_mapping):
    accuracy_by_type = defaultdict(float)
    count_by_type = defaultdict(int)

    for benchmark, accuracy in benchmark_acc_mapping.items():
        benchmark_type = benchmark_type_mapping.get(benchmark)
        if benchmark_type:
            accuracy_by_type[benchmark_type] += accuracy
            count_by_type[benchmark_type] += 1

    average_accuracy_by_type = {bt: accuracy_by_type[bt] / count_by_type[bt] for bt in accuracy_by_type}
    return average_accuracy_by_type


def show_aggregated_result(names):
    result = {}
    benchmark_type_mapping = get_benchmark_type_mapping()
    for name in names:
        res_dir = os.path.join(base_dir, name)
        benchmark_acc_mapping = get_benchmark_acc_mapping(res_dir)

        aggregated_result = get_aggregated_result(benchmark_acc_mapping, benchmark_type_mapping)
        result[name] = aggregated_result
    return result


def show_origin_result(names):
    result = {}
    for name in names:
        res_dir = os.path.join(base_dir, name)
        benchmark_acc_mapping = get_benchmark_acc_mapping(res_dir)

        result[name] = benchmark_acc_mapping
    return result

def main():
    show_origin_result()
    show_aggregated_result()


if __name__ == '__main__':
    main()
