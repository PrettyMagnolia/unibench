import os

import pandas as pd
from unibench.benchmarks_zoo.registry import get_benchmark_info, list_benchmarks
from collections import defaultdict
from unibench.common_utils import args

base_dir = '/mnt/shared/unibench/outputs'


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


def show_origin_result(model_ids):
    result = {}
    for name in model_ids:
        res_dir = os.path.join(base_dir, name)
        benchmark_acc_mapping = get_benchmark_acc_mapping(res_dir)

        result[name] = benchmark_acc_mapping
    return result

def main(model_ids):
    # 完整结果
    print(show_origin_result(model_ids))
    # 按照任务分类聚集后的结果
    print(show_aggregated_result(model_ids))


if __name__ == '__main__':
    

    main(args.model_ids)
