# 打开并读取 .f 文件
import os

import pandas as pd
# file_path = '/home/yifei/.cache/unibench/outputs/aggregate.f'


base_dir = '/data2/user_data/yifei/unibench/outputs'

names = [
    # 'clip_vit_b_16',
    # 'clip_vit_l_14',
    # 'clip_vit_l_14_336',
    # 'open_clip-vit-b_16',
    # 'open_clip_vit_l_14',
    # 'alpha_clip_vit_b_16',
    # 'alpha_clip_vit_l_14',
    'alpha_clip_vit_l_14_336',
]

# model_name = 'alpha_clip_vit_l_14'
res_dir = os.path.join(base_dir, names[0])

for file in os.listdir(res_dir):
    print(file)
    content = pd.read_feather(os.path.join(res_dir, file))
    acc = content['correctness'].mean() * 100
    print(f"{acc:.2f}")

# file_path = '/data2/user_data/yifei/unibench/outputs/vit_l_14/sun397.f'  # 替换为你的 .f 文件路径
#
# # 读取文件内容
# content = pd.read_feather(file_path)
#
# # 打印文件内容
# print(content)
#
# # 在 correctness 列上计算平均值
# correctness_mean = content['correctness'].mean()
#
# # 打印 correctness 平均值（即准确率）
# print(f"Correctness Mean (Accuracy): {correctness_mean:.4f}")