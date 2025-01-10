import os
from functools import partial

import clip
import open_clip
import alpha_clip
from unibench import Evaluator
from unibench.models_zoo.wrappers.clip import ClipModel, AlphaClipModel
import argparse

# names cannot contain '-

model_config = {
    'clip_rn50': {
        'model_name': 'RN50',
        'tokenizer_name': 'RN50',
        'model_path': '/mnt/shared/unibench/models/clip/RN50.pt',
        'load_type': 'clip',
    },
    'clip_vit_b_16': {
        'model_name': 'ViT-B-16',
        'tokenizer_name': 'ViT-B-16',
        'model_path': '/mnt/shared/unibench/models/clip/ViT-B-16.pt',
        'load_type': 'clip',
    },
    'clip_vit_l_14': {
        'model_name': 'ViT-L-14',
        'tokenizer_name': 'ViT-L-14',
        'model_path': '/mnt/shared/unibench/models/clip/ViT-L-14.pt',
        'load_type': 'clip',
    },
    'clip_vit_l_14_336': {
        'model_name': 'ViT-L-14-336',
        'tokenizer_name': 'ViT-L-14-336',
        'model_path': '/mnt/shared/unibench/models/clip/ViT-L-14-336px.pt',
        'load_type': 'clip',
    },
    'open_clip_convnext_base_w': {
        'model_name': 'convnext_base_w',
        'tokenizer_name': 'convnext_base_w',
        'model_path': '/mnt/shared/unibench/models/open-clip/CLIP-convnext_base_w-laion2B-s13B-b82K/open_clip_pytorch_model.bin',
        'load_type': 'open-clip',
    },
    'open_clip_vit_b_16': {
        'model_name': 'ViT-B-16',
        'tokenizer_name': 'ViT-B-16',
        'model_path': '/mnt/shared/unibench/models/open-clip/CLIP-ViT-B-16-laion2B-s34B-b88K/open_clip_pytorch_model.bin',
        'load_type': 'open-clip',
    },
    'open_clip_vit_l_14': {
        'model_name': 'ViT-L-14',
        'tokenizer_name': 'ViT-L-14',
        'model_path': '/mnt/shared/unibench/models/open-clip/CLIP-ViT-L-14-laion2B-s32B-b82K/open_clip_pytorch_model.bin',
        'load_type': 'open-clip',
    },
    'alpha_clip_vit_b_16': {
        'model_name': '/mnt/shared/unibench/models/clip/ViT-B-16.pt',
        'tokenizer_name': 'ViT-B-16',
        'model_path': '/mnt/shared/unibench/models/alpha-clip/clip_b16_grit20m_fultune_2xe.pth',
        'load_type': 'alpha-clip',
    },
    'alpha_clip_vit_l_14': {
        'model_name': '/mnt/shared/unibench/models/clip/ViT-L-14.pt',
        'tokenizer_name': 'ViT-L-14',
        'model_path': '/mnt/shared/unibench/models/alpha-clip/clip_l14_grit20m_fultune_2xe.pth',
        'load_type': 'alpha-clip',
    },
    'alpha_clip_vit_l_14_336': {
        'model_name': '/mnt/shared/unibench/models/clip/ViT-L-14-336px.pt',
        'tokenizer_name': 'ViT-L-14-336',
        'model_path': '/mnt/shared/unibench/models/alpha-clip/clip_l14_336_grit_20m_4xe.pth',
        'load_type': 'alpha-clip',
    },
    'semantic_clip_rn50': {
        'model_name': '/mnt/shared/unibench/models/clip/RN50.pt',
        'tokenizer_name': 'RN50',
        'model_path': '/mnt/shared/unibench/models/semantic-clip/semantic_clip_rn50.pth',
        'load_type': 'alpha-clip',
    },
    'semantic_clip_convnext_base_w': {
        'model_name': 'convnext_base_w',
        'tokenizer_name': 'convnext_base_w',
        'model_path': '/mnt/shared/unibench/models/semantic-clip/semantic_clip_convnext_base_w.pth',
        'load_type': 'alpha-clip',
    },
    'semantic_clip_vit_b_16_2e_6': {
        'model_name': '/mnt/shared/unibench/models/clip/ViT-B-16.pt',
        'tokenizer_name': 'ViT-B-16',
        'model_path': '/mnt/shared/unibench/models/semantic-clip/semantic_clip_vit_b_16_2e-6.pth',
        'load_type': 'alpha-clip',
    },
    'semantic_clip_vit_b_16_4e_6': {
        'model_name': '/mnt/shared/unibench/models/clip/ViT-B-16.pt',
        'tokenizer_name': 'ViT-B-16',
        'model_path': '/mnt/shared/unibench/models/semantic-clip/semantic_clip_vit_b_16_4e-6.pth',
        'load_type': 'alpha-clip',
    },
    'semantic_clip_vit_b_16_6e_6': {
        'model_name': '/mnt/shared/unibench/models/clip/ViT-B-16.pt',
        'tokenizer_name': 'ViT-B-16',
        'model_path': '/mnt/shared/unibench/models/semantic-clip/semantic_clip_vit_b_16_6e-6.pth',
        'load_type': 'alpha-clip',
    }

}


def get_model_config(name):
    model_name = model_config[name]['model_name']
    tokenizer_name = model_config[name]['tokenizer_name']
    model_path = model_config[name]['model_path']
    load_type = model_config[name]['load_type']
    return model_name, tokenizer_name, model_path, load_type


def load_model(name):
    model_name, tokenizer_name, model_path, load_type = get_model_config(name)
    if load_type == 'clip':
        model, _ = clip.load(model_path)
        input_resolution = model.visual.input_resolution
    elif load_type == 'open-clip':
        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=model_path)
        input_resolution = model.visual.image_size[0]
    elif load_type == 'alpha-clip':
        model, _ = alpha_clip.load(model_name, alpha_vision_ckpt_pth=model_path)
        try:
            # 尝试获取 `input_resolution` 属性
            input_resolution = model.visual.input_resolution
        except AttributeError:
            # 如果 `input_resolution` 不存在，则尝试使用 `image_size[0]`
            if hasattr(model.visual, "image_size"):
                input_resolution = model.visual.image_size[0]
            else:
                raise AttributeError("Neither `input_resolution` nor `image_size` is available in `model.visual`.")
    else:
        raise ValueError(f'Unknown load type: {load_type}')

    tokenizer = open_clip.get_tokenizer(tokenizer_name)
    return partial(
        AlphaClipModel if load_type == 'alpha-clip' else ClipModel,
        model=model,
        model_name=name.replace("-", "_"),
        tokenizer=tokenizer,
        input_resolution=input_resolution,
        logit_scale=model.logit_scale,
    )


def evaluate_models(names, benchmarks=None):
    eval = Evaluator(has_mask=True)

    # update benchmarks
    if benchmarks:
        eval.update_benchmark_list(benchmarks)

    # update models
    for name in names:
        eval.add_model(model=load_model(name))
    eval.update_model_list(names)

    eval.evaluate()
    # eval.show_results()


def main():
    parser = argparse.ArgumentParser(description='Evaluate models on benchmarks.')
    parser.add_argument('--model_ids', nargs='+', required=True, help='List of model names to evaluate')
    args = parser.parse_args()

    model_ids = args.model_ids

    benchmark_list = [
        'coco_order', 'flickr30k_order', 'sugarcrepe', 'vg_attribution', 'vg_relation', 'winoground', ## relation
        'clevr_count', 'clevr_distance', 'dmlab', 'dspr_orientation', 'dspr_x_position', 'dspr_y_position', 'smallnorb_azimuth', 'smallnorb_elevation', ## reasoning
    ]

    # for benchmark in benchmark_list:
    #     for name in names:
    #         res_file = '/mnt/shared/unibench/outputs' + '/' + name + '/' + benchmark + '.f'
    #         if os.path.exists(res_file):
    #             print(f'Result file {res_file} already exists. Skipping evaluation.')
    #             # continue

    #         print(f'Running benchmark: {benchmark}, model: {name}')
    #         evaluate_models([name], benchmarks=[benchmark])

    evaluate_models(model_ids, benchmarks=benchmark_list)


if __name__ == '__main__':
    main()
