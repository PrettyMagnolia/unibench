import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from functools import partial

import clip
import open_clip
import alpha_clip
from unibench import Evaluator
from unibench.models_zoo.wrappers.clip import ClipModel, AlphaClipModel

# names cannot contain '-

model_config = {
    'clip_vit_b_16': {
        'model_name': 'ViT-B-16',
        'tokenizer_name': 'ViT-B-16',
        'model_path': '/mnt/user_data/yifei/unibench/models/clip/ViT-B-16.pt',
        'load_type': 'clip',
    },
    'clip_vit_l_14': {
        'model_name': 'ViT-L-14',
        'tokenizer_name': 'ViT-L-14',
        'model_path': '/mnt/user_data/yifei/unibench/models/clip/ViT-L-14.pt',
        'load_type': 'clip',
    },
    'clip_vit_l_14_336': {
        'model_name': 'ViT-L-14-336',
        'tokenizer_name': 'ViT-L-14-336',
        'model_path': '/mnt/user_data/yifei/unibench/models/clip/ViT-L-14-336px.pt',
        'load_type': 'clip',
    },
    'open_clip-vit-b_16': {
        'model_name': 'ViT-B-16',
        'tokenizer_name': 'ViT-B-16',
        'model_path': '/mnt/user_data/yifei/unibench/models/open-clip/CLIP-ViT-B-16-laion2B-s34B-b88K/open_clip_pytorch_model.bin',
        'load_type': 'open-clip',
    },
    'open_clip_vit_l_14': {
        'model_name': 'ViT-L-14',
        'tokenizer_name': 'ViT-L-14',
        'model_path': '/mnt/user_data/yifei/unibench/models/open-clip/CLIP-ViT-L-14-laion2B-s32B-b82K/open_clip_pytorch_model.bin',
        'load_type': 'open-clip',
    },
    'alpha_clip_vit_b_16': {
        'model_name': '/mnt/user_data/yifei/unibench/models/clip/ViT-B-16.pt',
        'tokenizer_name': 'ViT-B-16',
        'model_path': '/mnt/user_data/yifei/unibench/models/alpha-clip/clip_b16_grit20m_fultune_2xe.pth',
        'load_type': 'alpha-clip',

    },
    'alpha_clip_vit_l_14': {
        'model_name': '/mnt/user_data/yifei/unibench/models/clip/ViT-L-14.pt',
        'tokenizer_name': 'ViT-L-14',
        'model_path': '/mnt/user_data/yifei/unibench/models/alpha-clip/clip_l14_grit20m_fultune_2xe.pth',
        'load_type': 'alpha-clip',
    },
    'alpha_clip_vit_l_14_336': {
        'model_name': '/mnt/user_data/yifei/unibench/models/clip/ViT-L-14-336px.pt',
        'tokenizer_name': 'ViT-L-14-336',
        'model_path': '/mnt/user_data/yifei/unibench/models/alpha-clip/clip_l14_336_grit_20m_4xe.pth',
        'load_type': 'alpha-clip',
    },

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
        input_resolution = model.visual.input_resolution
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


def evaluate_models(names, benchmarks=None, eval_dir=None):
    eval = Evaluator(benchmarks_dir=eval_dir) if eval_dir else Evaluator()

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
    names = [
        'clip_vit_b_16',
        'clip_vit_l_14',
        'clip_vit_l_14_336',
        'alpha_clip_vit_b_16',
        'alpha_clip_vit_l_14',
        'alpha_clip_vit_l_14_336',
        # 'open_clip-vit-b_16',
        # 'open_clip_vit_l_14',
    ]

    benchmark_list = [
        'clevr_distance', 'dspr_x_position', 'dspr_y_position', 'dspr_orientation',
        'sun397', 'retinopathy', 'resisc45', 'svhn', 'pets', 'eurosat', 'dtd',
        'dmlab', 'clevr_count', 'cifar100', 'caltech101', 'smallnorb_elevation',
        'pcam', 'smallnorb_azimuth', 'fer2013', 'voc2007', 'mnist', 'country211',
        'fgvc_aircraft', 'cars', 'cifar10', 'imageneta', 'imagenetr', 'imagenete',
        'objectnet', 'imagenet9', 'imagenetv2', 'flickr30k_order', 'sugarcrepe',
        'winoground', 'vg_attribution', 'vg_relation', 'coco_order', 'imagenet',
        'imagenetc'
    ]
    not_support_benchmarks = [
        'dtd',
        'fer2013',
        'imagenete',
        'imagenet9',
        'flickr30k_order',
        'sugarcrepe',
        'winoground',
        # 'vg_attribution',
        # 'vg_relation',
        'coco_order',
        'imagenetc',
    ]
    benchmark_list = [
        # 'vg_attribution', 'vg_relation'
        'imagenet1k'
    ]
    for benchmark in benchmark_list:
        if benchmark in not_support_benchmarks:
            print(f'{benchmark} benchmark is not supported')
            continue
        for name in names:
            res_file = '/mnt/user_data/yifei/unibench/outputs' + '/' + name + '/' + benchmark + '.f'
            if os.path.exists(res_file):
                print(f'Result file {res_file} already exists. Skipping evaluation.')
                # continue

            print(f'Running benchmark: {benchmark}, model: {name}')
            evaluate_models([name], benchmarks=[benchmark])

    # evaluate_models(names, benchmarks=benchmark_list)


if __name__ == '__main__':
    main()
