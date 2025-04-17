import os
from functools import partial

import open_clip
from unibench import Evaluator
from unibench.models_zoo.wrappers.open_clip import OpenClipModel
import argparse
import time
# names cannot contain '-

def main(args):
    benchmarks = [
        # 'imagenet1k', 'cifar10', 'cifar100', 'mnist', # obejct recognition
        'clevr_count_new',
        # 'clevr_count', 'clevr_distance', 
        # 'coco_order', 'flickr30k_order', 'sugarcrepe', 'vg_attribution', 'vg_relation', 'winoground', ## relation
        # 'countbench', 
        # 'dmlab', 'dspr_orientation', 'dspr_x_position', 'dspr_y_position', 
        # 'kitti_distance', 
        # 'smallnorb_azimuth', 'smallnorb_elevation', ## reasoning
    ]

    eval = Evaluator(
        has_mask=args.objects_sense_format is not None,
        test_mode=args.test_mode
    )

    # update benchmarks
    if benchmarks:
        eval.update_benchmark_list(benchmarks)

    # update model
    model, _, _ = open_clip.create_model_and_transforms(args.model_id, pretrained=args.model_path)
    input_resolution = model.visual.image_size[0]
    tokenizer = open_clip.get_tokenizer(args.model_id)

    open_clip_model = partial(
        OpenClipModel,
        model=model,
        model_name=args.exp_name,
        tokenizer=tokenizer,
        input_resolution=input_resolution,
        logit_scale=model.logit_scale,
    )
    eval.add_model(model=open_clip_model)
    eval.update_model_list([args.exp_name])

    eval.evaluate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run OPEN_CLIP evaluation")
    parser.add_argument("--model_id", type=str, default="ViT-B-16", help="Model ID for OPEN_CLIP")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the OPEN_CLIP model checkpoint")
    parser.add_argument("--objects_sense_format", type=str, default=None, help="Format of objects sense")
    # parser.add_argument("--objects_data", type=str, default=None, help="Path to file(s) with objects data")
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name")
    parser.add_argument("--test_mode", type=str, required=False, default=None, help="Experiment name")
    args = parser.parse_args()
    start_time = time.time()
    main(args)
    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")
    
