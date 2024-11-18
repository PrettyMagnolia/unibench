import os
import sys
sys.path.append('/home/yifei/code/AlphaCLIP/unibench')

import open_clip
from functools import partial
from unibench.evaluator import Evaluator
from unibench.models_zoo.wrappers.clip import AlphaClipModel
from unibench.common_utils.constants import OUTPUT_DIR

def evaluate(benchmarks, model, model_name):
    eval = Evaluator(has_mask=True)

    # update benchmarks
    if benchmarks:
        eval.update_benchmark_list(benchmarks)

    try:
        # 尝试获取 `input_resolution` 属性
        input_resolution = model.visual.input_resolution
    except AttributeError:
        # 如果 `input_resolution` 不存在，则尝试使用 `image_size[0]`
        if hasattr(model.visual, "image_size"):
            input_resolution = model.visual.image_size[0]
        else:
            raise AttributeError("Neither `input_resolution` nor `image_size` is available in `model.visual`.")

    model_name = model_name.replace("/", "-")
    tokenizer = open_clip.get_tokenizer(model_name)
    model_name = (model_name + '_tmp').replace('-', '_')
    model = partial(
        AlphaClipModel,
        model=model,
        model_name=model_name,
        tokenizer=tokenizer,
        input_resolution=input_resolution,
        logit_scale=model.logit_scale,
    )
    eval.add_model(model=model)
    eval.update_model_list([model_name])

    # delete last outputs
    output_dir = os.path.join(OUTPUT_DIR, model_name)
    if os.path.exists(os.path.join(OUTPUT_DIR, model_name)):
        os.system(f"rm -r {output_dir}")

    eval.evaluate()
