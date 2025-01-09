import os
import torch
from torch import nn
import open_clip
from functools import partial
from unibench.evaluator import Evaluator
from unibench.models_zoo.wrappers.clip import AlphaClipModel
from unibench.models_zoo.wrappers.glee import GLEEModel
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
        if hasattr(model, 'visual') and hasattr(model.visual, "image_size"):
            input_resolution = model.visual.image_size[0]
        else:
            # raise AttributeError("Neither `input_resolution` nor `image_size` is available in `model.visual`.")
            input_resolution = 224

    model_name = model_name.replace("/", "-")
    # tokenizer = open_clip.get_tokenizer(model_name)
    tokenizer = model.tokenizer
    model_name = (model_name + '_tmp').replace('-', '_')
    model = partial(
        GLEEModel,
        model=model,
        model_name=model_name,
        tokenizer=tokenizer,
        input_resolution=input_resolution,
        logit_scale=nn.Parameter(torch.tensor(4.6052, requires_grad=True)),
    )
    eval.add_model(model=model)
    eval.update_model_list([model_name])

    # delete last outputs
    output_dir = os.path.join(OUTPUT_DIR, model_name)
    if os.path.exists(os.path.join(OUTPUT_DIR, model_name)):
        os.system(f"rm -r {output_dir}")

    eval.evaluate()
