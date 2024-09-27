"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import inspect

import numpy as np
import torch
import torchvision.transforms as transforms

from .base import AbstractModel


class ClipModel(AbstractModel):
    def __init__(
            self,
            model,
            model_name,
            **kwargs,
    ):
        super(ClipModel, self).__init__(model, model_name, **kwargs)

    def compute_zeroshot_weights(self):
        zeroshot_weights = []
        for class_name in self.classes:
            texts = [template.format(class_name) for template in self.templates]

            class_embedding = self.get_text_embeddings(texts)

            class_embedding = class_embedding.mean(dim=0)
            class_embedding /= class_embedding.norm(dim=-1, keepdim=True)

            zeroshot_weights.append(class_embedding)
        self.zeroshot_weights = torch.stack(zeroshot_weights).T

    @torch.no_grad()
    def get_image_embeddings(self, images):
        image_features = self.model.encode_image(images.to(self.device))
        image_features /= image_features.norm(dim=1, keepdim=True)
        return image_features.unsqueeze(1)

    @torch.no_grad()
    def get_text_embeddings(self, captions):
        if (
                "truncate" in inspect.getfullargspec(self.tokenizer.__call__)[0]
                or "truncate" in inspect.getfullargspec(self.tokenizer)[0]
        ):
            caption_tokens = self.tokenizer(
                captions, context_length=self.context_length, truncate=True
            ).to(self.device)
        else:
            caption_tokens = self.tokenizer(
                captions, context_length=self.context_length
            ).to(self.device)

        caption_embeddings = self.model.encode_text(caption_tokens)
        caption_embeddings /= caption_embeddings.norm(dim=-1, keepdim=True)

        return caption_embeddings


mask_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Resize((224, 224)),
    transforms.Normalize(0.5, 0.26)
])


class AlphaClipModel(AbstractModel):
    def __init__(
            self,
            model,
            model_name,
            **kwargs,
    ):
        super(AlphaClipModel, self).__init__(model, model_name, **kwargs)

    def compute_zeroshot_weights(self):
        zeroshot_weights = []
        for class_name in self.classes:
            texts = [template.format(class_name) for template in self.templates]

            class_embedding = self.get_text_embeddings(texts)

            class_embedding = class_embedding.mean(dim=0)
            class_embedding /= class_embedding.norm(dim=-1, keepdim=True)

            zeroshot_weights.append(class_embedding)
        self.zeroshot_weights = torch.stack(zeroshot_weights).T

    @torch.no_grad()
    def get_image_embeddings(self, images, mask=None):
        # mask = mask_transform(np.ones((images.shape[-2], images.shape[-1])) * 255)
        # mask = mask_transform(np.zeros((images.shape[-2], images.shape[-1])) * 255)
        image_features = self.model.encode_image(images.to(self.device), mask.to(self.device))
        image_features /= image_features.norm(dim=1, keepdim=True)
        return image_features.unsqueeze(1)

    @torch.no_grad()
    def get_text_embeddings(self, captions):
        if (
                "truncate" in inspect.getfullargspec(self.tokenizer.__call__)[0]
                or "truncate" in inspect.getfullargspec(self.tokenizer)[0]
        ):
            caption_tokens = self.tokenizer(
                captions, context_length=self.context_length, truncate=True
            ).to(self.device)
        else:
            caption_tokens = self.tokenizer(
                captions, context_length=self.context_length
            ).to(self.device)

        caption_embeddings = self.model.encode_text(caption_tokens)
        caption_embeddings /= caption_embeddings.norm(dim=-1, keepdim=True)

        return caption_embeddings
