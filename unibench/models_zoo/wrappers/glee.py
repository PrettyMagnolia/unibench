import torch
from torch import nn
import inspect
from .base import AbstractModel


class GLEEModel(AbstractModel):
    def __init__(
            self,
            model,
            model_name,
            **kwargs,
    ):
        super(GLEEModel, self).__init__(model, model_name, **kwargs)

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
        feature_map = self.model.backbone(images, mask)['res3']
        gap = nn.AdaptiveAvgPool2d(1)
        pooled_features = gap(feature_map)
        image_features = pooled_features.view(pooled_features.size(0), -1)

        image_features /= image_features.norm(dim=1, keepdim=True)
        return image_features.unsqueeze(1)


    @torch.no_grad()
    def get_text_embeddings(self, captions):
        tokenized = self.model.tokenizer.batch_encode_plus(
            captions,
            max_length=self.model.cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN,  # 256
            padding='max_length' if self.model.cfg.MODEL.LANGUAGE_BACKBONE.PAD_MAX else "longest",  # max_length
            return_special_tokens_mask=True,
            return_tensors='pt',
            truncation=True
        ).to("cuda")

        texts = (tokenized['input_ids'], tokenized['attention_mask'])
        caption_embeddings = self.model.text_encoder(*texts)['pooler_output']
        caption_embeddings /= caption_embeddings.norm(dim=-1, keepdim=True)

        return caption_embeddings
