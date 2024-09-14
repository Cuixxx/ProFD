import torch
import torch.nn as nn
import numpy as np
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from .clip.model import Transformer
_tokenizer = _Tokenizer()
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from collections import OrderedDict
import torch.nn.functional as F
# from visualizer import get_local
from torchreid.utils.constants import *


class BNClassifier(nn.Module):
    # Source: https://github.com/upgirlnana/Pytorch-Person-REID-Baseline-Bag-of-Tricks
    def __init__(self, in_dim, class_num):
        super(BNClassifier, self).__init__()

        self.in_dim = in_dim
        self.class_num = class_num

        self.bn = nn.BatchNorm1d(self.in_dim)
        self.bn.bias.requires_grad_(False)  # BoF: this doesn't have a big impact on perf according to author on github
        self.classifier = nn.Linear(self.in_dim, self.class_num, bias=False)

        self._init_params()

    def forward(self, x):
        feature = self.bn(x)
        cls_score = self.classifier(feature)
        return feature, cls_score

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.001)  # ResNet = 0.01, Bof and ISP-reid = 0.001
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


########################################
#            Pooling heads             #
########################################

def init_part_attention_pooling_head(normalization, pooling, dim_reduce_output):
    if pooling == 'gap':
        parts_attention_pooling_head = GlobalAveragePoolingHead(dim_reduce_output, normalization)
    elif pooling == 'gmp':
        parts_attention_pooling_head = GlobalMaxPoolingHead(dim_reduce_output, normalization)
    elif pooling == 'gwap':
        parts_attention_pooling_head = GlobalWeightedAveragePoolingHead(dim_reduce_output, normalization)
    else:
        raise ValueError('pooling type {} not supported'.format(pooling))
    return parts_attention_pooling_head


class GlobalMaskWeightedPoolingHead(nn.Module):
    def __init__(self, depth, normalization='identity'):
        super().__init__()
        if normalization == 'identity':
            self.normalization = nn.Identity()
        elif normalization == 'batch_norm_3d':
            self.normalization = torch.nn.BatchNorm3d(depth, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        elif normalization == 'batch_norm_2d':
            self.normalization = torch.nn.BatchNorm2d(depth, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        elif normalization == 'batch_norm_1d':
            self.normalization = torch.nn.BatchNorm1d(depth, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        else:
            raise ValueError('normalization type {} not supported'.format(normalization))

    def forward(self, features, part_masks):
        part_masks = torch.unsqueeze(part_masks, 2)
        features = torch.unsqueeze(features, 1)
        parts_features = torch.mul(part_masks, features)

        N, M, _, _, _ = parts_features.size()
        parts_features = parts_features.flatten(0, 1)
        parts_features = self.normalization(parts_features)
        parts_features = self.global_pooling(parts_features)
        parts_features = parts_features.view(N, M, -1)
        return parts_features

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.001)  # ResNet = 0.01, Bof and ISP-reid = 0.001
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class GlobalMaxPoolingHead(GlobalMaskWeightedPoolingHead):
    global_pooling = nn.AdaptiveMaxPool2d((1, 1))


class GlobalAveragePoolingHead(GlobalMaskWeightedPoolingHead):
    global_pooling = nn.AdaptiveAvgPool2d((1, 1))


class GlobalWeightedAveragePoolingHead(GlobalMaskWeightedPoolingHead):
    def forward(self, features, part_masks):
        part_masks = torch.unsqueeze(part_masks, 2)
        features = torch.unsqueeze(features, 1)
        parts_features = torch.mul(part_masks, features)

        N, M, _, _, _ = parts_features.size()
        parts_features = parts_features.flatten(0, 1)
        parts_features = self.normalization(parts_features)
        parts_features = torch.sum(parts_features, dim=(-2, -1))
        part_masks_sum = torch.sum(part_masks.flatten(0, 1), dim=(-2, -1))
        part_masks_sum = torch.clamp(part_masks_sum, min=1e-6)
        parts_features_avg = torch.div(parts_features, part_masks_sum)
        parts_features = parts_features_avg.view(N, M, -1)
        return parts_features

class PromptEncoderWithoutPositionemb(nn.Module):
    def __init__(self, prompt_num=17,
                 transformer_width=512,
                 transformer_heads=8,
                 transformer_layers=1,
                 embed_dim=1024,
                 pretrained=None, **kwargs):
        super().__init__()

        self.pretrained = pretrained

        self.prompt_num = prompt_num

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=None
        )

        self.embed_dim = embed_dim

        # self.positional_embedding = nn.Parameter(torch.empty(self.prompt_num, transformer_width))
        self.ln_final = nn.LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))

    def init_weights(self, pretrained=None):
        return None

    def forward(self, prompt_emb):
        B, K, C = prompt_emb.shape

        x = prompt_emb
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        x = x.reshape(B, K, self.embed_dim)
        return x

class TransformerDecoderLayer(nn.Module):
    def __init__(
            self,
            d_model,
            nhead,
            dropout=0.1,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x, mem):
        q = k = v = self.norm1(x)
        x = x + self.self_attn(q, k, v)[0]
        q = self.norm2(x)
        x = x + self.cross_attn(q, mem, mem)[0]
        x = x + self.dropout(self.mlp(self.norm3(x)))
        return x


class ContextDecoder(nn.Module):
    def __init__(self,
                 transformer_width=256,
                 transformer_heads=4,
                 transformer_layers=6,
                 visual_dim=1024,
                 dropout=0.1,
                 **kwargs):
        super().__init__()

        self.memory_proj = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, transformer_width),
            nn.LayerNorm(transformer_width),
        )

        self.text_proj = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, transformer_width),
        )

        self.decoder = nn.ModuleList([
            TransformerDecoderLayer(transformer_width, transformer_heads, dropout) for _ in range(transformer_layers)
        ])

        self.out_proj = nn.Sequential(
            nn.LayerNorm(transformer_width),
            nn.Linear(transformer_width, visual_dim)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, text, visual):
        visual = self.memory_proj(visual)
        x = self.text_proj(text)

        for layer in self.decoder:
            x = layer(x, visual)

        return self.out_proj(x)

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x



class build_transformer(nn.Module):
    def __init__(self, num_classes, cfg):
        super(build_transformer, self).__init__()
        self.model_cfg = cfg.model.bpbreid
        self.model_name = cfg.model.bpbreid.backbone
        self.parts_num = self.model_cfg.masks.parts_num
        self.num_classes = num_classes
        self.upsample = self.model_cfg.upsample
        # use shared weights/parameters between each part branch for the identity classifier
        self.shared_parts_id_classifier = self.model_cfg.shared_parts_id_classifier
        # use continuous or binary visibility scores at train time:
        self.training_binary_visibility_score = self.model_cfg.training_binary_visibility_score
        # use continuous or binary visibility scores at test time:
        self.testing_binary_visibility_score = self.model_cfg.testing_binary_visibility_score

        if self.model_name == 'ViT-B-16':
            self.in_planes = 768
            self.in_planes_proj = 512
        elif self.model_name == 'RN50':
            self.in_planes = 2048
            self.in_planes_proj = 1024
        self.num_classes = num_classes
        self.sie_coe = cfg.model.bpbreid.SIE_COE

        self.prompt_encoder = PromptEncoderWithoutPositionemb(prompt_num=self.parts_num + 1, transformer_width=512,
                                                              transformer_heads=8, transformer_layers=1, embed_dim=512)
        self.context_decoder = ContextDecoder(transformer_width=256, transformer_heads=4, transformer_layers=3,
                                              visual_dim=512, dropout=0.1)

        # Init pooling layers
        self.dim_reduce_output=512
        self.global_pooling_head = nn.AdaptiveAvgPool2d(1)
        self.foreground_attention_pooling_head = GlobalAveragePoolingHead(self.dim_reduce_output)
        self.background_attention_pooling_head = GlobalAveragePoolingHead(self.dim_reduce_output)
        self.parts_attention_pooling_head = init_part_attention_pooling_head(self.model_cfg.normalization,
                                                                             self.model_cfg.pooling,
                                                                             self.dim_reduce_output)
        # Init id classifier
        self.global_identity_classifier = BNClassifier(self.dim_reduce_output, self.num_classes)
        self.background_identity_classifier = BNClassifier(self.dim_reduce_output, self.num_classes)
        self.foreground_identity_classifier = BNClassifier(self.dim_reduce_output, self.num_classes)
        self.concat_parts_identity_classifier = BNClassifier(self.parts_num * self.dim_reduce_output, self.num_classes)
        if self.shared_parts_id_classifier:
            # the same identity classifier weights are used for each part branch
            self.parts_identity_classifier = BNClassifier(self.dim_reduce_output, self.num_classes)
        else:
            # each part branch has its own identity classifier
            self.parts_identity_classifier = nn.ModuleList(
                [
                    BNClassifier(self.dim_reduce_output, self.num_classes)
                    for _ in range(self.parts_num)
                ]
            )
        #after pooling dim reduce layer
        spatial_feature_size = 768
        self.global_after_pooling_dim_reduce = AfterPoolingDimReduceLayer(spatial_feature_size, self.dim_reduce_output)
        self.foreground_after_pooling_dim_reduce = AfterPoolingDimReduceLayer(spatial_feature_size, self.dim_reduce_output)
        self.background_after_pooling_dim_reduce = AfterPoolingDimReduceLayer(spatial_feature_size, self.dim_reduce_output)
        self.parts_after_pooling_dim_reduce = AfterPoolingDimReduceLayer(spatial_feature_size, self.dim_reduce_output)

        self.h_resolution = int((cfg.data.height - 16) // cfg.model.bpbreid.stride_size[0] + 1)
        self.w_resolution = int((cfg.data.width - 16) // cfg.model.bpbreid.stride_size[1] + 1)
        self.vision_stride_size = cfg.model.bpbreid.stride_size
        clip_model = load_clip_to_cpu(cfg, self.model_name, self.h_resolution, self.w_resolution,
                                      self.vision_stride_size, self.num_classes)
        clip_model.to("cuda")

        self.image_encoder = clip_model.visual

        self.pose_prompt_learner = PromptLearnerPose(cfg, clip_model)
        self.text_encoder = TextEncoder(clip_model)
        scale = 768 ** -0.5
        self.visual_projector = nn.Parameter(scale * torch.randn(768, 768))

    def forward(self, x=None, cam_label=None, view_label=None, external_parts_masks=None):

        if self.model_name == 'RN50':
            image_features_last, image_features, image_features_proj = self.image_encoder(x)
            img_feature_last = nn.functional.avg_pool2d(image_features_last, image_features_last.shape[2:4]).view(
                x.shape[0], -1)
            img_feature = nn.functional.avg_pool2d(image_features, image_features.shape[2:4]).view(x.shape[0], -1)
            img_feature_proj = image_features_proj[0]

        elif self.model_name == 'ViT-B-16':
            if cam_label != None and view_label != None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label * self.view_num + view_label]
            elif cam_label != None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label]
            elif view_label != None:
                cv_embed = self.sie_coe * self.cv_embed[view_label]
            else:
                cv_embed = None

            #Global spatial_features
            image_features_last, image_features, image_features_proj = self.image_encoder(x, cv_embed)
            spatial_features = image_features @ self.visual_projector
            # spatial_features = image_features
            spatial_features = spatial_features[:, 1:, :]
            spatial_image_features_proj = image_features_proj[:, 1:, :]
            N, _, _ = spatial_features.shape

            # Pixels classification and parts attention weights
            pose_prompts = self.pose_prompt_learner()
            prompt_embs = self.text_encoder(pose_prompts, self.pose_prompt_learner.tokenized_prompts).unsqueeze(0).expand(N, -1, -1)
            prompt_embs = self.prompt_encoder(prompt_embs)
            prompt_embs = self.context_decoder(prompt_embs, spatial_image_features_proj)  # enhance pose prompt embeddings
            norm_prompt_embs = prompt_embs / prompt_embs.norm(dim=-1, keepdim=True)
            norm_image_features_proj = spatial_image_features_proj / spatial_image_features_proj.norm(dim=-1, keepdim=True)
            pixels_cls_scores = (torch.matmul(norm_image_features_proj, norm_prompt_embs.permute(0, 2, 1)) * 5).permute(0, 2, 1).reshape(N, self.parts_num+1, self.h_resolution, self.w_resolution)
            upsample_pixels_cls_scores = nn.functional.interpolate(pixels_cls_scores, [64, 32], mode='bilinear',
                                                     align_corners=True)
            pixels_parts_probabilities = F.softmax(pixels_cls_scores, dim=1)


        background_masks = pixels_parts_probabilities[:, 0]
        parts_masks = pixels_parts_probabilities[:, 1:]

        foreground_masks = parts_masks.max(dim=1)[0]
        global_masks = torch.ones_like(foreground_masks)

        # if external_parts_masks is not None:
        #     # hard masking
        #     external_parts_masks = nn.functional.interpolate(external_parts_masks, (16, 8), mode='bilinear',
        #                                                            align_corners=True)

        # Parts visibility
        if (self.training and self.training_binary_visibility_score) or (not self.training and self.testing_binary_visibility_score):
            pixels_parts_predictions = pixels_parts_probabilities.argmax(dim=1)  # [N, Hf, Wf]
            pixels_parts_predictions_one_hot = F.one_hot(pixels_parts_predictions, self.parts_num + 1).permute(0, 3, 1, 2)  # [N, K+1, Hf, Wf]
            parts_visibility = pixels_parts_predictions_one_hot.amax(dim=(2, 3)).to(torch.bool)  # [N, K+1]
        else:
            parts_visibility = pixels_parts_probabilities.amax(dim=(2, 3))  # [N, K+1]

        background_visibility = parts_visibility[:, 0]  # [N]
        foreground_visibility = parts_visibility.amax(dim=1)  # [N]
        parts_visibility = parts_visibility[:, 1:]  # [N, K]
        concat_parts_visibility = foreground_visibility
        global_visibility = torch.ones_like(foreground_visibility)  # [N]

        spatial_features = spatial_features.permute(0, 2, 1).reshape(N, -1, self.h_resolution, self.w_resolution)
        # Global embedding
        # global_embeddings = self.global_pooling_head(spatial_features).view(N, -1)  # [N, D]
        global_embeddings = image_features_proj[:, 0]
        # Foreground and background embeddings
        foreground_embeddings = self.foreground_attention_pooling_head(spatial_features,
                                                                       foreground_masks.unsqueeze(1)).flatten(1,2)  # [N, D]
        background_embeddings = self.background_attention_pooling_head(spatial_features,
                                                                       background_masks.unsqueeze(1)).flatten(1, 2)  # [N, D]
        # Part features
        parts_embeddings = self.parts_attention_pooling_head(spatial_features, parts_masks)  # [N, K, D]

        # global_embeddings = self.global_after_pooling_dim_reduce(global_embeddings)  # [N, D]
        foreground_embeddings = self.foreground_after_pooling_dim_reduce(foreground_embeddings)  # [N, D]
        background_embeddings = self.background_after_pooling_dim_reduce(background_embeddings)  # [N, D]
        parts_embeddings = self.parts_after_pooling_dim_reduce(parts_embeddings)  # [N, M, D]

        # Concatenated part features
        concat_parts_embeddings = parts_embeddings.flatten(1, 2)  # [N, K*D]

        # Identity classification scores
        bn_global_embeddings, global_cls_score = self.global_identity_classifier(
            global_embeddings)  # [N, D], [N, num_classes]
        bn_background_embeddings, background_cls_score = self.background_identity_classifier(
            background_embeddings)  # [N, D], [N, num_classes]
        bn_foreground_embeddings, foreground_cls_score = self.foreground_identity_classifier(
            foreground_embeddings)  # [N, D], [N, num_classes]
        bn_concat_parts_embeddings, concat_parts_cls_score = self.concat_parts_identity_classifier(
            concat_parts_embeddings)  # [N, K*D], [N, num_classes]
        bn_parts_embeddings, parts_cls_score = self.parts_identity_classification(self.dim_reduce_output, N,
                                                                                  parts_embeddings)  # [N, K, D], [N, K, num_classes]

        # Outputs
        embeddings = {
            GLOBAL: global_embeddings,  # [N, D]
            BACKGROUND: background_embeddings,  # [N, D]
            FOREGROUND: foreground_embeddings,  # [N, D]
            CONCAT_PARTS: concat_parts_embeddings,  # [N, K*D]
            PARTS: parts_embeddings,  # [N, K, D]
            BN_GLOBAL: bn_global_embeddings,  # [N, D]
            BN_BACKGROUND: bn_background_embeddings,  # [N, D]
            BN_FOREGROUND: bn_foreground_embeddings,  # [N, D]
            BN_CONCAT_PARTS: bn_concat_parts_embeddings,  # [N, K*D]
            BN_PARTS: bn_parts_embeddings,  #  [N, K, D]
        }

        visibility_scores = {
            GLOBAL: global_visibility,  # [N]
            BACKGROUND: background_visibility,  # [N]
            FOREGROUND: foreground_visibility,  # [N]
            CONCAT_PARTS: concat_parts_visibility,  # [N]
            PARTS: parts_visibility,  # [N, K]
        }

        id_cls_scores = {
            GLOBAL: global_cls_score,  # [N, num_classes]
            BACKGROUND: background_cls_score,  # [N, num_classes]
            FOREGROUND: foreground_cls_score,  # [N, num_classes]
            CONCAT_PARTS: concat_parts_cls_score,  # [N, num_classes]
            PARTS: parts_cls_score,  # [N, K, num_classes]
        }

        masks = {
            GLOBAL: global_masks,  # [N, Hf, Wf]
            BACKGROUND: background_masks,  # [N, Hf, Wf]
            FOREGROUND: foreground_masks,  # [N, Hf, Wf]
            CONCAT_PARTS: foreground_masks,  # [N, Hf, Wf]
            PARTS: parts_masks,  # [N, K, Hf, Wf]
        }
        if self.upsample == False:
            return embeddings, visibility_scores, id_cls_scores, pixels_cls_scores, spatial_features, masks
        else:
            return embeddings, visibility_scores, id_cls_scores, upsample_pixels_cls_scores, spatial_features, masks
    def parts_identity_classification(self, D, N, parts_embeddings):
        if self.shared_parts_id_classifier:
            # apply the same classifier on each part embedding, classifier weights are therefore shared across parts
            parts_embeddings = parts_embeddings.flatten(0, 1)  # [N*K, D]
            bn_part_embeddings, part_cls_score = self.parts_identity_classifier(parts_embeddings)
            bn_part_embeddings = bn_part_embeddings.view([N, self.parts_num, D])
            part_cls_score = part_cls_score.view([N, self.parts_num, -1])
        else:
            # apply K classifiers on each of the K part embedding, each part has therefore it's own classifier weights
            scores = []
            embeddings = []
            for i, parts_identity_classifier in enumerate(self.parts_identity_classifier):
                bn_part_embeddings, part_cls_score = parts_identity_classifier(parts_embeddings[:, i])
                scores.append(part_cls_score.unsqueeze(1))
                embeddings.append(bn_part_embeddings.unsqueeze(1))
            part_cls_score = torch.cat(scores, 1)
            bn_part_embeddings = torch.cat(embeddings, 1)

        return bn_part_embeddings, part_cls_score

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


def clipreid(num_classes, loss='part_based', pretrained=True, config=None, **kwargs):
    model = build_transformer(num_classes, cfg=config)
    return model

from .clip import clip


def load_clip_to_cpu(cfg, backbone_name, h_resolution, w_resolution, vision_stride_size, num_classes):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)
    return model


class PromptLearner(nn.Module):
    def __init__(self, num_class, dtype, token_embedding):
        super().__init__()
        # if dataset_name == "VehicleID" or dataset_name == "veri":
        #     ctx_init = "A photo of a X X X X vehicle."
        # else:
        #     ctx_init = "A photo of a X X X X person."
        ctx_init = "A photo of a X X X X person."

        ctx_dim = 512
        # use given words to initialize context vectors
        ctx_init = ctx_init.replace("_", " ")
        n_ctx = 4

        tokenized_prompts = clip.tokenize(ctx_init).cuda()
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype)
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

        n_cls_ctx = 4
        cls_vectors = torch.empty(num_class, n_cls_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(cls_vectors, std=0.02)
        self.cls_ctx = nn.Parameter(cls_vectors)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])
        self.register_buffer("token_suffix", embedding[:, n_ctx + 1 + n_cls_ctx:, :])
        self.num_class = num_class
        self.n_cls_ctx = n_cls_ctx

    def forward(self, label=None):
        if label == None:
            prefix = self.token_prefix[:, :2, :]
            suffix = self.token_suffix
            stuff = self.token_suffix[:, -1, :].expand(4 + 3, -1).unsqueeze(0)
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    suffix,  # (n_cls, *, dim)
                    stuff,
                ],
                dim=1,
            )
        else:
            cls_ctx = self.cls_ctx[label]
            b = label.shape[0]
            prefix = self.token_prefix.expand(b, -1, -1)
            suffix = self.token_suffix.expand(b, -1, -1)

            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    cls_ctx,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        return prompts


class Agg_PromptLearner(nn.Module):
    def __init__(self, num_class, dataset_name, dtype, token_embedding):
        super().__init__()
        if dataset_name == "VehicleID" or dataset_name == "veri":
            ctx_init = "A photo of a vehicle."
        else:
            ctx_init = "A photo of a person."

        # use given words to initialize context vectors
        ctx_init = ctx_init.replace("_", " ")
        tokenized_prompts = clip.tokenize(ctx_init).cuda()
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype)
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.embedding = embedding

    def forward(self):
        return self.embedding


class ConditionalPromptLearner(nn.Module):
    def __init__(self, num_class, dataset_name, dtype, token_embedding):
        super().__init__()
        if dataset_name == "VehicleID" or dataset_name == "veri":
            ctx_init = "A photo of a X X X X vehicle."
        else:
            ctx_init = "A photo of a X X X X person."

        ctx_dim = 512
        # use given words to initialize context vectors
        ctx_init = ctx_init.replace("_", " ")
        n_ctx = 4

        tokenized_prompts = clip.tokenize(ctx_init).cuda()
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype)
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

        n_cls_ctx = 4
        # cls_vectors = torch.empty(num_class, n_cls_ctx, ctx_dim, dtype=dtype)
        # nn.init.normal_(cls_vectors, std=0.02)
        # self.cls_ctx = nn.Parameter(cls_vectors)
        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(768, 768 // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(768 // 16, ctx_dim * n_cls_ctx))
        ]))
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])
        self.register_buffer("token_suffix", embedding[:, n_ctx + 1 + n_cls_ctx:, :])
        self.num_class = num_class
        self.n_cls_ctx = n_cls_ctx

    def forward(self, img):
        cls_ctx = self.meta_net[img].reshape(img.shape[0], -1, 512)
        b = img.shape[0]
        prefix = self.token_prefix.expand(b, -1, -1)
        suffix = self.token_suffix.expand(b, -1, -1)

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                cls_ctx,  # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        return prompts

class PromptLearnerPose(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        if cfg.model.bpbreid.masks.preprocess == 'eight':
            posenames = ['background', 'head', 'left_arm', 'right_arm', 'torso', 'left_leg', 'right_leg', 'left_feet', 'right_feet']
        elif cfg.model.bpbreid.masks.preprocess == 'five_v':
            posenames = ['background', 'head', 'upper_arms_torso', 'lower_arms_torso', 'legs', 'feet']
        n_cls = len(posenames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        # clip_imsize = clip_model.visual.input_resolution
        # cfg_imsize = cfg.INPUT.SIZE[0]
        # assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        posenames = [name.replace("_", " ") for name in posenames]
        name_lens = [len(_tokenizer.encode(name)) for name in posenames]
        prompts = [prompt_prefix + " " + name + "." for name in posenames]

        tokenized_prompts = torch.cat([clip.tokenize(p).cuda() for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,     # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        return prompts

class AfterPoolingDimReduceLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_p=None):
        super(AfterPoolingDimReduceLayer, self).__init__()
        # dim reduction used in ResNet and PCB
        layers = []
        layers.append(
            nn.Linear(
                input_dim, output_dim, bias=True
            )
        )
        layers.append(nn.BatchNorm1d(output_dim))
        layers.append(nn.ReLU(inplace=True))
        if dropout_p is not None:
            layers.append(nn.dropout(p=dropout_p))

        self.layers = nn.Sequential(*layers)
        self._init_params()

    def forward(self, x):
        if len(x.size()) == 3:
            N, K, _ = x.size()  # [N, K, input_dim]
            x = x.flatten(0, 1)
            x = self.layers(x)
            x = x.view(N, K, -1)
        else:
            x = self.layers(x)
        return x

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)