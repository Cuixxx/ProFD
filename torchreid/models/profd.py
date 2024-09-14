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
from .clip.transformer import PositionEmbeddingRandom, TwoWayTransformer


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

    def forward(self, x, mem, mask=None):
        q = k = v = self.norm1(x)
        x = x + self.self_attn(q, k, v)[0]
        q = self.norm2(x)
        x = x + self.cross_attn(q, mem, mem, attn_mask=mask)[0]
        x = x + self.dropout(self.mlp(self.norm3(x)))
        return x

class SemiAttentionDecoderLayer(nn.Module):
    def __init__(
            self,
            d_model,
            nhead,
            dropout=0.1,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.p2t_cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.t2p_cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.distill_attn = nn.MultiheadAttention(d_model, 1, dropout=dropout, batch_first=True)
        self.final_cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.norm5 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.spa_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mem, mask=None):
        # q = k = v = self.norm1(x)
        #self-attn
        quries = x

        #token to proxy absorbing
        keys = mem.clone()
        dk, attn0 = self.t2p_cross_attn(keys, quries, quries)
        keys = keys +dk
        keys = self.norm2(keys)

        #proxy to token absorbing
        x = quries.clone()
        dx, attn1 = self.p2t_cross_attn(quries, mem, mem)
        # x = x + self.p2t_cross_attn(quries, mem, mem)[0]
        x = x + dx
        dx, attn2 = self.distill_attn(self.spa_proj(quries), mem, mem)
        # dx, attn = self.p2t_cross_attn(quries, mem, mem)
        x = x + dx
        x = self.norm3(x)

        #final absorbing
        dx, attn3 = self.final_cross_attn(x, keys, keys)
        x = x + dx
        x = x + self.dropout(self.mlp(self.norm4(x)))
        x = self.norm5(x)

        # return x, torch.sqrt(attn3*attn0.permute(0, 2, 1))+attn1+attn2
        return x, attn2
class PartFeatureDecoder(nn.Module):
    def __init__(self,
                 transformer_layers=2,
                 transformer_width=256,
                 transformer_heads=4,
                 visual_dim=1024,
                 dropout=0.1,
                 ):
        super().__init__()
        self.memory_proj = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, transformer_width),
            nn.LayerNorm(transformer_width),
        )
        self.decoder = nn.ModuleList([
            SemiAttentionDecoderLayer(transformer_width, transformer_heads, dropout) for _ in range(transformer_layers)
        ])

        self.text_proj = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, transformer_width),
        )

        self.out_proj = nn.Sequential(
            nn.LayerNorm(transformer_width),
            nn.Linear(transformer_width, visual_dim),
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
        attn_list = []
        for layer in self.decoder:
            x, attn = layer(x, visual)
            attn_list.append(attn)
        return self.out_proj(x), 0.5 * (attn_list[0] + attn_list[1])

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
    # def __init__(self, num_classes, camera_num, view_num, cfg):
    def __init__(self, num_classes, camera_num, cfg):
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
        # self.camera_num = camera_num
        # self.view_num = view_num
        self.sie_coe = cfg.model.bpbreid.SIE_COE

        self.prompt_encoder = PromptEncoderWithoutPositionemb(prompt_num=self.parts_num + 1, transformer_width=512,
                                                              transformer_heads=8, transformer_layers=1, embed_dim=512)
        self.context_decoder = ContextDecoder(transformer_width=256, transformer_heads=4, transformer_layers=3,
                                              visual_dim=512, dropout=0.1)

        # Init pooling layers
        self.dim_reduce_output = 512
        # spatial_feature_size = 512
        spatial_feature_size = 512
        self.parts_feature_decoder = PartFeatureDecoder(transformer_layers=2, transformer_width=512, transformer_heads=8, visual_dim=512, dropout=0.1)
        # self.spatialdecoder = SpatialDecoder(self.parts_num + 1, transformer_width=512, transformer_heads=8, visual_dim=512, dropout=0.1)
        self.global_pooling_head = nn.AdaptiveAvgPool2d(1)
        self.foreground_attention_pooling_head = GlobalAveragePoolingHead(spatial_feature_size)
        self.background_attention_pooling_head = GlobalAveragePoolingHead(spatial_feature_size)
        self.parts_attention_pooling_head = init_part_attention_pooling_head(self.model_cfg.normalization,
                                                                             self.model_cfg.pooling,
                                                                             spatial_feature_size)
        self.external_attention_pooling_head = init_part_attention_pooling_head(self.model_cfg.normalization,
                                                                                self.model_cfg.pooling,
                                                                                spatial_feature_size)
        # after pooling dim reduce layer

        self.global_after_pooling_dim_reduce = AfterPoolingDimReduceLayer(spatial_feature_size, self.dim_reduce_output)
        self.foreground_after_pooling_dim_reduce = AfterPoolingDimReduceLayer(spatial_feature_size,
                                                                              self.dim_reduce_output)
        self.background_after_pooling_dim_reduce = AfterPoolingDimReduceLayer(spatial_feature_size,
                                                                              self.dim_reduce_output)
        self.parts_after_pooling_dim_reduce = AfterPoolingDimReduceLayer(spatial_feature_size, self.dim_reduce_output)

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

        self.h_resolution = int((cfg.data.height - 16) // cfg.model.bpbreid.stride_size[0] + 1)
        self.w_resolution = int((cfg.data.width - 16) // cfg.model.bpbreid.stride_size[1] + 1)
        self.vision_stride_size = cfg.model.bpbreid.stride_size
        clip_model = load_clip_to_cpu(cfg, self.model_name, self.h_resolution, self.w_resolution,
                                      self.vision_stride_size, self.num_classes)
        clip_model.to("cuda")
        clip_model_image = load_clip_to_cpu(cfg, self.model_name, self.h_resolution, self.w_resolution,
                                            self.vision_stride_size, self.num_classes)
        self.image_encoder = clip_model.visual

        self.pose_prompt_learner = PromptLearnerPose(cfg, clip_model, clip_model_image)
        self.text_encoder = TextEncoder(clip_model)
        self.background_cls = nn.Parameter(torch.randn(1, 512))
        self.IAI = nn.ModuleList([nn.Linear(512,1) for _ in range(self.parts_num)])
        self.DAI = nn.Linear(512,1)
        self.gamma = nn.Parameter(torch.ones(512) * 1e-4)
        self.cv_embed = nn.Parameter(torch.zeros(camera_num, self.in_planes))
        self.seg_classifier = nn.Parameter(torch.randn(6, 512)) # ablation study
        trunc_normal_(self.cv_embed, std=.02)
        print('camera number is : {}'.format(camera_num))

    def forward(self, x=None, cam_label=None, external_parts_masks=None):

        global prompt_embs
        if self.model_name == 'RN50':
            image_features_last, image_features, image_features_proj = self.image_encoder(x)
            img_feature_last = nn.functional.avg_pool2d(image_features_last, image_features_last.shape[2:4]).view(
                x.shape[0], -1)
            img_feature = nn.functional.avg_pool2d(image_features, image_features.shape[2:4]).view(x.shape[0], -1)
            img_feature_proj = image_features_proj[0]

        elif self.model_name == 'ViT-B-16':
            if cam_label != None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label]
            else:
                cv_embed = None

            #Global spatial_features
            image_features_last, image_features, image_features_proj = self.image_encoder(x, cv_embed)
            # spatial_features = image_features @ self.visual_projector
            # spatial_features = image_features[:, 1:, :]
            spatial_image_features_proj = image_features_proj[:, 1:, :]
            spatial_features = spatial_image_features_proj
            N, _, _ = spatial_image_features_proj.shape
            # if self.training:
            #     with torch.no_grad():
            #         _, _, zero_shot_features_proj = self.pose_prompt_learner.ZS_image_encoder(x, cv_embed)
            # Pixels classification and parts attention weights
            pose_prompts = self.pose_prompt_learner()
            prompt_embs = self.text_encoder(pose_prompts, self.pose_prompt_learner.tokenized_prompts).unsqueeze(0).expand(N, -1, -1)
            # prompt_embs = self.seg_classifier.unsqueeze(0).expand(N, -1, -1)
            prompt_embs = torch.cat([self.background_cls.expand(N, -1, -1), prompt_embs], dim=1)
            # pos_src = self.image_pe.expand(N, -1, -1, -1)
            # prompt_embs, spatial_image_features_proj = self.twowaytransformer(image_embedding= spatial_image_features_proj, image_pe= pos_src, point_embedding=prompt_embs)
            # prompt_embs = self.prompt_encoder(prompt_embs)
            # prompt_diff = self.context_decoder(prompt_embs, image_features_proj)  # enhance pose prompt embeddings
            # prompt_embs = prompt_embs + self.gamma * prompt_diff
            norm_prompt_embs = F.normalize(prompt_embs, dim=-1, p=2)
            norm_prompt_embs = torch.cat([norm_prompt_embs[:,0].unsqueeze(1), norm_prompt_embs[:,2:]],dim=1)
            norm_image_features_proj = F.normalize(spatial_image_features_proj, dim=-1, p=2)
            pixels_cls_scores = (torch.matmul(norm_image_features_proj, norm_prompt_embs.permute(0, 2, 1)) * 5).permute(0, 2, 1).reshape(N, self.parts_num+1, self.h_resolution, self.w_resolution)
            upsample_pixels_cls_scores = nn.functional.interpolate(pixels_cls_scores, [64, 32], mode='bilinear', align_corners=True)

        # spatial_features = spatial_image_features_proj.permute(0, 2, 1).reshape(N, -1, self.h_resolution, self.w_resolution)
        spatial_features = spatial_features.permute(0, 2, 1).reshape(N, -1, self.h_resolution, self.w_resolution)
        embeddings, attn_scores = self.parts_feature_decoder(prompt_embs, spatial_image_features_proj)

        # temp_scores = torch.softmax(torch.matmul(norm_image_features_proj, F.normalize(prompt_embs, dim=-1, p=2).permute(0, 2, 1)) * 10, dim=1).permute(0, 2, 1).reshape(N,
        #                                                                                                          self.parts_num + 2,
        #                                                                                                          self.h_resolution,
        #                                                                                                          self.w_resolution)
        # embeddings = self.parts_attention_pooling_head(spatial_features, temp_scores)
        # if external_parts_masks is not None:
        #     # hard masking
        #     spatial_image_features_proj_tmp = spatial_image_features_proj.permute(0, 2, 1).reshape(N, -1, self.h_resolution, self.w_resolution)
        #     external_parts_masks_tmp = F.softmax(64 * F.avg_pool2d(external_parts_masks, kernel_size=(4, 4),stride=(self.vision_stride_size[0]//4,self.vision_stride_size[1]//4)),dim=1)
        #     parts_embeddings = self.parts_attention_pooling_head(spatial_image_features_proj_tmp, external_parts_masks_tmp[:, 1:])  # [N, K, D]

        global_embeddings = image_features_proj[:, 0]
        background_embeddings = embeddings[:, 0]
        foreground_embeddings = embeddings[:, 1]
        parts_embeddings = embeddings[:, 2:]
        # attn_scores = temp_scores # ablation study
        attn_scores = attn_scores.reshape(N, self.parts_num + 2, self.h_resolution, self.w_resolution)
        attn_max = attn_scores.amax(dim=(2, 3)).unsqueeze(2)
        attn_min = attn_scores.amin(dim=(2, 3)).unsqueeze(2)
        attn_scores_01 = ((attn_scores.flatten(2)-attn_min)/(attn_max-attn_min)).reshape(N, -1, self.h_resolution, self.w_resolution)
        background_masks = attn_scores_01[:, 0]
        foreground_masks = attn_scores_01[:, 1]
        parts_masks = attn_scores_01[:, 2:]
        global_masks = torch.ones_like(foreground_masks)

        # Parts visibility
        parts_visibility = torch.sigmoid(torch.cat([item(parts_embeddings[:,i]) for i, item in enumerate(self.IAI)],dim=1))
        # parts_visibility_DAI = parts_visibility+torch.sigmoid(self.DAI(parts_embeddings)).squeeze()
        parts_visibility_DAI = parts_visibility
        parts_visibility_DAI = torch.softmax(parts_visibility_DAI/0.1, dim=1)

        background_visibility = torch.ones_like(background_embeddings[:, 0]) # [N]
        foreground_visibility = torch.ones_like(background_visibility)  # [N]
        concat_parts_visibility = foreground_visibility
        global_visibility = torch.ones_like(foreground_visibility)  # [N]

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
        if external_parts_masks is not None:
            pixels_parts_masks = external_parts_masks.argmax(dim=1)  # [N, Hf, Wf]
            pixels_parts_masks_one_hot = F.one_hot(pixels_parts_masks, self.parts_num + 1).permute(0, 3, 1, 2)  # [N, K+1, Hf, Wf]
            external_parts_visibility = pixels_parts_masks_one_hot.amax(dim=(2, 3)).to(torch.bool)  # [N, K+1]
            # parts_visibility = external_parts_visibility
            # hard masking
            spatial_image_features_proj = spatial_image_features_proj.permute(0, 2, 1).reshape(N, -1, self.h_resolution, self.w_resolution)
            # external_parts_masks = nn.functional.interpolate(external_parts_masks, (self.h_resolution, self.w_resolution), mode='bilinear',
            #                                                        align_corners=True)
            external_parts_masks = F.softmax(64 * F.avg_pool2d(external_parts_masks, kernel_size=(4, 4),stride=(self.vision_stride_size[0]//4,self.vision_stride_size[1]//4)),dim=1)
            # external_foreg_masks = (1-external_parts_masks[:, 0]).unsqueeze(1)
            # external_parts_masks = torch.cat([external_parts_masks[:,0].unsqueeze(1), external_foreg_masks, external_parts_masks[:, 1:]],dim=1)
            external_parts_embeddings = self.parts_attention_pooling_head(spatial_image_features_proj, external_parts_masks[:, 1:])  # [N, K, D]
            bn_external_parts_embeddings, _ = self.concat_parts_identity_classifier(external_parts_embeddings.flatten(1,2))
            # bn_external_parts_embeddings, _ = self.parts_identity_classification(self.dim_reduce_output, N,
            #                                                                           external_parts_embeddings)  # [N, K, D], [N, K, num_classes]

        # Outputs
        if self.training:
            embeddings = {
                GLOBAL: global_embeddings,  # [N, D]
                BACKGROUND: background_embeddings,  # [N, D]
                FOREGROUND: foreground_embeddings,  # [N, D]
                CONCAT_PARTS: concat_parts_embeddings,  # [N, K*D]
                PARTS: parts_embeddings,  # [N, K, D]
                EX_PARTS: external_parts_embeddings,
                TEXT: prompt_embs,
                BN_GLOBAL: bn_global_embeddings,  # [N, D]
                BN_BACKGROUND: bn_background_embeddings,  # [N, D]
                BN_FOREGROUND: bn_foreground_embeddings,  # [N, D]
                BN_CONCAT_PARTS: bn_concat_parts_embeddings,  # [N, K*D]
                BN_PARTS: bn_parts_embeddings,  #  [N, K, D]
                # ZS_FEAT_PROJ: zero_shot_features_proj[:, 0], # [N, D]
            }
        else:
            embeddings = {
                GLOBAL: global_embeddings,  # [N, D]
                BACKGROUND: background_embeddings,  # [N, D]
                FOREGROUND: foreground_embeddings,  # [N, D]
                CONCAT_PARTS: concat_parts_embeddings,  # [N, K*D]
                PARTS: parts_embeddings,  # [N, K, D]
                EX_PARTS: bn_external_parts_embeddings,
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
            EX_PARTS: external_parts_visibility, # [N, K]
            DAI: parts_visibility_DAI
        }
        if self.training:
            id_cls_scores = {
                GLOBAL: global_cls_score,  # [N, num_classes]
                BACKGROUND: background_cls_score,  # [N, num_classes]
                FOREGROUND: foreground_cls_score,  # [N, num_classes]
                CONCAT_PARTS: concat_parts_cls_score,  # [N, num_classes]
                PARTS: parts_cls_score,  # [N, K, num_classes]
                # ZS_FEAT_PROJ: zs_cls_score, # [N, num_classes]
                # SD_GLOBAL: sd_cls_score, # [N, num_classes]
            }
        else:
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
        # if self.upsample == False:
        #     return embeddings, visibility_scores, id_cls_scores, pixels_cls_scores, spatial_features, masks
        # else:
        #     return embeddings, visibility_scores, id_cls_scores, upsample_pixels_cls_scores, spatial_features, masks
        return embeddings, visibility_scores, id_cls_scores, [attn_scores, upsample_pixels_cls_scores], spatial_features, masks
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


def clipreid_bpb(num_classes, camera_num, loss='part_based', pretrained=True, config=None, **kwargs):
    model = build_transformer(num_classes, camera_num, cfg=config)
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

class PromptLearnerPoseExternalKnowledge(nn.Module):
    def __init__(self, cfg, clip_model, clip_model_image):
        super().__init__()
        if cfg.model.bpbreid.masks.preprocess == 'eight':
            posenames = ['background',
                         'head. The head is the uppermost part of the human body, housing the brain and sensory organs like the eyes, nose, ears, and mouth. It typically sits atop the neck and shoulders. A round or oval-shaped structure with facial features',
                         'left_arm. The left arm is one of the two upper limbs of the human body, situated on the left side when facing the individual. It consists of the upper arm, forearm, wrist. A limb extending from the shoulder to the hand on the left side of the body',
                         'right_arm. The right arm is one of the two upper limbs of the human body, positioned on the right side when facing the individual. It consists of the upper arm, forearm, wrist, and hand. A limb extending from the shoulder to the hand on the right side of the body ',
                         'torso. The torso refers to the central part of the human body, excluding the head, neck, arms, and legs. It includes the chest, abdomen, and back. A midsection with a ribcage, abdominal muscles, and spinal column',
                         'left_leg. The left leg is the portion of the lower limb situated on the left side when facing the individual. It consists of the thigh, knee, and shin. A limb extending from the hip to below the knee on the left side of the body',
                         'right_leg. The right leg is the segment of the lower limb positioned on the right side when facing the individual. It encompasses the thigh, knee, and shin. A limb extending from the hip to below the knee on the right side of the body in images or scenes',
                         'left_feet. The left foot is the terminal part of the left leg, positioned on the left side when facing the individual. It consists of the ankle, heel, sole, and toes. A singular appendage situated on the left side of the body',
                         'right_feet. The right foot is the terminal part of the right leg, positioned on the right side when facing the individual. It comprises the ankle, heel, sole, and toes. A singular appendage situated on the right side of the body']
        elif cfg.model.bpbreid.masks.preprocess == 'five_v':
            posenames = ['background',
                        'head. The head is the uppermost part of the human body, housing the brain and sensory organs like the eyes, nose, ears, and mouth. It typically sits atop the neck and shoulders. A round or oval-shaped structure with facial features',
                         'upper_arms_torso. The upper arms torso refers to the upper part of the torso, specifically including the region from the shoulders to the chest. This area encompasses the upper portion of the arms, connecting them to the trunk of the body. The area between the shoulders and chest, where the arms meet the torso',
                         'lower_arms_torso. The lower arms torso is the lower part of the torso, extending from the chest to the waist. The area below the chest, down to the waist, excluding the upper arms',
                         'legs. The legs are the lower limbs of the human body, situated below the torso and above the feet. They consist of the thighs, knees, shins, and calves. The portions extending from the hips to the feet, excluding the feet themselves',
                         'feet. Feet are the terminal part of the lower limbs, located at the bottom of the legs. They consist of the ankle, heel, sole, and toes. The structures at the end of the legs, typically bearing weight and exhibiting features such as toes']
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
            self.ZS_image_encoder = clip_model_image.visual
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

class PromptLearnerPose(nn.Module):
    def __init__(self, cfg, clip_model, clip_model_image):
        super().__init__()
        if cfg.model.bpbreid.masks.preprocess == 'eight':
            posenames = ['people', 'head', 'left_arm', 'right_arm', 'torso', 'left_leg', 'right_leg', 'left_feet', 'right_feet']
        elif cfg.model.bpbreid.masks.preprocess == 'five_v':
            posenames = ['people', 'head', 'upper_arms_torso', 'lower_arms_torso', 'legs', 'feet']
        elif cfg.model.bpbreid.masks.preprocess == 'two_v':
            posenames = ['people', 'torso_arms_head',  'legs']
        elif cfg.model.bpbreid.masks.preprocess == 'three_v':
            posenames = ['people', 'head_mask',  'torso_arms_mask', 'legs_mask']
        elif cfg.model.bpbreid.masks.preprocess == 'four_v':
            posenames = ['people', 'head_mask',  'arms_torso_mask', 'legs_mask', 'feet_mask']
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
            self.ZS_image_encoder = clip_model_image.visual
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

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