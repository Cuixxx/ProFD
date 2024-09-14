import collections
from abc import ABC

import numpy as np
import torch
import torch.nn.functional as F
from torch import autograd, nn
from torch.cuda import amp
import tqdm

def extract_image_features(model, cluster_loader, use_amp=False):
    global_image_features = []
    part_image_features = []
    labels = []
    visibilities = []

    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm.tqdm(cluster_loader, desc='Extract image features')):
            img = data['image'].cuda()
            target = data['pid'].cuda()
            masks = data['mask'].cuda() if 'mask' in data else None
            camid = data['camid'].cuda()
            with amp.autocast(enabled=use_amp):
                output = model(img, external_parts_masks=masks)
                global_image_feature = output[0]['bn_globl']
                part_image_feature = output[0]['ex_parts']
                part_visibility = output[1]['ex_parts'][:, 1:]
                for i, gfeat, pfeat, vis in zip(target, global_image_feature, part_image_feature, part_visibility):
                    labels.append(i)
                    visibilities.append(vis)
                    global_image_features.append(gfeat.cpu())
                    part_image_features.append(pfeat.cpu())
    labels_list = torch.stack(labels, dim=0).cuda()
    visibilities_list = torch.stack(visibilities, dim=0).cuda()
    global_image_features_list = torch.stack(global_image_features, dim=0).cuda()  # NC
    part_image_features_list = torch.stack(part_image_features, dim=0).cuda()
    return global_image_features_list, part_image_features_list, labels_list, visibilities_list

def compute_cluster_centroids(features, labels):
    """
    Compute L2-normed cluster centroid for each class.
    """
    num_classes = len(labels.unique()) - 1 if -1 in labels else len(labels.unique())
    centers = torch.zeros((num_classes, features.shape[1]), dtype=torch.float32)
    for i in range(num_classes):
        idx = torch.where(labels == i)[0]
        temp = features[idx,:]
        if len(temp.shape) == 1:
            temp = temp.reshape(1, -1)
        centers[i,:] = temp.mean(0)
    return F.normalize(centers, dim=1)

class CM(autograd.Function):

    @staticmethod
    @amp.custom_fwd
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for x, y in zip(inputs, targets):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None




class CM_Hard(autograd.Function):

    @staticmethod
    @amp.custom_fwd
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)

        for index, features in batch_centers.items():
            distances = []
            for feature in features:
                distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())

            median = np.argmin(np.array(distances))
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * features[median]
            ctx.features[index] /= ctx.features[index].norm()

        return grad_inputs, None, None, None

def cm(inputs, indexes, features, momentum=0.5):
    return CM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))

def cm_hard(inputs, indexes, features, momentum=0.5):
    return CM_Hard.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))

class ClusterMemoryAMP(nn.Module, ABC):
    def __init__(self, temp=0.05, momentum=0.2, use_hard=False):
        super(ClusterMemoryAMP, self).__init__()
        self.momentum = momentum
        self.temp = temp
        self.use_hard = use_hard
        self.features = None
        self.lam = 0.3

    def forward(self, inputs, targets, cams=None, epoch=None, dis_mat =None, pesudo_label = None):
        inputs = F.normalize(inputs, dim=1).cuda()
        if self.use_hard:
            outputs = cm_hard(inputs, targets, self.features, self.momentum)
        else:
            outputs = cm(inputs, targets, self.features, self.momentum)
        if dis_mat is not None:
            outputs += self.lam*dis_mat[targets]
            outputs /= self.temp
        else:
            outputs /= self.temp
        if pesudo_label is not None:
            logit_p = torch.gather(outputs, dim=1, index=pesudo_label)
            loss = torch.mean(torch.logsumexp(outputs, dim=1)-torch.logsumexp(logit_p, dim=1))
        else:
            loss = F.cross_entropy(outputs, targets)
        return loss