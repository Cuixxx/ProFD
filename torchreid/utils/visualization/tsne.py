import random

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
# 假设你有嵌入向量数据 x 和对应的标签 y
# 这里我使用随机生成的数据作为示例
num_samples = 1000
num_classes = 7
embedding_dim = 50

x = np.random.randn(num_samples, embedding_dim)
y = np.random.randint(0, num_classes, size=num_samples)

# 使用 t-SNE 进行降维


# 可视化
def tsne_vis(x, y):
    x_g = x[:,0].cpu().numpy()
    x_p = x[:,1:].flatten(1).cpu().numpy()
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    x_tsne = tsne.fit_transform(x_p)
    x_min, x_max = x_tsne.min(0), x_tsne.max(0)
    x_norm = (x_tsne - x_min) / (x_max - x_min)  # 归一化
    fig = plt.figure(figsize=(8, 8))
    random.seed(251)
    unique_values, counts = np.unique(y, return_counts=True)
    top_40_indices = np.argsort(counts)[::-1][12:16]
    # random_class = unique_values[top_40_indices].tolist()
    random_class = [6813, 6648, 4726, 5474, 729, 6476, 3, 631]
    for i in random_class:
        plt.scatter(x_norm[y == i, 0], x_norm[y == i, 1], label=f'ID {i}', alpha=0.5)
    # plt.legend(fontsize=14)
    plt.xticks([])
    plt.yticks([])
    plt.savefig("./worse_p.pdf", dpi=600)
    plt.show()
