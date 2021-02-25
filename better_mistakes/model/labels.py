import torch

from hyptorch.nn import *

def make_all_soft_labels(distances, classes, hardness):
    distance_matrix = torch.Tensor([[distances[c1, c2] for c1 in classes] for c2 in classes])
    max_distance = torch.max(distance_matrix)
    distance_matrix /= max_distance
    soft_labels = torch.exp(-hardness * distance_matrix) / torch.sum(torch.exp(-hardness * distance_matrix), dim=0)

    ##### 将soft-labels装换成在双曲空间上的表示；

    return soft_labels


def make_batch_onehot_labels(target, num_classes, batch_size, gpu):
    onehot_labels = torch.zeros((batch_size, num_classes), dtype=torch.float32).cuda(gpu)
    for i in range(batch_size):
        onehot_labels[i, target[i]] = 1.0
    return onehot_labels


def make_batch_soft_labels(all_soft_labels, target, num_classes, batch_size, gpu):
    soft_labels = torch.zeros((batch_size, num_classes), dtype=torch.float32).cuda(gpu)
    for i in range(batch_size):
        this_label = all_soft_labels[:, target[i]]
        soft_labels[i, :] = this_label

    t = ToPoincare(c=1.0, ball_dim=2)
    h = HyperbolicMLR(ball_dim=2, n_classes=num_classes, c=1.0)

    soft_labels = t.forward(soft_labels)
    soft_labels = h.forward(soft_labels)

    return soft_labels
