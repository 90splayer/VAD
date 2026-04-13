import torch

def topk_mean(scores, k):
    topk, _ = torch.topk(scores, k, dim=1)
    return torch.mean(topk, dim=1)


def unbiased_mil_loss(normal_scores, abnormal_scores, k=5):

    topk_abnormal = topk_mean(abnormal_scores, k)
    topk_normal = topk_mean(normal_scores, k)

    ranking_loss = torch.mean(torch.clamp(1 - topk_abnormal + topk_normal, min=0))

    sparsity_loss = torch.mean(abnormal_scores)

    return ranking_loss + 0.0001 * sparsity_loss