import torch


def index_points(points, idx):
    """
    :param points: points.shape == (B, N, C)
    :param idx: idx.shape == (B, N, K)
    :return:indexed_points.shape == (B, N, K, C)
    """
    raw_shape = idx.shape
    idx = idx.reshape(raw_shape[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.shape[-1]))
    return res.view(*raw_shape, -1)


def knn(a, b, k):
    """
    :param a: a.shape == (B, N, C)
    :param b: b.shape == (B, M, C)
    :param k: int
    """
    inner = -2 * torch.matmul(a, b.transpose(2, 1))  # inner.shape == (B, N, M)
    aa = torch.sum(a**2, dim=-1, keepdim=True)  # aa.shape == (B, N, 1)
    bb = torch.sum(b**2, dim=-1, keepdim=True)  # bb.shape == (B, M, 1)
    pairwise_distance = (
        -aa - inner - bb.transpose(2, 1)
    )  # pairwise_distance.shape == (B, N, M)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # idx.shape == (B, N, K)
    return idx


def inner_production(points):
    """
    :param points: points.shape == (B, N, K=3, C)
    :return res: res.shape == (B, N, K=3)
    """
    points[:, :, 0, :], points[:, :, 1, :], points[:, :, 2, :] = (
        points[:, :, 0, :] * points[:, :, 1, :],
        points[:, :, 1, :] * points[:, :, 2, :],
        points[:, :, 2, :] * points[:, :, 0, :],
    )

    res = torch.sum(points, dim=-1)
    return res


def knn_inner_products(points, k):
    """
    :param points: points.shape == (B, N, C)
    :param k: int
    :return prodocts: products.shape == (B, N, K)
    """
    inner_idx = knn(a=points, b=points, k=k)
    nn_points = index_points(points=points, idx=inner_idx) - points.unsqueeze(2)
    products = inner_production(points=nn_points)

    return products
