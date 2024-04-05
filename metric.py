import numpy.typing as npt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import ortho_group

def _check_size(x):
    if len(x.shape) == 2:
        # (batch size, d) --> (bs, 1, d)
        x = np.expand_dims(x, axis=1)
    return x


def _exp_cos_kernel(x, features, kernel_width):
    """
    Returns RBF kernel.
    args:
        x:              torch.Tensor (n, d)
        mu:             torch.Tensor (k, d)
        kernel_width:   float
    returns:
        K_mu_x:         torch.Tensor (n, k)
    """
    # TODO: LOG this for numerical stability??
    # print("SHAPES", x @ , np.linalg.norm(x, axis=-1, keepdims=True).shape, np.linalg.norm(features, keepdims=True).shape)
    # Inner product on last dimension
    inner_p = (features @ np.swapaxes(x, -1, -2)).squeeze(axis=-1)
    cos_inner_prod = inner_p / \
        (np.linalg.norm(x, axis=-1) * np.linalg.norm(features, axis=-1))
    # print("COS INNER PRODUCT", cos_inner_prod)
    # cos_inner_prod = cosine_similarity(x, features)
    exp = np.exp(cos_inner_prod / kernel_width)
    return exp
    normed = exp / np.sum(exp, axis=-1, keepdims=True)
    ret = np.nan_to_num(normed, nan=0.0)
    # print(ret.shape, x.shape, features.shape, cos_inner_prod.shape, exp.shape)
    return ret


# TODO: make kernel width a parameter
def predict_proba(x: npt.NDArray, batch_size=-1, kernel_width=0.1):
    x = _check_size(x)
    n_dims = x.shape[-1]
    features = np.eye(n_dims)
    # features = ortho_group.rvs(n_dims) # TODO: WE NEED TO SAVE THIS IF WE USE THIS
    features = np.expand_dims(features, axis=0)
    assert x.shape[2] == features.shape[2]

    if batch_size > 0:
        out = np.zeros((x.shape[0], features.shape[1]))
        for i in range(0, x.shape[0], batch_size):
            top = min(i + batch_size, x.shape[0])
            out[i:top] = _exp_cos_kernel(
                x[i:top], features, kernel_width=kernel_width)
        return out
    return _exp_cos_kernel(x, features, kernel_width=kernel_width)
