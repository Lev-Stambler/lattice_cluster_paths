import numpy.typing as npt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import ortho_group

_kernel_feat = None

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

def _inner_product(x, features, kernel_width=None):
    return (features @ np.swapaxes(x, -1, -2)).squeeze(axis=-1)

def make_kernel_feat(n_dims: int):
    global _kernel_feat
    if _kernel_feat is not None:
        return _kernel_feat

    features = np.eye(n_dims)
    features = np.repeat(features, 2, axis=-1)
    features[:, 1::2] *= -1
    features = features.T
    # features = ortho_group.rvs(n_dims) # TODO: WE NEED TO SAVE THIS IF WE USE THIS
    features = np.expand_dims(features, axis=0)
    # assert x.shape[2] == features.shape[2]
    _kernel_feat = features
    return features

def feature_prob(x: npt.NDArray, feature_idx: int, kernel_width=0.01):
    kernel = _inner_product
    x = _check_size(x)
    kern = make_kernel_feat(x.shape[-1])

    # v = -1 if feature_idx % 2 == 1 else 1
    # inner = np.zeros((1, 1, x.shape[-1] * 2))
    # inner[0, 0, feature_idx] = 1
    
    r = kernel(x, kern[:, feature_idx, :], kernel_width)[:, 0]
    return r


# TODO: make kernel width a parameter
# TODO: make the type of kernel used a parm
def predict_proba(x: npt.NDArray, batch_size=-1, kernel_width=0.01):
    kernel = _inner_product

    x = _check_size(x)
    n_dims = x.shape[-1]
    features = make_kernel_feat(n_dims)

    if batch_size > 0:
        out = np.zeros((x.shape[0], features.shape[1]))
        for i in range(0, x.shape[0], batch_size):
            top = min(i + batch_size, x.shape[0])
            out[i:top] = kernel(
                x[i:top], features, kernel_width=kernel_width)
        return out
    return kernel(x, features, kernel_width=kernel_width)
