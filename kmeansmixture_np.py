import torch
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import numpy.typing as npt

class KMeansMixture(torch.nn.Module):
    """
    """

    def __init__(self, n_components, n_features):
        super(KMeansMixture, self).__init__()

        self.n_components = n_components
        self.n_features = n_features
        self.mu = torch.nn.Parameter(torch.randn(
            self.n_components, self.n_features), requires_grad=False)

        self._init_params()

    def _init_params(self):
        pass

    def fit(self, x: npt.NDArray, seed, delta=1e-3, n_iter=100, warm_start=False):
        """
        Fits model to the data.
        args:
            x:          torch.Tensor (n, d) or (n, k, d)
        options:
            delta:      float
            n_iter:     int
            warm_start: bool
        """
        kmeans = MiniBatchKMeans(n_clusters=self.n_components,
                                 random_state=seed,
                                 batch_size=16,
                                 n_init="auto")

        kmeans.fit(x)
        self.mu = torch.nn.Parameter(torch.tensor( kmeans.cluster_centers_))

        # cluster_ids_x, cluster_centers = kmeans(
            # X=x, num_clusters=self.n_components,
            # distance='euclidean', device=x.device,
        # )
        # print("CLUSTER SHAPE", cluster_centers.shape)
        # self.mu = torch.nn.Parameter(cluster_centers)

    def check_size(self, x):
        if len(x.size()) == 2:
            # (n, d) --> (n, 1, d)
            x = x.unsqueeze(1)

        return x

    def dot_product(self, x):
        """
        Returns dot product of each sample with each cluster.
        args:
            x:          torch.Tensor (n, d) or (n, 1, d)
        returns:
            y:          torch.Tensor (n, k)
        """
        x = self.check_size(x)

        mu = self.mu.unsqueeze(0).to(x.device)
        return torch.sum(x * mu, dim=-1)

    def distances_squared(self, x):
        """
        Returns squared distances to each cluster.
        args:
            x:          torch.Tensor (n, d) or (n, 1, d)
        returns:
            y:          torch.Tensor (n, k)
        """
        x = self.check_size(x)

        diff = x - self.mu.unsqueeze(0).to(x.device)
        squared_distances = (diff ** 2).sum(-1)
        return squared_distances

    def predict_proba(self, x):
        """
        Returns normalized probabilities of class membership.
        args:
            x:          torch.Tensor (n, d) or (n, 1, d)
        returns:
            y:          torch.LongTensor (n)
        """
        x = self.check_size(x)

        diff = x - self.mu.unsqueeze(0).to(x.device)

        # Calculate squared distances (Euclidean)
        squared_distances = (diff ** 2).sum(-1)
        distances = squared_distances.sqrt()
        inverse_distances = (1.0 / distances)  # .detach().cpu().numpy()
        probabilities = inverse_distances / \
            torch.sum(inverse_distances, dim=1, keepdims=True)
        return probabilities
