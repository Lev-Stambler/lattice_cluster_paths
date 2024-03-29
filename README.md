## Installation weirdness
Using kmeans-pytorch
```bash
pip install https://github.com/subhadarship/kmeans_pytorch.git
```
^^ Does not work. Need
```
# https://github.com/subhadarship/kmeans_pytorch/issues/16
            if selected.shape[0] == 0:
                selected = X[torch.randint(len(X), (1,))]s
```
added right after
```
        for index in range(num_clusters):
            selected = torch.nonzero(choice_cluster == index).squeeze().to(device)

            selected = torch.index_select(X, 0, selected)

```