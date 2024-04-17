## Installation weirdness
Using kmeans-pytorch
```bash
pip install git+https://github.com/subhadarship/kmeans_pytorch.git
```

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

## NEXT STEPS
We *NEED* some easier way to both run the tests to give what goes where and also to **auto-ish** generate an interoperability score

Its also not at all clear what makes a path a good "interoperability" path. We need to define this better. I think we can use ideas of "entropy" collapse. I.e. we want to think about how **strongly** a path causes entropy to collapse. I.e. we want to be certain that the path is lit up without having the *whole dataset* light up on said path.

Some smaller things
- Use **normalization**? Should we do this?

### Some cool paths

Path: [[8, 713], [498, 645], [411, 312], [411, 553], [411, 312], [663]] (word preceeding to make things smaller/ larger etc)

It also seems like using the last few layers to differentiate is crucuial as well
Its not at all clear though how to do this...

I have a feeling that we need to just get the highest average weight cliques and use those...