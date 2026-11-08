"""Build the feature files the simulator expects (run once, from the data/ folder).

Outputs (in the current folder), both from the same extractor:
  cifar10_features.npy   (60000 x 512)  ResNet18 (ImageNet weights) penultimate-layer features
  cifar10_labels.npy     (60000,)        CIFAR-10 train + test, 10 classes
  pathmnist_features.npy (60000 x 512)  ResNet18 (ImageNet weights) penultimate-layer features
  pathmnist_labels.npy   (60000,)        random 60000 of the PathMNIST training set, 9 classes

Setup (once):
  pip install numpy torch torchvision medmnist
Run:
  python make_features.py cifar10    # CIFAR-10 only (what is needed now)
  python make_features.py            # both datasets
Downloads happen automatically (CIFAR-10 ~170 MB via torchvision; PathMNIST ~200 MB via medmnist).
On a GPU this takes a few minutes per dataset; on a laptop CPU roughly 20-40 minutes per dataset.
"""
import sys
import numpy as np

N = 60000          # images per dataset
SEED = 0


def resnet_features(X, name):
    """X: N x H x W x 3 uint8 -> N x 512 ResNet18 penultimate-layer features (images resized to 224)."""
    import torch, torchvision
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = torch.nn.Identity(); net.eval().to(dev)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    feats = []
    with torch.no_grad():
        for s in range(0, len(X), 256):
            b = torch.from_numpy(X[s:s + 256]).permute(0, 3, 1, 2).float() / 255
            b = torch.nn.functional.interpolate(b, size=224, mode="bilinear", align_corners=False)
            b = ((b - mean) / std).to(dev)
            feats.append(net(b).cpu().numpy())
            print(f"  {name} {min(s + 256, len(X))}/{len(X)}", end="\r", flush=True)
    return np.concatenate(feats).astype(np.float32)


def cifar10():
    from torchvision.datasets import CIFAR10
    tr = CIFAR10(root="./data", train=True, download=True)
    te = CIFAR10(root="./data", train=False, download=True)
    X = np.concatenate([tr.data, te.data])                    # 60000 x 32 x 32 x 3, uint8
    y = np.concatenate([np.array(tr.targets), np.array(te.targets)])
    F = resnet_features(X, "cifar10")
    np.save("cifar10_features.npy", F); np.save("cifar10_labels.npy", y)
    print("\ncifar10:", F.shape, "classes:", np.unique(y).size)


def pathmnist():
    from medmnist import PathMNIST
    ds = PathMNIST(split="train", download=True, root="./data")
    X = ds.imgs                                               # 89996 x 28 x 28 x 3, uint8
    y = ds.labels.ravel()
    idx = np.random.default_rng(SEED).choice(len(X), N, replace=False)
    F = resnet_features(X[idx], "pathmnist")
    np.save("pathmnist_features.npy", F); np.save("pathmnist_labels.npy", y[idx])
    print("\npathmnist:", F.shape, "classes:", np.unique(y).size)


if __name__ == "__main__":
    which = sys.argv[1:] or ["cifar10", "pathmnist"]
    if "cifar10" in which: cifar10()
    if "pathmnist" in which: pathmnist()
