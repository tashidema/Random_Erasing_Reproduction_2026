from pathlib import Path
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

import transforms

random.seed(42)
torch.manual_seed(42)

dataset = CIFAR10(root="./data", train=True, download=False)
pil_image, label = dataset[0]
original = ToTensor()(pil_image)

# Force erase for visual check.
eraser = transforms.RandomErasing(
    probability=1.0,
    sl=0.02,
    sh=0.4,
    r1=0.3,
)
erased = eraser(original.clone())

changed = (erased != original).any(dim=0)
print(f"Class: {dataset.classes[label]}")
print(f"Changed pixels: {changed.sum().item()}")
if not changed.any():
    raise RuntimeError("No pixels changed. Check the RandomErasing transform.")

fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
for ax, tensor, title in zip(
    axes,
    [original, erased],
    ["Original", "Random pixel filling"],
):
    ax.imshow(tensor.permute(1, 2, 0).numpy(), interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")

fig.tight_layout()
output = Path("visual_checks")
output.mkdir(exist_ok=True)
path = output / "cifar10_random_pixel_check_visualcheck.png"
fig.savefig(path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {path}")