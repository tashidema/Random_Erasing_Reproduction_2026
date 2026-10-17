"""Figure 3 evaluation; run from the Random-Erasing project directory."""
import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from models.cifar.preact18 import preresnetimagenet


def occlude(images, level, rng):
    images = images.clone()
    if level == 0:
        return images
    height, width = images.shape[-2:]
    for image in images:
        for _ in range(1000):
            ratio = rng.uniform(0.3, 1 / 0.3)
            h = round(math.sqrt(level * height * width * ratio))
            w = round(math.sqrt(level * height * width / ratio))
            if 1 <= h <= height and 1 <= w <= width:
                top = int(rng.integers(0, height - h + 1))
                left = int(rng.integers(0, width - w + 1))
                # Uniform random byte values, converted to the [0, 1] pixel scale.
                patch = rng.integers(0, 256, size=(3, h, w), dtype=np.uint8)
                image[:, top:top+h, left:left+w] = torch.from_numpy(patch).float() / 255
                break
        else:
            raise RuntimeError('Could not sample a fitting occlusion rectangle')
    return images


def load_model(path, device):
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    state = {k.removeprefix('module.'): v for k, v in checkpoint['state_dict'].items()}
    model = preresnetimagenet(num_classes=10)
    model.load_state_dict(state, strict=True)
    print(f'Loaded: {path}', flush=True)
    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=[42])
    parser.add_argument('--data', default='./data')
    parser.add_argument('--output', default='occlusion_results')
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--occlusion-seed', type=int, default=2026)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}', flush=True)
    dataset = CIFAR10(args.data, train=False, download=False, transform=ToTensor())
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010], device=device).view(1, 3, 1, 1)
    for seed in args.seeds:
        models = [load_model(Path(f'cifar10_preact18_{name}_seed{seed}') / 'model_best.pth.tar', device)
                  for name in ['baseline', 'random_erasing']]
        path = output / f'seed{seed}.csv'
        with path.open('w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['seed', 'occlusion_level', 'baseline_accuracy', 'baseline_error',
                             're_accuracy', 're_error'])
            for step in range(11):
                level = step / 20
                # Reset for every model pair: identical test corruption across training seeds.
                rng = np.random.default_rng(args.occlusion_seed + step)
                correct = [0, 0]
                total = 0
                with torch.inference_mode():
                    for images, labels in loader:
                        images = occlude(images, level, rng).to(device)
                        images = (images - mean) / std
                        labels = labels.to(device)
                        for i, model in enumerate(models):
                            correct[i] += (model(images).argmax(1) == labels).sum().item()
                        total += labels.numel()
                acc = [100 * count / total for count in correct]
                error = [100 - value for value in acc]
                writer.writerow([seed, level, acc[0], error[0], acc[1], error[1]])
                file.flush()
                print(f'Seed {seed} | Occlusion {level:.2f} | '
                      f'Baseline error {error[0]:.2f}% | RE error {error[1]:.2f}%', flush=True)
        print(f'Saved: {path}', flush=True)
        del models
    print('Evaluation finished.', flush=True)


if __name__ == '__main__':
    main()
