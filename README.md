# DropMix

Official implementation of [The Effects of Mixed Sample Data Augmentation are Class Dependent](https://arxiv.org/abs/2307.09136).

## Dataset Preparation

### CIFAR-exp

The CIFAR code supports three dataset names:

- `cifar10`
- `cifar100`
- `tiny-imagenet-200`

#### CIFAR-10 and CIFAR-100

For `cifar10` and `cifar100`, the code uses the torchvision dataset loaders with `download=True`. If you pass a writable `--data_dir`, the dataset will be downloaded automatically on first use.

Practical meaning:

- You do not need to prepare the CIFAR archive manually.
- You only need to set `--data_dir` to a local folder where torchvision can store the dataset.

#### Tiny-ImageNet-200

For `tiny-imagenet-200`, the code expects a manually prepared folder and reads it with `torchvision.datasets.ImageFolder`.

Expected layout:

```text
tiny-imagenet-200/
  train/
    <class_name>/
      images/
      ...
  val/
    images/
    val_annotations.txt
```

The repository also includes a helper that reorganizes validation images into class folders using `val_annotations.txt`.

Relevant file:

- `CIFAR-exp/load_data.py`

After reorganization, the validation directory should be readable through class-specific folders under `val/images`.

### ImageNet-exp

The ImageNet code does not download data automatically. It expects an ImageNet-style folder that is already present on disk.

The training scripts use resized copies of ImageNet:

- `ILSVRC2012-sz/160`
- `ILSVRC2012-sz/352`

Expected layout for each resized copy:

```text
ILSVRC2012-sz/
  160/
    train/
      <class_name>/
        *.JPEG
    val/
      <class_name>/
        *.JPEG
  352/
    train/
      <class_name>/
        *.JPEG
    val/
      <class_name>/
        *.JPEG
```

The original scripts assume an original ImageNet root at:

```text
ILSVRC2012/
  train/
  val/
```

and then create resized copies with:

- `ImageNet-exp/resize.py`

The resize script uses:

- source: `ILSVRC2012`
- destination: `ILSVRC2012-sz`
- output sizes: `160` and `352`

So the practical preparation flow for ImageNet is:

1. Prepare the original ImageNet directory with standard `train/` and `val/` folders.
2. Run `ImageNet-exp/resize.py` after adjusting the source and destination paths for your machine if needed.
3. Use the resulting `160` and `352` roots in the provided shell scripts.

## How `dropmix_rate` Is Controlled

In the paper, `dropmix_rate` means the fraction of data excluded from MSDA. A larger `dropmix_rate` means a larger portion of data is trained without MSDA.

### CIFAR-exp

`dropmix_rate` is exposed directly in `CIFAR-exp/main.py` and passed by the CIFAR shell scripts. During training, a step skips MSDA when a random number is smaller than `dropmix_rate`.

Example:

```bash
bash CIFAR-exp/run_cutmix.sh 0 0.5
```

This runs the CIFAR CutMix-style path with `dropmix_rate=0.5`.

### ImageNet-exp

`dropmix_rate` is defined in the ImageNet YAML config files and used in `cutmix_drop.py` to choose how many samples in the current batch are excluded from CutMix.

The config folder names already encode this value:

- `cutmix_d0.2` -> `dropmix_rate: 0.2`

Example:

```bash
bash ImageNet-exp/shell/cutmix_d0.2.sh
```

This runs the ImageNet DropMix training path using the configs where `dropmix_rate: 0.2`.

## Analysis

- `analysis/ndc_rdc.py`

This script summarizes the two class-level metrics used in the paper:

- `NDC`: number of classes whose recall decreases relative to vanilla
- `delta-RDC`: mean recall change over the degraded classes only

Dataset-specific notes are included here:

- `CIFAR-exp/analysis/README.md`
- `ImageNet-exp/analysis/README.md`

