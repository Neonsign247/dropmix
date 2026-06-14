# ImageNet Analysis Notes

This folder documents the cleaned analysis structure used for the paper metrics.

## Array format

For ImageNet, the original notebook computes per-class recall from two arrays:

```text
correct_classes_all: [num_dropmix_rates, num_seeds, num_classes]
number_classes_all:  [num_dropmix_rates, num_seeds, num_classes]
```

Per-class recall is computed as:

```text
correct_classes_all / number_classes_all * 100
```

## Vanilla baseline

The metric script expects a 1D vanilla per-class recall array:

```text
[num_classes]
```

This matches the role of the baseline used in the original notebook tables.

If you start from vanilla count arrays, first convert them to recall with `correct_classes_all / number_classes_all * 100`, then average over the seed axis and save the resulting `[num_classes]` array.

## NDC and delta-RDC

The cleaned metric script is:

- `analysis/ndc_rdc.py`

It can consume either a ready-made recall array or the original correct/total count arrays.

Example with count arrays:

```bash
python analysis/ndc_rdc.py \
  --correct path/to/correct_classes_all_cutmix.npy \
  --total path/to/number_classes_all_cutmix.npy \
  --baseline-recall path/to/imagenet_vanilla_recall.npy \
  --reverse-rates
```

Here:

- `NDC` counts how many classes degrade relative to vanilla
- `delta-RDC` is the mean recall change over only those degraded classes
