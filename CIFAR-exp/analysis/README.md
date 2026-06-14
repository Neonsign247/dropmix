# CIFAR Analysis Notes

This folder documents the cleaned analysis structure used for the paper metrics.

## Array format

For CIFAR, the notebooks use a recall array with shape:

```text
[num_dropmix_rates, num_seeds, num_classes]
```

Each entry stores per-class recall in percent.

## Vanilla baseline

The original notebook builds the vanilla reference by averaging the vanilla runs collected from the first slice of each method sweep. For a public re-run, the only required input is a 1D array:

```text
[num_classes]
```

containing the final vanilla per-class recall.

If your vanilla sweep has shape `[num_seeds, num_classes]`, you can build this file by averaging over the seed axis and saving the result as a `.npy` array.

## NDC and delta-RDC

The cleaned metric script is:

- `analysis/ndc_rdc.py`

It compares each DropMix-rate setting against the vanilla per-class recall:

- `NDC`: number of classes whose recall decreases
- `delta-RDC`: mean recall change over only the degraded classes

Example:

```bash
python analysis/ndc_rdc.py \
  --recall path/to/cifar_mixup.npy \
  --baseline-recall path/to/cifar_vanilla_recall.npy \
  --reverse-rates
```

Use `--reverse-rates` when the saved sweep is ordered from large to small `dropmix_rate`, which matches the original notebook pattern.
