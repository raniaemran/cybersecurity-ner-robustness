# Clean Baseline Results

**Date:** 2026-09-11
**Model:** distilbert-base-uncased
**Dataset:** AnnoCTR (boschresearch/anno-ctr-lrec-coling-2024)
**Train size:** 500 examples
**Validation size:** 200 examples
**Test size:** 200 examples
**Epochs:** 1
**Batch size:** 16
**Max length:** 128
**Hardware:** Google Colab (CPU)
**Metric:** Token-level precision / recall / F1 (weighted, zero_division=0)

## Results

| Condition | Precision | Recall | F1 |
|-----------|-----------|--------|-----|
| Clean English | 0.7369 | 0.8584 | 0.7931 |

## Notes

- Dataset loaded directly from official AnnoCTR GitHub JSON files.
- Hugging Face `load_dataset` was not used because the HF dataset script
  (`AnnoCTR.py`) is no longer supported by `datasets` v3+.
- `seqeval` was replaced with `sklearn.metrics` due to build failures on
  Python 3.13 in Colab.
- Training loss (final): 1.0538
- Validation loss (final): 0.8766
