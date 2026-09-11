# Cybersecurity NER Robustness Under Multilingual & Obfuscated Text

**A controlled evaluation framework for measuring the robustness of cybersecurity Named Entity Recognition (NER) under multilingual, code-switched, and linguistically perturbed text.**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22713461.svg)](https://doi.org/10.5281/zenodo.22713461)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dataset: AnnoCTR](https://img.shields.io/badge/Dataset-AnnoCTR-blue)](https://github.com/boschresearch/anno-ctr-lrec-coling-2024)

---

## Preprint

**Omran, R. (2026).** *A Framework for Evaluating Robustness of Cybersecurity Named Entity Recognition: Clean Baseline and Controlled Perturbation Design - Preliminary Report* (Version 2.1). Zenodo.

- **DOI:** [10.5281/zenodo.22713461](https://doi.org/10.5281/zenodo.22713461)
- **Record:** https://zenodo.org/records/22713461

---

## Overview

This project investigates how transformer-based cybersecurity NER models perform when threat intelligence text contains:

- **Multilingual content** (Arabic translations of non-entity words)
- **Code-switching** (systematic language mixing near entities)
- **Controlled obfuscation** (case variation, spacing, Unicode, punctuation)
- **Post-hoc normalization** (Unicode NFC, whitespace collapsing, lowercasing)

## Research Questions

- **RQ1.** How does cybersecurity NER performance change when non-entity context is translated into Arabic while entity spans are preserved?
- **RQ2.** How does Arabic-English code-switching around cybersecurity entities affect NER performance?
- **RQ3.** How do controlled surface-form obfuscations affect cybersecurity NER performance?
- **RQ4.** To what extent can normalization recover performance after controlled perturbations?

> **Scope note.** Version 2.1 of the preprint reports the clean baseline only. The perturbation experiments (RQ1-RQ4) are ongoing and will be reported in a future revision. No perturbation results are claimed at this time.

## Experimental Conditions

| Condition | Description |
|-----------|-------------|
| **Clean (C1)** | Original cybersecurity text |
| **Multilingual (C2)** | Non-entity words translated to Arabic; entity spans preserved |
| **Code-switched (C3)** | Arabic phrases inserted around entities (patterns CS-1, CS-4) |
| **Obfuscated (C4)** | Case randomization, spacing, punctuation, Unicode homoglyphs |
| **Normalized (C5)** | Obfuscated text after Unicode NFC + whitespace + lowercase |

## Results

### Clean baseline (real, verified)

| Metric | Value |
|--------|-------|
| Precision | 0.7369 |
| Recall | 0.8584 |
| **F1** | **0.7931** |

**Setup:** `distilbert-base-uncased`, 500 training examples, 200 validation, 200 test, 1 epoch, batch size 16, max length 128, seed 42, Google Colab CPU. Metric: token-level precision/recall/F1 (weighted, `zero_division=0`).

### Perturbation conditions

Perturbation experiments are in progress. Results will be added in a future revision once the perturbation pipeline passes reproducibility checks (see preprint section 9 for the audit checklist).

## Project Structure
