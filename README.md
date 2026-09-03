# 🔐 Cybersecurity NER Robustness Under Multilingual & Obfuscated Text

**A controlled evaluation framework for measuring the robustness of cybersecurity Named Entity Recognition (NER) under multilingual, code-switched, and linguistically perturbed text.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face Datasets](https://img.shields.io/badge/%F0%9F%A4%97-Datasets-ffcc66)](https://huggingface.co/datasets/priamai/AnnoCTR)

---

## 📋 Overview

This project investigates how transformer-based cybersecurity NER models perform when threat intelligence text contains:

- **Multilingual content** (Arabic translations of non-entity words)
- **Code-switching** (systematic language mixing near entities)
- **Controlled obfuscation** (case variation, spacing, Unicode, punctuation)

## 🎯 Research Question

> *How does multilingual, code-switched, and controlled textual obfuscation affect named entity recognition in cybersecurity threat intelligence?*

## 🧪 Experimental Conditions

| Condition | Description |
|-----------|-------------|
| **Clean (C1)** | Original cybersecurity text |
| **Multilingual (C2)** | Text containing multiple languages |
| **Code-switched (C3)** | Languages mixed within the same sentence |
| **Obfuscated (C4)** | Controlled changes to entity surface forms |
| **Normalized (C5)** | Obfuscated/code-switched text after normalization |

## 📁 Project Structure
