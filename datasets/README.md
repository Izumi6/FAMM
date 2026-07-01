# Dataset Provenance

This directory stores benchmark datasets for FAMM evaluation.

## Datasets Used

| Dataset | Source | Purpose |
|:---|:---|:---|
| LoCoMo | [GitHub](https://github.com/snap-stanford/LoCoMo) | Long-term conversational memory evaluation |
| MemoryAgentBench | [OpenReview](https://openreview.net) | Unified memory competency evaluation |
| Memora | [arXiv](https://arxiv.org) | Forgetting-aware memory accuracy |

## Directory Structure

```
datasets/
├── raw/          # Original downloaded data (gitignored)
├── processed/    # Preprocessed for FAMM evaluation harnesses
└── README.md     # This file
```

## Reproduction

To download and preprocess datasets:
```bash
python evaluation/benchmarks/download_datasets.py
```

> **Note:** Raw dataset files are gitignored due to size.
> They must be downloaded separately for reproduction.
