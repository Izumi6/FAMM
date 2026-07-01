# FAMM: Reproducibility Guide

This document provides step-by-step instructions to reproduce the FAMM experiments, including environment setup, running tests, and generating the exact figures and tables presented in the manuscript.

## Environment Specifications
- **Operating System:** Tested on macOS / Linux
- **Python Version:** 3.11.x (Strictly required for Pydantic v2 compatibility)
- **Primary Dependencies:**
  - `chromadb >= 0.5.0`
  - `sentence-transformers >= 3.0.0`
  - `scikit-learn >= 1.5.0`
  - `pydantic >= 2.7.0`
  - `matplotlib >= 3.9.0`

## Installation Steps

1. **Clone the repository and enter the directory:**
   ```bash
   git clone <repository-url>
   cd "FAMM IEEE"
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate
   ```

3. **Install the package in editable mode with development dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

## Reproducing Tests

To verify the integrity of the framework, run the test suite. Ensure your `PYTHONPATH` includes the root directory.

```bash
PYTHONPATH=. pytest tests/ -v
```
**Expected Output:** 82 tests passed. 

## Reproducing Experiments

The full experiment suite generates all paper figures, the scaled comparison table, and the ablation study table.

1. **Run the experiment script:**
   ```bash
   PYTHONPATH=. python run_full_experiments.py
   ```

2. **Expected Outputs:**
   The script takes ~30-60 seconds (depending on embedding inference speed). 
   Outputs are saved to two directories:
   
   - **`experiments/figures/`**:
     - `fig1_decay_curves.pdf/png`: Adaptive vs uniform decay.
     - `fig2_utility_distribution.pdf/png`: Utility scores by source type.
     - `fig3_comparison_scaled.pdf/png`: Bar chart of baseline comparisons.
     - `fig4_ablation.pdf/png`: Bar chart of ablation results.
     - `fig5_feature_importance.pdf/png`: Heuristic weights breakdown.
     
   - **`experiments/results/`**:
     - `full_experiment_results.json`: Contains the exact numerical data for Table I and Table II in the manuscript.

## Compiling the Manuscript

The manuscript is written in LaTeX using the IEEEtran class.

1. Navigate to the paper directory:
   ```bash
   cd paper
   ```
2. Compile the PDF using `pdflatex` or `latexmk`:
   ```bash
   pdflatex main.tex
   bibtex main
   pdflatex main.tex
   pdflatex main.tex
   ```
3. The resulting `main.pdf` contains the finalized IEEE-formatted manuscript.
