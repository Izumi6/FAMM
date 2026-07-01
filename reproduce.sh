#!/bin/bash
set -e

echo "============================================="
echo "FAMM - Full Reproducibility Pipeline"
echo "============================================="

echo ""
echo "[1/6] Setting up environment & installing dependencies..."
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

echo ""
echo "[2/6] Running Small-Scale Validation & Unit Tests..."
python -m pytest tests/

echo ""
echo "[3/6] Running Consolidation Benchmark..."
python evaluation/benchmark_consolidation.py

echo ""
echo "[4/6] Running Large-Scale Experiments (This will take ~45 minutes)..."
python run_large_scale_experiments.py

echo ""
echo "[5/6] Generating Results and Figures..."
python generate_csv_results.py
python generate_figures.py

echo ""
echo "[6/6] Compiling IEEE Paper..."
cd paper
pdflatex main.tex
pdflatex main.tex
cd ..

echo ""
echo "============================================="
echo "Reproducibility Test Passed Successfully!"
echo "All outputs are in experiments/results/ and paper/figures/"
echo "============================================="
