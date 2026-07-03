"""Compile FAMM IEEE paper using local TinyTeX installation."""
import subprocess
import os
from pathlib import Path

paper_dir = Path("paper")
os.chdir(paper_dir)

# Add TinyTeX to PATH
tinytex_bin = os.path.expanduser("~/Library/TinyTeX/bin/universal-darwin")
os.environ["PATH"] = f"{tinytex_bin}:{os.environ['PATH']}"

def run(cmd):
    print(f"  Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        # Check for missing packages
        if "not found" in result.stdout or "not found" in result.stderr:
            missing = result.stdout + result.stderr
            print(f"  Missing package detected. Output:\n{missing[:500]}")
        return False
    return True

print("=== FAMM IEEE Paper Compilation ===\n")

# Full build cycle: pdflatex → bibtex → pdflatex × 2
steps = [
    ("Pass 1: pdflatex", "pdflatex -interaction=nonstopmode main.tex"),
    ("BibTeX", "bibtex main"),
    ("Pass 2: pdflatex", "pdflatex -interaction=nonstopmode main.tex"),
    ("Pass 3: pdflatex", "pdflatex -interaction=nonstopmode main.tex"),
]

for name, cmd in steps:
    print(f"\n[{name}]")
    if not run(cmd):
        print(f"  Warning: {name} had issues, continuing...")

# Check result
pdf_path = Path("main.pdf")
if pdf_path.exists():
    size = pdf_path.stat().st_size
    print(f"\n✓ Success: main.pdf ({size:,} bytes)")
    
    # Count warnings
    log = Path("main.log").read_text()
    undefined = log.count("undefined")
    errors = log.count("! ")
    print(f"  Undefined refs: {undefined}")
    print(f"  Errors: {errors}")
else:
    print("\n✗ Failed: No PDF generated")
