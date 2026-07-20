#!/bin/bash --login
#SBATCH -p multicore
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH -o /net/scratch/n94921cs/normet-py/ci_check_%j.out
#SBATCH -e /net/scratch/n94921cs/normet-py/ci_check_%j.err

set -uo pipefail
module purge; unset LD_LIBRARY_PATH PYTHONPATH PYTHONHOME
module load apps/binapps/anaconda3/2024.10
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="/usr/lib64:/lib64:${LD_LIBRARY_PATH:-}"
source activate normet
# ~/.local/bin (pip --user installs of ruff/pytest/mypy) sits ahead of the
# conda env in this login shell's PATH, silently shadowing the env's own
# ruff/pytest/mypy with unrelated versions (e.g. a pytest lacking pytest-cov).
# Force the activated env's bin dir to the front so bare commands resolve
# to it, not ~/.local/bin.
export PATH="${CONDA_PREFIX}/bin:${PATH}"
cd /net/scratch/n94921cs/normet-py

echo "### python: $(python --version)"
echo "### which python: $(which python)"

echo "=== STEP: ruff check src tests ==="
ruff check src tests
RUFF_STATUS=$?
echo "### ruff check exit code: ${RUFF_STATUS}"

if [ "${RUFF_STATUS}" -ne 0 ]; then
  echo
  echo "=== STEP: ruff check --fix --select I001 src tests (import-sort only; other findings left for manual review) ==="
  ruff check --fix --select I001 src tests
  echo "### ruff check --fix exit code: $?"

  echo
  echo "=== STEP: ruff check src tests (after) ==="
  ruff check src tests
  echo "### ruff check exit code (after): $?"
fi

echo
echo "=== STEP: ruff format --check src tests (before) ==="
ruff format --check src tests
FMT_STATUS=$?
echo "### ruff format --check exit code (before): ${FMT_STATUS}"

if [ "${FMT_STATUS}" -ne 0 ]; then
  echo
  echo "=== STEP: ruff format src tests (applying fixes, diff is whitespace-only per manual review) ==="
  ruff format src tests
  echo "### ruff format exit code: $?"

  echo
  echo "=== STEP: ruff format --check src tests (after) ==="
  ruff format --check src tests
  echo "### ruff format --check exit code (after): $?"
fi

echo
echo "=== STEP: pytest --cov=normet --cov-fail-under=70 ==="
MPLBACKEND=Agg PYTENSOR_FLAGS="cxx=" pytest --cov=normet --cov-report=term-missing --cov-fail-under=70
echo "### pytest exit code: $?"

echo
echo "=== STEP: mypy src/normet ==="
mypy src/normet
echo "### mypy exit code: $?"

echo
echo "=== STEP: pip install -e .[docs] ==="
pip install -e ".[docs]"
echo "### pip install docs exit code: $?"

echo
echo "=== STEP: sphinx-build -W docs/ docs/_build/ ==="
sphinx-build -W docs/ docs/_build/
echo "### sphinx-build exit code: $?"

echo
echo "### ALL STEPS COMPLETE ###"
