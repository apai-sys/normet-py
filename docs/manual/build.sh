#!/usr/bin/env bash
# Build the normet manual PDF.
#
# Passes are run until LaTeX stops asking for another. A fixed count of two can
# silently under-run: text added late in the document shifts page numbers, the second
# pass ends on "Label(s) may have changed. Rerun to get cross-references right." and
# the PDF carries a stale table of contents.
# The .aux is deleted at the end, so each build starts cold and needs the full count.
#
# The figures are screenshots of the real window; make them first with
#     QT_QPA_PLATFORM=offscreen python docs/manual/make_figures.py
set -e
cd "$(dirname "$0")"

max_passes=5
for pass in $(seq 1 $max_passes); do
    pdflatex -interaction=nonstopmode -halt-on-error normet_manual.tex > /dev/null
    if ! grep -qE "Rerun to get|Rerun LaTeX" normet_manual.log; then
        echo "Settled after $pass pass(es)."
        break
    fi
    if [ "$pass" -eq "$max_passes" ]; then
        echo "ERROR: cross-references still unsettled after $max_passes passes." >&2
        exit 1
    fi
done

if grep -q "Reference .* undefined" normet_manual.log; then
    echo "ERROR: undefined references remain:" >&2
    grep "Reference .* undefined" normet_manual.log >&2
    exit 1
fi

grep -o "([0-9]* pages" normet_manual.log | tail -1 | tr -d '('
rm -f normet_manual.aux normet_manual.log normet_manual.out normet_manual.toc
echo "Built normet_manual.pdf"
