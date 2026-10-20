#!/usr/bin/env bash
# Build the normet manual PDFs: normet_manual.pdf (English, pdflatex) and
# normet_manual_zh.pdf (Chinese, xelatex with ctex). `build.sh en` or `build.sh zh`
# builds one of them.
#
# Passes are run until LaTeX stops asking for another. A fixed count of two can
# silently under-run: text added late in the document shifts page numbers, the second
# pass ends on "Label(s) may have changed. Rerun to get cross-references right." and
# the PDF carries a stale table of contents.
# The .aux is deleted at the end, so each build starts cold and needs the full count.
#
# The figures are screenshots of the real window; make them first with
#     QT_QPA_PLATFORM=offscreen python docs/manual/make_figures.py
#     NORMET_UI_LANGUAGE=zh QT_QPA_PLATFORM=offscreen python docs/manual/make_figures.py
set -e
cd "$(dirname "$0")"

build() {
    local name=$1 engine=$2
    local max_passes=5
    for pass in $(seq 1 $max_passes); do
        if ! "$engine" -interaction=nonstopmode -halt-on-error "$name.tex" > /dev/null; then
            echo "ERROR: $engine stopped on $name.tex:" >&2
            grep -A3 "^!" "$name.log" >&2 || true
            exit 1
        fi
        if ! grep -qE "Rerun to get|Rerun LaTeX" "$name.log"; then
            echo "$name: settled after $pass pass(es)."
            break
        fi
        if [ "$pass" -eq "$max_passes" ]; then
            echo "ERROR: $name: cross-references still unsettled after $max_passes passes." >&2
            exit 1
        fi
    done

    if grep -q "Reference .* undefined" "$name.log"; then
        echo "ERROR: $name: undefined references remain:" >&2
        grep "Reference .* undefined" "$name.log" >&2
        exit 1
    fi

    grep -o "([0-9]* pages" "$name.log" | tail -1 | tr -d '('
    rm -f "$name.aux" "$name.log" "$name.out" "$name.toc"
    echo "Built $name.pdf"
}

which=${1:-all}
if [ "$which" = all ] || [ "$which" = en ]; then
    build normet_manual pdflatex
fi
if [ "$which" = all ] || [ "$which" = zh ]; then
    # The Chinese book needs XeLaTeX and the ctex classes (TeX Live's ctex package; on
    # BasicTeX: sudo tlmgr update --self --all && sudo tlmgr install ctex).
    if ! kpsewhich ctexrep.cls > /dev/null; then
        echo "ERROR: normet_manual_zh needs the ctex package (ctexrep.cls not found)." >&2
        exit 1
    fi
    build normet_manual_zh xelatex
fi
