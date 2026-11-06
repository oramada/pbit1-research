.PHONY: all figures verify clean
all: main.pdf supplement.pdf
main.pdf: main.tex preamble.tex references.bib $(wildcard figures/*)
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
supplement.pdf: supplement.tex preamble.tex $(wildcard figures/*)
	latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex
figures:
	python tools/build_figures.py
verify: all
	python tools/verify_paper.py
clean:
	latexmk -c main.tex supplement.tex
