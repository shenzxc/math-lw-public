.PHONY: pdf pdf-latexmk clean status

pdf:
	tectonic -X compile main.tex --outdir outputs

pdf-latexmk:
	latexmk -pdf -interaction=nonstopmode main.tex

status:
	./experiments/status_dashboard.sh

clean:
	rm -f outputs/main.pdf
	rm -f main.aux main.bbl main.blg main.fdb_latexmk main.fls main.log main.out main.pdf
