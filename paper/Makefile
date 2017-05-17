# Shishir Tandale
# tex->pdf Makefile

%: %.tex
	pdflatex $@
	biber $@
	pdflatex $@
	mv $@.pdf out

clean:
	rm *.log
	rm *.aux
	rm *.bcf
	rm *.xml
	rm *.bbl
	rm *.blg
