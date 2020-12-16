SHELL := /bin/bash
OUT_DIR	= ../build
BUILD_ARGS = -synctex=1 -interaction=nonstopmode -file-line-error -pdf -outdir=${OUT_DIR} --shell-escape
ROOT_FILE = main.tex

build:
	openout_any=a latexmk ${BUILD_ARGS} ${ROOT_FILE}

clean:
	@cd ${OUT_DIR} && rm -f ./*.{fls,fdb_latexmk,synctex.gz,aux,bbl,blg,loa,lof,log,lol,lot,nav,out,snm,toc}

.DEFAULT_GOAL := build
