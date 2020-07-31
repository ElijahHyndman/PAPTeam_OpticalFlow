compile:
	python3 setup.py build_ext --inplace

run: clean
	python3 TestSuite.py

all: compile run

clean:
	rm -rf ./output/*
	clear
