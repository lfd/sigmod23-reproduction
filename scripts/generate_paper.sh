#!/bin/bash

JOB=paper
INPUT=main.tex
OUTPUT=build

# in case the script is not started from within qsa-repro directory
if [ ! "${PWD}" = "/home/repro/qsa-repro" ]; then
    cd /home/repro/qsa-repro/
fi

cd paper/

echo "started generating paper..."
mkdir -p $OUTPUT
pdflatex --jobname=$OUTPUT/$JOB $INPUT
biber $OUTPUT/$JOB
pdflatex --jobname=$OUTPUT/$JOB $INPUT
pdflatex --jobname=$OUTPUT/$JOB $INPUT
echo "paper generation done."

cd ..
