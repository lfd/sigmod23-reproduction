#!/bin/bash

# in case the script is not started from within sigmod-repro directory
if [ ! "${PWD}" = "/home/repro/sigmod-repro" ]; then
    cd /home/repro/sigmod-repro/
fi

cd base

echo "Started running DB/QPU co-design experiments..."

python3 TranspilationExperiment.py
echo "Co-design experiments done."

cd /home/repro/sigmod-repro/scripts/plotting
echo "Plotting co-design results..."
Rscript codesign_plotting.r
echo "Plotting done."

cd /home/repro