#!/bin/bash

# in case the script is not started from within sigmod-repro directory
if [ ! "${PWD}" = "/home/repro/sigmod-repro" ]; then
    cd /home/repro/sigmod-repro/
fi

cd base

echo "Started running DWave experiments..."
python3 DWaveExperiments.py
echo "Dwave experiments done."

cd /home/repro/sigmod-repro/scripts/plotting
echo "Plotting DWave results..."
Rscript dwave_plotting.r
echo "Plotting done."

cd /home/repro
