#!/bin/bash

# in case the script is not started from within sigmod-repro directory
if [ ! "${PWD}" = "/home/repro/sigmod-repro" ]; then
    cd /home/repro/sigmod-repro/
fi

cd base

echo "Started running IBMQ experiments..."

python3 IBMQExperiments.py
echo "IBMQ experiments done."

cd /home/repro/sigmod-repro/scripts/plotting
echo "Plotting IBMQ results..."
Rscript ibmq_plotting.r
echo "Plotting done."

cd /home/repro