#!/bin/bash

# in case the script is not started from within qsa-repro directory
if [ ! "${PWD}" = "/home/repro/sigmod-repro" ]; then
    cd /home/repro/sigmod-repro/
fi

cd scripts

# run all IBMQ experiments
./run_ibmq.sh

# run all DWave experiments
./run_dwave.sh

# run all DB-QPU co-design experiments
./run_codesign.sh

cd ..