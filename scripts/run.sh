#!/bin/bash

if [ $# -eq 0 ]; then
	echo "Usage: ./scripts/run.sh [all|ibmq_only|dwave_only|codesign_only|bash]"
	exit 1
fi

# in case the script is not started from within sigmod-repro directory
if [ ! "${PWD}" = "/home/repro/sigmod-repro" ]; then
    cd /home/repro/sigmod-repro/
fi

cd scripts/

if [ "$1" = "all" ]; then
	./run_all.sh
elif [ "$1" = "ibmq_only" ]; then
	./run_ibmq.sh
elif [ "$1" = "dwave_only" ]; then
	./run_dwave.sh
elif [ "$1" = "codesign_only" ]; then
	./run_codesign.sh
elif [ "$1" = "bash" ]; then
	# launch shell
	cd ..
	/bin/bash
	exit 0
else
    echo "Usage: ./scripts/run.sh [all|ibmq_only|dwave_only|codesign_only|bash]"
fi

cd ..

# launch shell
/bin/bash