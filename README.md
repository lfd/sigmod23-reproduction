# Ready to Leap (by Co-Design)? Join Order Optimisation on Quantum Hardware

This repository contains a complete reproduction package, including code and data artifacts, for "Ready to Leap (by Co-Design)? Join Order Optimisation on Quantum Hardware", accepted at SIGMOD 2023.

## Reproduction

### Build Docker Image

```
docker build -t sigmod-repro .
```

### Create Container

```
docker run --name sigmod-repro -it sigmod-repro [<option>]
```
The `<option>` specifies which operations are performed on container start.

Available options are:
* `ibmq_only`: Runs only IBM Q experiments
* `dwave_only`: Runs only D-Wave experiments
* `codesign_only`: Runs only DB-QPU co-design experiments
* `all`: Runs all of the above
* `bash`: Launches an interactive shell

For a full reproduction, we recommend running
```
docker run --name sigmod-repro -it sigmod-repro all
```
Finally, to move the generated plots onto the host system, the following command may be used:
```
docker cp sigmod-repro:/home/repro/sigmod-repro/plots <destination_path>
```

## Project Structure

A readme file with detailed instructions on the reproduction process can be found in the docs directory, alongside a supplementary description (Quantum_Foundations.pdf) on various quantum computing concepts helpful for comprehending our paper (though not strictly necessary).

Licenses required for reproducing some parts of our experiments are to be placed in the licenses folder. Reproduced plots for various figures will be placed in the plots directory, upon completion of the reproduction process. Scripts for running our reproduction and plotting data are contained within the scripts directory.

Source code for our experiments can be found in the base folder, where code for experiments on IBM-Q and D-Wave QPUs is provided in base/IBMQExperiments.py and base/DWaveExperiments.py respectively. The python script base/TranspilationExperiment.py includes code used for analysing the feasibility of join ordering on co-designed custom QPUs.

Query data, as well as experimental results, for both, our experimental analysis on contemporary QPUs (see Sec. 4) and our analysis for quantum circuit transpiled onto co-designed QPUs (see Sec. 6), can be found in the base/ExperimentalAnalysis and base/TranspilationExperiment directories respectively, as described below.

Python scripts for all steps of our JO-QPU approach (e.g., JO-QUBO generation and quantum circuit generation) are contained in the base/Scripts directory. The qubit topology of the contemporary QPUs considered for our analysis can be found in the base/couplings directory.

## Query Data

Input data for the join ordering problem, as considered by our approach for QPUs, consists of relation cardinalities, predicates and their selectivities, and finally, the threshold values used to approximate the intermediate cardinalities. As such, rather than declarative SQL statements, query data is given as follows:

* card.txt contains a list of relation cardinalities, with list indexes corresponding to the respective relations (e.g., the first cardinality corresponds to relation with index 0)
* pred.txt contains a list of join predicates, given again as lists with 2 elements corresponding to the indexes of the respective relations (i.e., [[0, 1], [1, 2]] represents two predicates, the former for relations 0 and 1, and the latter for relations 1 and 2)
* pred_sel.txt contains a list of predicate selectivities, with the list order corresponding to the order of predicates as provided in pred.txt
* thres.txt contains a list of threshold values for approximating intermediate cardinalities

For our experiments on contemporary QPUs, we consider both, realistic queries generated using established methods applied for prior work on join order optimisation, and hand-crafted queries created in such a way that imperfections of contemporary QPUs are minimised. For realistic query loads, we generated a set of queries in accordance to the method described in Steinbrunn et al. [1] (base/ExperimentalAnalysis/DWave/QPUPerformance/Problems/SteinbrunnQueries), and moreover generated queries using the same method and setup as applied in Mancini et al. [2], to compare QPU results against their results for classical state-of-the-art JO solvers (base/ExperimentalAnalysis/DWave/QPUPerformance/Problems/ManciniQueries).

To analyse the effect of QPU imperfections, we hand-crafted a third set of queries with integer logarithmic cardinalities, selectivities and threshold values (base/ExperimentalAnalysis/DWave/QPUPerformance/Problems/IntLogQueries). Such queries allow us to identify the limits of contemporary QPUs, where the required discretisation of continuous cardinalities and selectivities is very costly, independently of QPU discretisation costs.

## References

[1] Michael Steinbrunn, Guido Moerkotte, and Alfons Kemper. 1997. Heuristic and
randomized optimization for the join ordering problem. The VLDB journal 6
(1997), 191–208.

[2] Riccardo Mancini, Srinivas Karthik, Bikash Chandra, Vasilis Mageirakos, and
Anastasia Ailamaki. 2022. Efficient massively parallel join optimization for
large queries. In SIGMOD ’22: International Conference on Management of Data,
Philadelphia, PA, USA, June 12 - 17, 2022, Zachary Ives, Angela Bonifati, and
Amr El Abbadi (Eds.). ACM, 122–135.
