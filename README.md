# QAOA Graph Decomposition

This repository contains the algorithm code, datasets, and benchmarking results for the "Sampling
Matchings with QAOA for Sparse Graph Decomposition" paper: https://arxiv.org/pdf/2509.10657

## Installation

```sh
git clone https://github.com/victor-ibm/QAOA-graph-decomposition.git
conda create -n ENV-NAME python=3.10
conda activate ENV-NAME
pip install -e .
```

The decomposition algorithm requires the IBM CPLEX optimisation software.

You can install the free IBM ILOG CPLEX Optimization Studio Community Edition at
https://www.ibm.com/account/reg/us-en/signup?formid=urx-20028.

Download the correct version for your operating system and follow the installation instructions.
We use the CPLEX python API, so make sure to activate your environment (created using the
installation guide above) and enter the following command:

```sh
python /PATH-TO-CPLEX/CPLEX_Studio_Community2211/python/setup.py install
```

NOTE: CPLEX only supports python up to version 3.10, so make sure you're not using a newer version.

This [script](graph_scheduling/usage_example.py) shows an example of how to use the code. 

## Benchmarking results

You can view the results by entering the following into the terminal:

```sh
hdf5view -f graph_scheduling/results/results.hdf5
```

Alternatively, you can load the data for further manipulation using the python code:

```python
import h5py

with h5py.File("graph_scheduling/results/results.hdf5", "r") as data:
    ...
```

## License

This project uses two different licenses:

- Apache 2.0 for all code.
- CC BY 4.0 for instances and raw data.


## Contributors and acknowledgements


This work is a collaboration between The Hartree Centre STFC, United Kingdom (1), E.ON Digital
Technology GmbH, Essen, Germany (2), and IBM Quantum and IBM Research Europe — Dublin, Ireland (3).

**Contributors:** George Pennington (1), Naeimeh Mohseni (2), Oscar Wallis (1), Francesca Schiavello
(1), Stefano Mensa (1), Corey O'Meara (2), Giorgio Cortiana (2), and Víctor Valls (3).

This work was supported by the Hartree National Centre for Digital Innovation, a UK
Government-funded collaboration between STFC and IBM.
