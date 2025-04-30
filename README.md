# quantum-close-neighbors

Implementation of "Hybrid Quantum Algorithms for N-Body Simulations" in QCNC 2025.

### Usage 
This project uses the [uv](https://docs.astral.sh/uv/getting-started/) python package manager.

To setup, run the following in the root directory of the repository:
1. `uv venv` - create a virtual environment 
2. `uv pip install .` - installs the current package
   - This makes it so the `quantum_close_neighbors` package within `src` can be imported from your environment
   - If you plan on making edits to the code, install in editable mode: `uv pip install -e .`

To use the notebooks, select the created `.venv` folder as the notebook kernel.

To run individual scripts within the scripts folder use `uv run -m scripts.<script> <script-arguments>`.

### Code Documentation

#### `quantum-close-neighbors` (package within `src`)
This package contains the code used to implement the algorithms discussed in our QCNC paper.
- `grovers`
  - `grovers.py`
    - Defines an abstract interface for Grovers
    - Interface doesn't require actually running the algorithm and returning an element, but rather returns if the measurement after the Grover iterations returned a marked/unmarked and seen/unseen element
  - `probabilistic_grovers.py`
    - Implementation of the abstract Grover interface
    - Does not actually run Grover's algorithm on a simulated quantum computer, but uses computed probablilty of pulling a marked/unmarked and seen/unseen element to implement the abstact Grovers interface
    - Used in the simulations to generate results for our paper, since it is much faster than actually simulating a quantum computer
  - `qiskit_grovers.py`
    - Provides implementations of abstract Grover interface that actually simulate Grover's algorithm with Qiskit
    - Note that qiskit simulations were not used to generate results for of our QCNC paper
    - `QiskitGroversPhaseOracle` simulates Grovers with the oracle implemented as a boolean circuit
          - The oracle used here is a boolean function that is constructed based on the precomputed marked elements (not an actual quantum circuit to lookup particle positions, compute distance and compare if it is less than the threshold)
    - `QiskitGroversStaticPhaseOracle` is an optimized version for the case that the oracle that does not need to be modified (sampling with replacement)
- `close_neighbors.py`
  - Implementation of close neighbors algorithms from our paper
  - Close Neighbors 1 is implemented by the function `close_neighbors`
  - Close Neighbors 2 is implemented by the function `close_neighbors_2`
  - Refer to the paper and comments in the code for details
- `combarro.py`
  - Implementation of algorithms proposed in by Combarro et al. (2023) in the paper "Quantum algorithms to compute the neighbour list of N-body simulations"
  - These are the baseline algorithms used for comparisons in our paper

#### Notebooks
- `benchmarking.ipynb`
  - This notebook was used to generate the results in the QCNC paper
  - For a specified number of particles and number of neighbors per particle, simulates Grovers and generates results for our algorithms and the baseline algorithms
- `parameter_tuning.ipynb`
  - Experiments in hyperparameter tuning for Close Neighbors 1 (but not relevant to the paper results)
- `test_grovers.ipynb` and `test_grovers_p_success.ipynb`
  - These notebooks are used to test that the probabilistic grovers and Qiskit grovers produce the same results
  - Validates that our probabilistic approach used to "simulate" Grovers is correct

#### Scripts (folder within `src`)
Contains executable scripts

- `qiskit_sim_with_replacement.py`
   - Runs close neighbors with replacement using the `QiskitGroversStaticPhaseOracle`
   - Not used to generate paper results
   - Example usage: `uv run -m scripts.qiskit_sim_with_replacement <num_particles> <num_neighbors_per_particle> <path/to/output/directory>`
