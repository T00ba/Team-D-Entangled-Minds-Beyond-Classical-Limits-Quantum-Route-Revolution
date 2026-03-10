# Beyond Classical: Quantum Route Optimization

A hybrid quantum-classical implementation of the **Beyond Classical:Quantum Route Optimization** for solving the Traveling Salesman Problem (TSP) on 150 real-world locations in Islamabad, Pakistan.

---

## Project Report

The full theoretical background and methodology is documented in the project report:
**`Beyond Classical: Quantum Route Optimization`**

It covers:
- Ising encoding of the TSP
- QAOA circuit design with W-state initialization and Grover-inspired mixer
- CI-QAOA architecture and clustering strategy
- Comparative analysis against classical solvers

---

## Team D

| Name |
|------|
| Tooba Bibi |
| Muskan Aman Khan |
| Misbah Riaz |
| Muhammad Ayyaz |
| Naeem Ahmed |

---

## Repository Structure
```
├── qubo.py                  # QUBO formulation of the TSP
├── quantum_circuits.py      # QAOA circuit implementation
├── clustering_solver.py     # Clustered-QAOA (CI-QAOA) solver
├── classical_optimizer.py   # Parameter optimizer for CI-QAOA (COBYLA)
├── classical_solver.py      # Classical TSP solver (baseline comparison)
├── orchestrator.ipynb       # Main notebook — runs full pipeline & results
├── tour_comparison_150cities_20260205_223956.xlsx             # Islamabad locations dataset (150 points)
└── README.md
```

---

## How It Works

The algorithm follows a four-stage pipeline:

1. **QUBO Formulation** (`qubo.py`)
   Encodes the TSP as a Quadratic Unconstrained Binary Optimization problem using Ising encoding. City visits are represented as binary variables and mapped to Pauli-Z operators.

2. **QAOA Circuit** (`quantum_circuits.py`)
   Builds the parameterized quantum circuit with:
   - W-state initialization (partially valid tour superposition)
   - Cost unitary driven by the QUBO Hamiltonian
   - Grover-inspired mixer to stay within feasible tour space

3. **Clustered-QAOA Solver** (`clustering_solver.py`)
   Implements the CI-QAOA divide-and-conquer strategy:
   - Groups 150 locations into small clusters via Agglomerative Hierarchical Clustering
   - Solves a meta-TSP across clusters with QAOA
   - Solves each sub-TSP within clusters with QAOA
   - Assembles the full global tour

4. **Parameter Optimization** (`classical_optimizer.py`)
   Uses **COBYLA** to classically optimize the QAOA parameters (γ, β) by minimizing the expected cost from quantum circuit measurements.

5. **Classical Baseline** (`classical_solver.py`)
   Runs a classical TSP solver on the same dataset for direct performance comparison.

---

## 🚀 Running the Project

### Prerequisites
```bash
pip install qiskit numpy pandas scipy matplotlib 
```

### Run the Full Pipeline

Open and run all cells in:
```
orchestrator.ipynb
```

This notebook:
- Loads the Islamabad dataset from `dataset.xlsx`
- Runs both the CI-QAOA and classical solvers
- Outputs tour distances, approximation ratios, and route visualizations

---

## Dataset

The dataset (`tour_comparison_150cities_20260205_223956.xlsx`) contains **150 real locations in Islamabad, Pakistan**, with latitude and longitude coordinates used to compute pairwise distances for the TSP.

---



## Expected Results

- CI-QAOA solution quality is competitive with classical **Simulated Annealing**
---

## Contact

For questions or collaboration, feel free to reach out via GitHub or toobabb@gmail.com.
