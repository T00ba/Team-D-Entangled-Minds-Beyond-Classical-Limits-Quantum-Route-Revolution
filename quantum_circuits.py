# quantum_circuits.py
"""
QAOA implementation for TSP.
Based on: "Quantum Approaches to Urban Logistics" paper.
"""

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import MCPhaseGate
import numpy as np
from typing import Tuple, Optional


class TSPQAOA:
    """QAOA implementation for TSP."""
    
    def __init__(self, n_cities: int):
        """
        Initialize QAOA for TSP.
        
        Args:
            n_cities: Number of cities
        """
        self.n_cities = n_cities
        self.n_vars = n_cities * n_cities
    
    def create_initial_state(self) -> QuantumCircuit:
        """
        Create initial state: ⊗_{t=1}^n |D_1^n⟩.
        Each time register in uniform superposition of one-hot states.
        """
        qc = QuantumCircuit(self.n_vars, name="initial_state")
        
        for t in range(self.n_cities):
            start = t * self.n_cities
            
            if self.n_cities == 1:
                qc.x(start)
            elif self.n_cities == 2:
                qc.h(start)
                qc.cx(start, start + 1)
                qc.x(start)
            elif self.n_cities == 3:
                # W3 state preparation using your specified method
                # Start with |000⟩ at positions: start, start+1, start+2
                # 1. Apply Ry on first qubit
                angle = 2 * np.arccos(1 / np.sqrt(3))
                qc.ry(angle, start)
                
                # 2. Apply Control Hadamard where first qubit is control and second is target
                # Using Qiskit's controlled-Hadamard gate
                qc.ch(start, start + 1)  # control=first, target=second
                
                # 3. Apply CNOT on second and third qubit where second is control and third is target
                qc.cx(start + 1, start + 2)
                
                # 4. Apply CNOT on first and second qubit where first is control and second is target
                qc.cx(start, start + 1)
                
                # 5. Apply X operator on first qubit
                qc.x(start)
            else:
                # Original W-state preparation for n_cities > 3
                qc.h(start)
                for i in range(1, self.n_cities):
                    angle = 2 * np.arccos(1 / np.sqrt(self.n_cities - i + 1))
                    target = start + i
                    qc.ry(angle, target)
                    qc.cx(target, start)
        
        return qc
    
    def create_cost_operator(self, Q: np.ndarray, gamma: Parameter) -> QuantumCircuit:
        """
        Create cost operator: exp(-iγ H_C).
        
        Args:
            Q: QUBO matrix
            gamma: γ parameter
        
        Returns:
            Quantum circuit implementing cost operator
        """
        qc = QuantumCircuit(self.n_vars, name="cost_op")
        
        # Precompute single-qubit coefficients
        single_coeffs = np.zeros(self.n_vars)
        
        for i in range(self.n_vars):
            # Contribution from diagonal
            single_coeffs[i] += Q[i, i] / 4
            
            # Contributions from off-diagonal
            for j in range(self.n_vars):
                if i != j:
                    single_coeffs[i] -= Q[i, j] / 4
        
        # Apply single-qubit Rz gates
        for i in range(self.n_vars):
            coeff = single_coeffs[i]
            if abs(coeff) > 1e-12:
                qc.rz(-2 * gamma * coeff, i)
        
        # Apply two-qubit terms
        for i in range(self.n_vars):
            for j in range(i + 1, self.n_vars):
                coeff = Q[i, j] / 4
                if abs(coeff) > 1e-12:
                    qc.cx(i, j)
                    qc.rz(2 * gamma * coeff, j)
                    qc.cx(i, j)
        
        return qc
    
    def create_grover_mixer(self, beta: Parameter) -> QuantumCircuit:
        """
        Create Grover mixer operator.
        
        Args:
            beta: β parameter
        
        Returns:
            Quantum circuit implementing Grover mixer
        """
        qc = QuantumCircuit(self.n_vars, name="grover_mixer")
        
        for t in range(self.n_cities):
            qubits = list(range(t * self.n_cities, (t + 1) * self.n_cities))
            
            if self.n_cities == 1:
                qc.rz(-beta, qubits[0])
            elif self.n_cities == 2:
                qc.p(-beta/2, qubits[0])
                qc.p(-beta/2, qubits[1])
                qc.cx(qubits[0], qubits[1])
                qc.p(beta/2, qubits[1])
                qc.cx(qubits[0], qubits[1])
            else:
                qc.h(qubits)
                mcp = MCPhaseGate(beta, num_ctrl_qubits=self.n_cities-1)
                qc.append(mcp, qubits)
                qc.h(qubits)
        
        return qc
    
    def construct_circuit(
        self,
        Q: np.ndarray,
        p: int = 1,
        add_measurement: bool = True
    ) -> Tuple[QuantumCircuit, ParameterVector, ParameterVector]:
        """
        Construct complete QAOA circuit.
        
        Args:
            Q: QUBO matrix
            p: Number of QAOA layers
            add_measurement: Whether to add measurement gates
        
        Returns:
            (circuit, gammas, betas) where:
            - circuit: Complete QAOA circuit
            - gammas: γ parameters vector
            - betas: β parameters vector
        """
        # Create registers
        qr = QuantumRegister(self.n_vars, 'q')
        cr = ClassicalRegister(self.n_vars, 'c') if add_measurement else None
        
        if add_measurement:
            qc = QuantumCircuit(qr, cr, name=f"QAOA_TSP_{self.n_cities}_p{p}")
        else:
            qc = QuantumCircuit(qr, name=f"QAOA_TSP_{self.n_cities}_p{p}")
        
        # Create parameters
        gammas = ParameterVector('γ', p)
        betas = ParameterVector('β', p)
        
        # Initial state
        init_circuit = self.create_initial_state()
        qc.compose(init_circuit, qubits=qr[:], inplace=True)
        
        # QAOA layers
        for layer in range(p):
            # Cost operator
            cost_circuit = self.create_cost_operator(Q, gammas[layer])
            qc.compose(cost_circuit, qubits=qr[:], inplace=True)
            
            # Mixer operator
            mixer_circuit = self.create_grover_mixer(betas[layer])
            qc.compose(mixer_circuit, qubits=qr[:], inplace=True)
        
        # Measurement
        if add_measurement:
            qc.measure(qr, cr)
        
        return qc, gammas, betas


# Helper function for backward compatibility
def construct_qaoa_circuit(n_cities: int, Q: np.ndarray, p: int = 1):
    """Legacy function for backward compatibility."""
    qaoa = TSPQAOA(n_cities)
    return qaoa.construct_circuit(Q, p)