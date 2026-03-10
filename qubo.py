# qubo.py
"""
QUBO formulation for TSP with constraints.
Based on: "Quantum Approaches to Urban Logistics" paper.
"""

import numpy as np
from typing import Optional, Tuple, List
from qiskit_optimization import QuadraticProgram


class TSPQUBO:
    """QUBO formulation for Traveling Salesman Problem."""
    
    def __init__(self, n_cities: int):
        """
        Initialize TSP QUBO formulator.
        
        Args:
            n_cities: Number of cities in TSP instance
        """
        self.n_cities = n_cities
        self.n_vars = n_cities * n_cities
    
    def _var_index(self, city: int, time: int) -> int:
        """Get variable index for city at time step."""
        return city * self.n_cities + time
    
    def build(
        self,
        distance_matrix: np.ndarray,
        node_categories: Optional[np.ndarray] = None,
        road_constraints: Optional[np.ndarray] = None,
        time_constraints: Optional[np.ndarray] = None,
        penalty_scale: float = 0.1  # NEW: control constraint strength
    ) -> Tuple[QuadraticProgram, np.ndarray, np.ndarray, float]:
        """
        Build complete QUBO for TSP.
        
        Args:
            distance_matrix: n x n matrix of distances/costs
            node_categories: Optional array for BNC constraint (0/1 values)
            road_constraints: Optional binary matrix for blocked roads
            time_constraints: Optional binary matrix for time windows
            penalty_scale: Scale factor for constraint penalties (0.1 means 10% of base)
        
        Returns:
            (qp, Q, q, constant) where:
            - qp: QuadraticProgram object
            - Q: Quadratic coefficient matrix (symmetric)
            - q: Linear coefficient vector
            - constant: Constant term
        """
        # Initialize
        Q = np.zeros((self.n_vars, self.n_vars))
        q = np.zeros(self.n_vars)
        constant = 0.0
        
        # Base penalty weight - use average distance
        if np.any(distance_matrix > 0):
            avg_distance = np.mean(distance_matrix[distance_matrix > 0])
        else:
            avg_distance = 1.0  # Default if all distances are 0
        
        λ_base = avg_distance * 2
        
        # Scale constraints
        λ_p = λ_base * penalty_scale
        λ_k = λ_base * penalty_scale
        λ_R = λ_base * penalty_scale
        λ_T = λ_base * penalty_scale
        
        # === 1. DISTANCE TERM (Equation 8) ===
        for i in range(self.n_cities):
            for j in range(self.n_cities):
                if i != j:
                    ω_ij = distance_matrix[i, j]
                    
                    # Forward edges
                    for t in range(self.n_cities - 1):
                        idx1 = self._var_index(i, t)
                        idx2 = self._var_index(j, t + 1)
                        Q[idx1, idx2] += ω_ij
                    
                    # Return edge
                    idx1 = self._var_index(i, self.n_cities - 1)
                    idx2 = self._var_index(j, 0)
                    Q[idx1, idx2] += ω_ij
        
        # === 2. CITY CONSTRAINT (Equation 10) ===
        for i in range(self.n_cities):
            # Quadratic terms (t ≠ s)
            for t in range(self.n_cities):
                idx_t = self._var_index(i, t)
                for s in range(t + 1, self.n_cities):
                    idx_s = self._var_index(i, s)
                    Q[idx_t, idx_s] += λ_p
            
            # Linear terms
            for t in range(self.n_cities):
                idx = self._var_index(i, t)
                q[idx] -= λ_p
            
            # Constant
            constant += λ_p
        
        # === 3. BINARY NODE COMPATIBILITY (Equation 13-14) ===
        if node_categories is not None:
            for i in range(self.n_cities):
                for j in range(self.n_cities):
                    if i != j and node_categories[i] == node_categories[j]:
                        penalty = λ_k
                        
                        for t in range(self.n_cities - 1):
                            idx1 = self._var_index(i, t)
                            idx2 = self._var_index(j, t + 1)
                            Q[idx1, idx2] += penalty
                        
                        idx1 = self._var_index(i, self.n_cities - 1)
                        idx2 = self._var_index(j, 0)
                        Q[idx1, idx2] += penalty
        
        # === 4. ROAD CONSTRAINTS (Equation 16-17) ===
        if road_constraints is not None:
            for i in range(self.n_cities):
                for j in range(self.n_cities):
                    if i != j and road_constraints[i, j] == 1:
                        penalty = λ_R
                        
                        for t in range(self.n_cities - 1):
                            idx1 = self._var_index(i, t)
                            idx2 = self._var_index(j, t + 1)
                            Q[idx1, idx2] += penalty
                        
                        idx1 = self._var_index(i, self.n_cities - 1)
                        idx2 = self._var_index(j, 0)
                        Q[idx1, idx2] += penalty
        
        # === 5. TIME CONSTRAINTS (Equation 19) ===
        if time_constraints is not None:
            for i in range(self.n_cities):
                for t in range(self.n_cities):
                    if time_constraints[i, t] == 1:
                        idx = self._var_index(i, t)
                        q[idx] += λ_T
        
        # === 6. MAKE MATRIX SYMMETRIC ===
        Q_sym = np.zeros((self.n_vars, self.n_vars))
        for i in range(self.n_vars):
            for j in range(self.n_vars):
                if i == j:
                    Q_sym[i, j] = Q[i, j]
                else:
                    Q_sym[i, j] = (Q[i, j] + Q[j, i]) / 2
        
        # === 7. CREATE QUADRATIC PROGRAM ===
        qp = QuadraticProgram(name=f"TSP_{self.n_cities}")
        
        for idx in range(self.n_vars):
            city = idx // self.n_cities
            time = idx % self.n_cities
            qp.binary_var(name=f"x_{city}_{time}")
        
        qp.minimize(
            linear=q.tolist(),
            quadratic=Q_sym,
            constant=constant
        )
        
        return qp, Q_sym, q, constant
    
    @staticmethod
    def decode_solution(bitstring: str, n_cities: int) -> Tuple[List[int], np.ndarray]:
        """
        Decode bitstring to tour.
        
        Args:
            bitstring: Binary string representing solution
            n_cities: Number of cities
        
        Returns:
            (tour, x_vector) where tour is list of city indices in order
        """
        n_vars = n_cities * n_cities
        
        # Ensure correct length
        if len(bitstring) > n_vars:
            bitstring = bitstring[-n_vars:]
        else:
            bitstring = bitstring.zfill(n_vars)
        
        # Convert to array
        x = np.array([int(bit) for bit in bitstring])
        
        # Parse into tour
        tour = []
        for t in range(n_cities):
            for city in range(n_cities):
                idx = city * n_cities + t
                if x[idx] == 1:
                    tour.append(city)
                    break
        
        return tour, x
    
    @staticmethod
    def compute_cost(Q: np.ndarray, q: np.ndarray, const: float, x: np.ndarray) -> float:
        """Compute QUBO cost for a solution vector."""
        return x @ Q @ x + q @ x + const


# Helper function for backward compatibility
def build_complete_qubo(n_cities: int, distance_matrix: np.ndarray, **kwargs):
    """Legacy function for backward compatibility."""
    builder = TSPQUBO(n_cities)
    return builder.build(distance_matrix, **kwargs)