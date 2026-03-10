#classical_optimizer.py
print(">>> classical_optimizer loaded from:", __file__)
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
import time
import warnings

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit_algorithms.optimizers import COBYLA
from qiskit.quantum_info import Statevector

# ============================================================
# FIXED BITSTRING DECODING FUNCTIONS
# ============================================================

def fixed_decode_bitstring(bitstring: str, n_cities: int) -> List[int]:
    """
    FIXED: Decode bitstring to city tour with robust handling.
    Handles both integer and string bitstrings from Qiskit.
    """
    n_vars = n_cities * n_cities
    
    # Ensure correct length
    if len(bitstring) != n_vars:
        if len(bitstring) > n_vars:
            # Take last n_vars bits (Qiskit sometimes returns longer strings)
            bitstring = bitstring[-n_vars:]
        else:
            # Pad with zeros
            bitstring = bitstring.zfill(n_vars)
    
    # Convert to array of integers
    try:
        bits = [int(b) for b in bitstring]
    except:
        # If bitstring contains non-binary characters
        return []
    
    # Group by time step (stride by n_cities)
    tour = [-1] * n_cities
    
    for time in range(n_cities):
        # Get bits for this time step: bits[time], bits[time + n_cities], ...
        time_bits = [bits[time + i*n_cities] for i in range(n_cities)]
        
        # Count ones in this time slot
        ones_count = sum(time_bits)
        if ones_count != 1:
            return []  # Invalid: not exactly one city per time
        
        # Find which city is at this time
        city_idx = time_bits.index(1)
        tour[time] = city_idx
    
    # Check all cities visited (should be permutation of 0..n_cities-1)
    if sorted(tour) != list(range(n_cities)):
        return []
    
    return tour


def is_valid_tour(tour: List[int], n_cities: int) -> bool:
    """Check if tour is valid."""
    if len(tour) != n_cities:
        return False
    if -1 in tour:
        return False
    if len(set(tour)) != n_cities:
        return False
    if sorted(tour) != list(range(n_cities)):
        return False
    return True


def compute_tour_cost(tour: List[int], distance_matrix: np.ndarray) -> float:
    """Compute tour cost."""
    n = len(tour)
    if n != distance_matrix.shape[0]:
        return float('inf')
    
    if not is_valid_tour(tour, n):
        return float('inf')
    
    try:
        cost = 0.0
        # Path cost
        for i in range(n - 1):
            cost += distance_matrix[tour[i], tour[i + 1]]
        # Return to start
        cost += distance_matrix[tour[-1], tour[0]]
        return cost
    except (IndexError, ValueError):
        return float('inf')


# ============================================================
# FIXED STATEVECTOR COMPUTATION
# ============================================================

def fixed_compute_statevector_expectation(
    circuit: QuantumCircuit,
    params: Dict[Parameter, float],
    distance_matrix: np.ndarray
) -> float:
    """
    FIXED: Expectation computation using statevector.
    Returns realistic costs, not 4000 penalty values.
    """
    n_cities = distance_matrix.shape[0]
    n_vars = n_cities * n_cities
    
    # Bind parameters (handle both Qiskit methods)
    try:
        bound_circuit = circuit.assign_parameters(params, inplace=False)
    except:
        try:
            bound_circuit = circuit.bind_parameters(params)
        except Exception as e:
            print(f"  Parameter binding error: {e}")
            return 100.0 * n_cities  # Reduced penalty
    
    # Remove measurements if present
    if bound_circuit.num_clbits > 0:
        bound_circuit = bound_circuit.remove_final_measurements(inplace=False)
    
    # Get statevector
    try:
        state = Statevector(bound_circuit)
        probs = state.probabilities_dict()
    except Exception as e:
        print(f"  Statevector error: {e}")
        return 100.0 * n_cities  # Reduced penalty
    
    # Compute expectation
    total_cost = 0.0
    valid_count = 0
    total_prob = 0.0
    
    for bitstring, probability in probs.items():
        if probability < 1e-8:
            continue
            
        total_prob += probability
        
        # Convert to binary string (handle Qiskit's different formats)
        try:
            if isinstance(bitstring, int):
                bit_str = bin(bitstring)[2:].zfill(n_vars)
            elif isinstance(bitstring, str):
                bit_str = bitstring
            else:
                continue
        except:
            continue
        
        # Decode tour using fixed decoder
        tour = fixed_decode_bitstring(bit_str, n_cities)
        
        if tour:
            # Valid tour - compute actual cost
            cost = compute_tour_cost(tour, distance_matrix)
            if cost < float('inf'):
                total_cost += probability * cost
                valid_count += 1
        else:
            # Invalid tour - add moderate penalty
            total_cost += probability * 50.0 * n_cities  # Reduced penalty
    
    # Debug output
    if len(probs) > 0:
        print(f"  State analysis: {valid_count} valid / {len(probs)} total, prob_sum={total_prob:.4f}")
    
    if valid_count == 0:
        # No valid tours found - return reasonable penalty
        avg_distance = np.mean(distance_matrix[distance_matrix > 0])
        return avg_distance * n_cities * 2.0  # Reasonable penalty based on distances
    
    return total_cost


# ============================================================
# FIXED OPTIMIZER CLASS
# ============================================================

class FixedQAOAOptimizer:
    """
    FIXED: QAOA optimizer with proper cost function.
    """
    
    def __init__(
        self,
        p: int = 1,
        max_iterations: int = 50,
        random_seed: int = 42
    ):
        self.p = p
        self.max_iterations = max_iterations
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Use COBYLA with reasonable settings
        self.optimizer = COBYLA(
        maxiter=max_iterations,
        tol=1e-4,
        rhobeg=0.5
        )

        
        # Track optimization history
        self.history = []
    
    def create_cost_function(
        self,
        circuit: QuantumCircuit,
        gammas: ParameterVector,
        betas: ParameterVector,
        distance_matrix: np.ndarray
    ):
        """Create cost function using fixed expectation computation."""
        
        def param_dict(params_array):
            """Create parameter dictionary from array."""
            param_dict = {}
            for i in range(self.p):
                param_dict[gammas[i]] = params_array[i]
                param_dict[betas[i]] = params_array[self.p + i]
            return param_dict
        
        def cost_function(params_array):
            """Cost evaluation function."""
            try:
                # Keep parameters in reasonable range
                params_array = np.clip(params_array, -2*np.pi, 2*np.pi)
                
                # Create parameter dict
                params = param_dict(params_array)
                
                # Compute expectation using FIXED function
                cost = fixed_compute_statevector_expectation(
                    circuit=circuit,
                    params=params,
                    distance_matrix=distance_matrix
                )
                
                # Store history for debugging
                self.history.append({
                    'params': params_array.copy(),
                    'cost': cost,
                    'time': time.time()
                })
                
                # Print progress occasionally
                if len(self.history) % 5 == 0:
                    print(f"    Iter {len(self.history)}: cost = {cost:.4f}")
                
                return cost
                
            except Exception as e:
                print(f"  Cost function error: {e}")
                return 1000.0  # High but not extreme penalty
    
        return cost_function
    
    def optimize(
        self,
        circuit: QuantumCircuit,
        gammas: ParameterVector,
        betas: ParameterVector,
        distance_matrix: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Run optimization.
        """
        print(f"Starting QAOA optimization (p={self.p}, max_iter={self.max_iterations})...")
        
        # Initialize parameters - small random values
        initial_params = np.random.random(2 * self.p) * 0.5
        
        # Create cost function
        cost_func = self.create_cost_function(
            circuit, gammas, betas, distance_matrix
        )
        
        # Run optimization
        try:
            result = self.optimizer.minimize(
                fun=cost_func,
                x0=initial_params
            )
            
            print(f"Optimization finished. Final cost: {result.fun:.4f}")
            print(f"Best parameters: {np.round(result.x, 4)}")
            
            return result.x, result.fun
            
        except Exception as e:
            print(f"Optimization failed: {e}")
            print("Returning initial parameters as fallback...")
            return initial_params, cost_func(initial_params)
    
    def sample_best_tour(
        self,
        circuit: QuantumCircuit,
        optimized_params: np.ndarray,
        gammas: ParameterVector,
        betas: ParameterVector,
        distance_matrix: np.ndarray,
        shots: int = 1000
    ) -> Tuple[List[int], float, Dict[str, Any]]:
        """
        Sample the best tour from optimized parameters.
        """
        n_cities = distance_matrix.shape[0]
        
        # Create parameter dict
        param_dict = {}
        for i in range(self.p):
            param_dict[gammas[i]] = optimized_params[i]
            param_dict[betas[i]] = optimized_params[self.p + i]
        
        # Bind parameters
        try:
            bound_circuit = circuit.assign_parameters(param_dict, inplace=False)
        except:
            try:
                bound_circuit = circuit.bind_parameters(param_dict)
            except Exception as e:
                print(f"Sampling error (binding): {e}")
                return [], float('inf'), {'error': str(e)}
        
        # Remove measurements for statevector
        if bound_circuit.num_clbits > 0:
            bound_circuit = bound_circuit.remove_final_measurements(inplace=False)
        
        # Get statevector
        try:
            state = Statevector(bound_circuit)
            probs = state.probabilities_dict()
        except Exception as e:
            print(f"Sampling error (statevector): {e}")
            return [], float('inf'), {'error': str(e)}
        
        # Find best tour
        best_tour = []
        best_cost = float('inf')
        all_costs = []
        valid_tours = []
        
        for bitstring, probability in probs.items():
            if probability < 1e-6:
                continue
            
            # Convert to binary string
            try:
                if isinstance(bitstring, int):
                    bit_str = bin(bitstring)[2:].zfill(n_cities * n_cities)
                else:
                    bit_str = bitstring
            except:
                continue
            
            # Decode tour
            tour = fixed_decode_bitstring(bit_str, n_cities)
            
            if tour:
                # Compute cost
                cost = compute_tour_cost(tour, distance_matrix)
                if cost < float('inf'):
                    all_costs.append(cost)
                    valid_tours.append((tour, cost, probability))
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_tour = tour
        
        # Statistics
        stats = {
            'best_cost': best_cost if best_cost < float('inf') else None,
            'avg_cost': np.mean(all_costs) if all_costs else None,
            'num_valid': len(all_costs),
            'total_samples': len(probs),
            'valid_tours': len(valid_tours)
        }
        
        # Print top tours
        if valid_tours:
            print(f"Found {len(valid_tours)} valid tours:")
            valid_tours.sort(key=lambda x: x[1])  # Sort by cost
            for i, (tour, cost, prob) in enumerate(valid_tours[:3]):  # Top 3
                print(f"  Tour {i+1}: {tour}, cost={cost:.4f}, prob={prob:.4f}")
        
        return best_tour, best_cost, stats


# ============================================================
# BACKWARD COMPATIBILITY WRAPPER (FIXED)
# ============================================================

class QAOAClassicalOptimizer:
    """
    Wrapper for backward compatibility using the FIXED optimizer.
    """
    
    def __init__(
        self,
        p: int = 1,
        max_iterations: int = 100,
        tolerance: float = 1e-6
    ):
        # Use the FIXED optimizer
        self.optimizer = FixedQAOAOptimizer(
            p=p,
            max_iterations=min(max_iterations, 100),  # Cap at 100
            random_seed=42
        )
    
    def optimize(
        self,
        circuit: QuantumCircuit,
        gammas: ParameterVector,
        betas: ParameterVector,
        distance_matrix: np.ndarray,
        initial_params: np.ndarray = None,
        shots_per_iter: Optional[int] = None  # Ignored for statevector
    ) -> Tuple[np.ndarray, float]:
        return self.optimizer.optimize(
            circuit=circuit,
            gammas=gammas,
            betas=betas,
            distance_matrix=distance_matrix
        )
    
    def perform_final_sampling(
        self,
        circuit: QuantumCircuit,
        optimized_params: np.ndarray,
        gammas: ParameterVector,
        betas: ParameterVector,
        distance_matrix: np.ndarray,
        shots: Optional[int] = None  # Ignored for statevector
    ) -> Tuple[List[int], float, Dict[str, Any]]:
        return self.optimizer.sample_best_tour(
            circuit=circuit,
            optimized_params=optimized_params,
            gammas=gammas,
            betas=betas,
            distance_matrix=distance_matrix,
            shots=shots or 1000
        )


# ============================================================
# LEGACY FUNCTIONS (FOR BACKWARD COMPATIBILITY)
# ============================================================

# Keep these for compatibility but they're not used by the fixed optimizer
def decode_bitstring_safely(bitstring: str, n_cities: int) -> List[int]:
    """Legacy function - use fixed_decode_bitstring instead."""
    return fixed_decode_bitstring(bitstring, n_cities)

def compute_tour_cost_safe(tour: List[int], distance_matrix: np.ndarray) -> float:
    """Legacy function - use compute_tour_cost instead."""
    return compute_tour_cost(tour, distance_matrix)

def compute_statevector_expectation(
    circuit: QuantumCircuit,
    params: Dict[Parameter, float],
    distance_matrix: np.ndarray
) -> float:
    """Legacy function - use fixed_compute_statevector_expectation instead."""
    return fixed_compute_statevector_expectation(circuit, params, distance_matrix)

def initialize_parameters(p: int, random_seed: int = None) -> np.ndarray:
    """Legacy function."""
    if random_seed is not None:
        np.random.seed(random_seed)
    return np.random.random(2 * p) * 2 * np.pi

def create_parameter_dict(
    gammas: ParameterVector,
    betas: ParameterVector,
    params_array: np.ndarray,
    p: int
) -> Dict[Parameter, float]:
    """Legacy function."""
    param_dict = {}
    for i in range(p):
        param_dict[gammas[i]] = params_array[i]
        param_dict[betas[i]] = params_array[p + i]
    return param_dict


# ============================================================
# TEST FUNCTION
# ============================================================

def test_fixed_optimizer():
    """Test the fixed optimizer."""
    print("\n" + "="*60)
    print("TESTING FIXED OPTIMIZER")
    print("="*60)
    
    # Create a simple 2-city problem
    n_cities = 2
    distance_matrix = np.array([
        [0, 5],
        [3, 0]
    ])
    
    # Create a simple circuit
    from qiskit.circuit import Parameter
    gamma = Parameter('γ')
    beta = Parameter('β')
    
    circuit = QuantumCircuit(4)  # 2² = 4 qubits
    
    # Simple parameterized circuit
    circuit.h([0, 1, 2, 3])
    circuit.rz(gamma, 0)
    circuit.rz(beta, 1)
    
    # Create parameter vectors
    from qiskit.circuit import ParameterVector
    gammas = ParameterVector('γ', 1)
    betas = ParameterVector('β', 1)
    
    # Test optimizer
    optimizer = FixedQAOAOptimizer(p=1, max_iterations=10)
    
    print(f"\nTesting with {n_cities} cities, distance matrix:")
    print(distance_matrix)
    
    # Test optimization
    params, cost = optimizer.optimize(circuit, gammas, betas, distance_matrix)
    
    print(f"\nOptimization result:")
    print(f"  Final cost: {cost:.4f}")
    print(f"  Parameters: {params}")
    
    # Test sampling
    tour, tour_cost, stats = optimizer.sample_best_tour(
        circuit, params, gammas, betas, distance_matrix
    )
    
    print(f"\nSampling result:")
    print(f"  Best tour: {tour}")
    print(f"  Tour cost: {tour_cost:.4f}")
    print(f"  Valid tours found: {stats.get('num_valid', 0)}")
    
    # Test decode function
    print(f"\nTesting decode function:")
    test_cases = [
        ('1001', [0, 1]),  # Valid: city0@time0, city1@time1
        ('0110', [1, 0]),  # Valid: city1@time0, city0@time1
        ('1100', []),      # Invalid: two at time0
        ('0011', []),      # Invalid: two at time1
    ]
    
    for bitstr, expected in test_cases:
        result = fixed_decode_bitstring(bitstr, 2)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{bitstr}' -> {result} (expected {expected})")


if __name__ == "__main__":
    # Run test when file is executed directly
    test_fixed_optimizer()