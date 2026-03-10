# classical_solver.py (continued - add Ant Colony Optimization)
# Essential Qiskit imports for quantum optimization
import numpy as np
import random
import time
import math
from itertools import permutations
from typing import List, Tuple, Dict, Any
from qubo import TSPQUBO
   
   

class ClassicalTSP:
    """Classical TSP solvers with proper constraint handling matching QUBO."""
    
    def __init__(self, n_cities: int, distance_matrix: np.ndarray, **kwargs):
        """
        Initialize classical solver with QUBO formulation.
        
        Args:
            n_cities: Number of cities
            distance_matrix: Distance matrix between cities
            **kwargs: Additional constraints for QUBO builder
        """
        self.n_cities = n_cities
        self.distance_matrix = distance_matrix
        
        # Store original kwargs for constraint extraction
        self.constraint_kwargs = kwargs
        
        # Build QUBO
        self.builder = TSPQUBO(n_cities)
        self.qp, self.Q, self.q, self.const = self.builder.build(
            distance_matrix, **kwargs
        )
        
        # Extract constraint penalties from QUBO matrix
        self._extract_constraints_from_qubo()
        
        # Cache for constraint-aware cost calculations
        self.constraint_cache = {}
    
    def _extract_constraints_from_qubo(self):
        """Extract constraint penalties from QUBO matrix for use during search."""
        n = self.n_cities
        
        # 1. Extract city constraints (penalty for visiting same city at different times)
        self.city_constraint_penalty = 0.0
        for city in range(n):
            for t1 in range(n):
                for t2 in range(t1 + 1, n):
                    idx1 = city * n + t1
                    idx2 = city * n + t2
                    penalty = self.Q[idx1, idx2]
                    if penalty > 0:
                        self.city_constraint_penalty = max(self.city_constraint_penalty, penalty)
        
        # 2. Extract node compatibility constraints
        self.node_compatibility_penalties = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Check all time transitions
                    for t in range(n - 1):
                        idx1 = i * n + t
                        idx2 = j * n + (t + 1)
                        penalty = self.Q[idx1, idx2]
                        # Penalty beyond distance is constraint penalty
                        if penalty > self.distance_matrix[i, j]:
                            self.node_compatibility_penalties[i, j] = penalty - self.distance_matrix[i, j]
        
        # 3. Extract road constraints
        self.road_constraint_penalties = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Check all time transitions
                    for t in range(n - 1):
                        idx1 = i * n + t
                        idx2 = j * n + (t + 1)
                        penalty = self.Q[idx1, idx2]
                        # If penalty is huge compared to distance, it's a road constraint
                        if penalty > self.distance_matrix[i, j] * 10:  # Threshold
                            self.road_constraint_penalties[i, j] = penalty
        
        # 4. Extract time constraints from linear terms
        self.time_constraint_penalties = np.zeros((n, n))
        for city in range(n):
            for time in range(n):
                idx = city * n + time
                penalty = self.q[idx]
                if penalty > 0:  # Positive q means penalty for visiting at this time
                    self.time_constraint_penalties[city, time] = penalty
        
        print(f"[DEBUG] Extracted constraints: city_penalty={self.city_constraint_penalty:.2f}")
    
    def _calculate_transition_cost(self, city_i: int, city_j: int, time_i: int, time_j: int) -> float:
        """Calculate cost for transition from city_i at time_i to city_j at time_j."""
        # Base distance
        cost = self.distance_matrix[city_i, city_j]
        
        # Add constraint penalties
        if city_i == city_j and time_i != time_j:
            cost += self.city_constraint_penalty
        
        # Node compatibility penalty
        cost += self.node_compatibility_penalties[city_i, city_j]
        
        # Road constraint penalty
        cost += self.road_constraint_penalties[city_i, city_j]
        
        # Time constraint penalty for destination
        cost += self.time_constraint_penalties[city_j, time_j]
        
        return cost
    
    def compute_qubo_cost_for_tour(self, tour: List[int]) -> float:
        """
        Compute QUBO cost for a tour by converting to proper bitstring.
        
        Args:
            tour: Permutation of cities [c0, c1, ..., cn-1]
        
        Returns:
            Full QUBO cost (matching quantum measurement)
        """
        return self.builder.compute_cost(self.Q, self.q, self.const, 
                                         self.tour_to_bitstring(tour))
    
    def tour_to_bitstring(self, tour: List[int]) -> np.ndarray:
        """Convert tour to one-hot encoding bitstring."""
        x = np.zeros(self.n_cities * self.n_cities)
        for t, city in enumerate(tour):
            idx = city * self.n_cities + t
            x[idx] = 1
        return x
    
    def bitstring_to_tour(self, x: np.ndarray) -> List[int]:
        """Convert bitstring back to tour."""
        tour = []
        x_matrix = x.reshape(self.n_cities, self.n_cities)
        for t in range(self.n_cities):
            city = np.argmax(x_matrix[:, t])
            tour.append(city)
        return tour
    
    def is_valid_one_hot_encoding(self, x: np.ndarray) -> bool:
        """Check if bitstring satisfies TSP one-hot constraints."""
        x_matrix = x.reshape(self.n_cities, self.n_cities)
        column_check = np.all(np.sum(x_matrix, axis=0) == 1)
        row_check = np.all(np.sum(x_matrix, axis=1) == 1)
        return bool(column_check and row_check)
    
    def compute_actual_distance(self, tour: List[int]) -> float:
        """
        Compute actual distance cost (ignoring constraints).
        
        Args:
            tour: List of cities in order
            
        Returns:
            Total distance of the tour
        """
        total = 0.0
        n = len(tour)
        for i in range(n):
            j = (i + 1) % n
            total += self.distance_matrix[tour[i], tour[j]]
        return total
    
    def _find_constraint_violations(self, tour: List[int]) -> Dict[str, List]:
        """Find constraint violations in a tour."""
        violations = {
            'node_compatibility': [],
            'road_constraint': [],
            'time_constraint': []
        }
        
        n = len(tour)
        
        # Check all transitions
        for i in range(n):
            current_city = tour[i]
            next_city = tour[(i + 1) % n]
            current_time = i
            
            # Node compatibility violation
            if self.node_compatibility_penalties[current_city, next_city] > 0:
                violations['node_compatibility'].append((i, (i + 1) % n))
            
            # Road constraint violation
            if self.road_constraint_penalties[current_city, next_city] > 0:
                violations['road_constraint'].append((i,))
            
            # Time constraint violation
            if self.time_constraint_penalties[current_city, current_time] > 0:
                violations['time_constraint'].append((current_city, current_time))
        
        return violations
    
    def _count_constraint_violations(self, tour: List[int]) -> int:
        """Count total constraint violations in a tour."""
        violations = self._find_constraint_violations(tour)
        return sum(len(v) for v in violations.values())
    
    # ========== ANT COLONY OPTIMIZATION ==========
    
    def constraint_aware_ant_colony(
        self,
        n_ants: int = 5,           # CHANGED: Reduced for comparable total evaluations
        n_iterations: int = 100,    # CHANGED: 500 × 15 = 7,500 total tours (matches CL-QAOA)
        alpha: float = 1.0,
        beta: float = 2.0,
        rho: float = 0.5,
        q0: float = 0.9,
        tau0: float = 1.0,
        use_constraint_awareness: bool = True
    ) -> Tuple[List[int], float, Dict[str, Any]]:
        """
        Ant Colony Optimization with constraint awareness.
        
        Args:
            n_ants: Number of ants (15 × 500 = 7,500 total tours)
            n_iterations: Number of iterations
            alpha: Importance of pheromone
            beta: Importance of heuristic information
            rho: Evaporation rate
            q0: Exploitation probability
            tau0: Initial pheromone level
            use_constraint_awareness: Whether to incorporate constraint penalties
            
        Returns:
            (best_tour, best_cost, metadata)
        """
        start_time = time.time()
        
        n = self.n_cities
        best_tour = None
        best_cost = float('inf')
        
        # Initialize pheromone matrix
        tau = np.ones((n, n)) * tau0
        np.fill_diagonal(tau, 0)  # No pheromone on self-transitions
        
        # Statistics
        iteration_best_costs = []
        
        for iteration in range(n_iterations):
            ant_tours = []
            ant_costs = []
            
            # Each ant builds a solution
            for ant in range(n_ants):
                # Random starting city
                start_city = random.randint(0, n - 1)
                tour = [start_city]
                visited = set([start_city])
                
                # Build tour step by step
                for step in range(1, n):
                    current_city = tour[-1]
                    current_time = step - 1
                    
                    # Calculate transition probabilities
                    allowed = [j for j in range(n) if j not in visited]
                    
                    if not allowed:
                        # Fallback: visit remaining cities in any order
                        remaining = [j for j in range(n) if j not in visited]
                        tour.extend(remaining)
                        break
                    
                    # Calculate probabilities
                    probabilities = []
                    for next_city in allowed:
                        next_time = step
                        
                        # Heuristic information (eta)
                        # Base: inverse of distance
                        eta = 1.0 / (self.distance_matrix[current_city, next_city] + 1e-10)
                        
                        # Incorporate constraints if enabled
                        if use_constraint_awareness:
                            # Calculate constraint-aware cost
                            constraint_penalty = 0.0
                            
                            # Node compatibility penalty
                            constraint_penalty += self.node_compatibility_penalties[current_city, next_city]
                            
                            # Road constraint penalty
                            constraint_penalty += self.road_constraint_penalties[current_city, next_city]
                            
                            # Time constraint penalty
                            constraint_penalty += self.time_constraint_penalties[next_city, next_time]
                            
                            # Adjust heuristic based on constraints
                            if constraint_penalty > 0:
                                eta *= 1.0 / (1.0 + constraint_penalty)
                        
                        # Pheromone and heuristic combination
                        prob = (tau[current_city, next_city] ** alpha) * (eta ** beta)
                        probabilities.append(prob)
                    
                    # Normalize probabilities
                    total = sum(probabilities)
                    if total == 0:
                        probabilities = [1.0 / len(allowed)] * len(allowed)
                    else:
                        probabilities = [p / total for p in probabilities]
                    
                    # Choose next city
                    if random.random() < q0:
                        # Exploitation: choose best
                        next_idx = np.argmax(probabilities)
                    else:
                        # Exploration: choose based on probability
                        next_idx = random.choices(range(len(allowed)), weights=probabilities)[0]
                    
                    next_city = allowed[next_idx]
                    tour.append(next_city)
                    visited.add(next_city)
                
                # Compute QUBO cost
                cost = self.compute_qubo_cost_for_tour(tour)
                ant_tours.append(tour)
                ant_costs.append(cost)
                
                # Update best solution
                if cost < best_cost:
                    best_cost = cost
                    best_tour = tour.copy()
            
            # Update pheromones
            # Evaporation
            tau *= (1.0 - rho)
            
            # Add pheromone from ants
            for tour, cost in zip(ant_tours, ant_costs):
                # Amount of pheromone to deposit
                delta_tau = 1.0 / (cost + 1e-10)  # Better solutions deposit more pheromone
                
                # Add pheromone to edges in the tour
                for i in range(n):
                    j = (i + 1) % n
                    tau[tour[i], tour[j]] += delta_tau
                    tau[tour[j], tour[i]] += delta_tau  # Symmetric
            
            # Keep track of iteration best
            iteration_best = min(ant_costs)
            iteration_best_costs.append(iteration_best)
            
            # Optional: Apply local search to best ant
            if iteration % 10 == 0 and iteration > 0:
                best_ant_idx = np.argmin(ant_costs)
                improved_tour = self._apply_local_search(ant_tours[best_ant_idx])
                improved_cost = self.compute_qubo_cost_for_tour(improved_tour)
                
                if improved_cost < best_cost:
                    best_cost = improved_cost
                    best_tour = improved_tour
        
        runtime = time.time() - start_time
        
        metadata = {
            'runtime': runtime,
            'iterations': n_iterations,
            'n_ants': n_ants,
            'method': 'constraint_aware_ant_colony',
            'use_constraint_awareness': use_constraint_awareness,
            'qubo_cost': best_cost,
            'actual_distance': self.compute_actual_distance(best_tour),
            'constraint_violations': self._count_constraint_violations(best_tour),
            'iteration_best_costs': iteration_best_costs
        }
        
        return best_tour, best_cost, metadata
    def _apply_local_search(self, tour: List[int]) -> List[int]:
        """Apply 2-opt local search to improve a tour."""
        n = len(tour)
        best_tour = tour.copy()
        best_cost = self.compute_qubo_cost_for_tour(tour)
        improved = True
        
        while improved:
            improved = False
            for i in range(n - 1):
                for j in range(i + 1, n):
                    # Try 2-opt swap
                    new_tour = best_tour.copy()
                    new_tour[i:j+1] = reversed(new_tour[i:j+1])
                    new_cost = self.compute_qubo_cost_for_tour(new_tour)
                    
                    if new_cost < best_cost:
                        best_tour = new_tour
                        best_cost = new_cost
                        improved = True
                        break
                if improved:
                    break
        
        return best_tour
    
    # ========== EXISTING METHODS (kept for compatibility) ==========
    
    def brute_force(self, max_n: int = 8) -> Tuple[List[int], float, Dict[str, Any]]:
        """True optimal search using QUBO cost function."""
        if self.n_cities > max_n:
            raise ValueError(f"Brute force only for n <= {max_n}. Got n={self.n_cities}")
        
        start_time = time.time()
        best_tour = None
        best_cost = float('inf')
        
        total_tours = math.factorial(self.n_cities)
        print(f"Searching {total_tours} tours with QUBO evaluation...")
        
        for perm in permutations(range(1, self.n_cities)):
            tour = [0] + list(perm)
            cost = self.compute_qubo_cost_for_tour(tour)
            
            if cost < best_cost:
                best_cost = cost
                best_tour = tour
        
        runtime = time.time() - start_time
        
        metadata = {
            'runtime': runtime,
            'total_tours': total_tours,
            'method': 'brute_force',
            'bitstring': self.tour_to_bitstring(best_tour) if best_tour else None,
            'is_valid_encoding': self.is_valid_one_hot_encoding(
                self.tour_to_bitstring(best_tour)
            ) if best_tour else False,
            'qubo_cost': best_cost
        }
        
        return best_tour, best_cost, metadata
    
    def simulated_annealing(
        self,
        max_iterations: int = 100,
        initial_temperature: float = None,
        cooling_rate: float = 0.99
    ) -> Tuple[List[int], float, Dict[str, Any]]:
        """Simulated Annealing using direct QUBO cost evaluation."""
        start_time = time.time()
        
        current_tour = list(range(self.n_cities))
        random.shuffle(current_tour)
        current_cost = self.compute_qubo_cost_for_tour(current_tour)
        
        best_tour = current_tour.copy()
        best_cost = current_cost
        
        if initial_temperature is None:
            sample_costs = []
            for _ in range(min(100, self.n_cities * 10)):
                random_tour = list(range(self.n_cities))
                random.shuffle(random_tour)
                cost = self.compute_qubo_cost_for_tour(random_tour)
                sample_costs.append(cost)
            initial_temperature = np.std(sample_costs) * 2
        
        temperature = initial_temperature
        iterations_without_improvement = 0
        
        for iteration in range(max_iterations):
            i, j = random.sample(range(self.n_cities), 2)
            i, j = min(i, j), max(i, j)
            
            new_tour = current_tour.copy()
            new_tour[i:j+1] = reversed(new_tour[i:j+1])
            new_cost = self.compute_qubo_cost_for_tour(new_tour)
            
            delta = new_cost - current_cost
            if delta < 0 or random.random() < math.exp(-delta / temperature):
                current_tour = new_tour
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_tour = current_tour.copy()
                    best_cost = current_cost
                    iterations_without_improvement = 0
                else:
                    iterations_without_improvement += 1
            else:
                iterations_without_improvement += 1
            
            temperature *= cooling_rate
            
            if iterations_without_improvement > max_iterations // 10:
                break
        
        runtime = time.time() - start_time
        
        metadata = {
            'runtime': runtime,
            'iterations': iteration + 1,
            'final_temperature': temperature,
            'method': 'simulated_annealing',
            'bitstring': self.tour_to_bitstring(best_tour),
            'is_valid_encoding': True,
            'qubo_cost': best_cost
        }
        
        return best_tour, best_cost, metadata
    
    def constraint_aware_simulated_annealing(
        self,
        max_iterations: int = 100,  # CHANGED: 7,500 iterations to match CL-QAOA
        initial_temperature: float = None,
        cooling_rate: float = 0.995,
        use_constraint_aware_moves: bool = True
    ) -> Tuple[List[int], float, Dict[str, Any]]:
        """Simulated Annealing that uses constraint information during search."""
        start_time = time.time()
        
        # Generate initial tour using random permutation
        current_tour = list(range(self.n_cities))
        random.shuffle(current_tour)
        current_cost = self.compute_qubo_cost_for_tour(current_tour)
        
        best_tour = current_tour.copy()
        best_cost = current_cost
        
        if initial_temperature is None:
            sample_costs = []
            for _ in range(20):
                if use_constraint_aware_moves:
                    neighbor = self._generate_constraint_aware_neighbor(current_tour)
                else:
                    neighbor = self._generate_random_neighbor(current_tour)
                cost = self.compute_qubo_cost_for_tour(neighbor)
                sample_costs.append(abs(cost - current_cost))
            
            initial_temperature = np.mean(sample_costs) * 5
            if initial_temperature < 1.0:
                initial_temperature = 10.0
        
        temperature = initial_temperature
        no_improve_count = 0
        max_no_improve = max_iterations // 20
        
        for iteration in range(max_iterations):
            if use_constraint_aware_moves and iteration % 3 != 0:
                new_tour = self._generate_constraint_aware_neighbor(current_tour)
            else:
                new_tour = self._generate_random_neighbor(current_tour)
            
            new_cost = self.compute_qubo_cost_for_tour(new_tour)
            
            delta = new_cost - current_cost
            
            constraint_factor = 1.0
            if self._count_constraint_violations(new_tour) < self._count_constraint_violations(current_tour):
                constraint_factor = 2.0
            
            acceptance_prob = math.exp(-delta / (temperature * constraint_factor))
            
            if delta < 0 or random.random() < acceptance_prob:
                current_tour = new_tour
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_tour = current_tour.copy()
                    best_cost = current_cost
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
            
            temperature *= cooling_rate
            
            if no_improve_count > max_no_improve:
                temperature *= 1.5
                no_improve_count = 0
            
            if no_improve_count > max_no_improve * 3:
                break
        
        runtime = time.time() - start_time
        
        metadata = {
            'runtime': runtime,
            'iterations': iteration + 1,
            'final_temperature': temperature,
            'method': 'constraint_aware_simulated_annealing',
            'use_constraint_aware_moves': use_constraint_aware_moves,
            'qubo_cost': best_cost,
            'actual_distance': self.compute_actual_distance(best_tour),
            'constraint_violations': self._count_constraint_violations(best_tour)
        }
        
        return best_tour, best_cost, metadata
    def _generate_constraint_aware_neighbor(self, tour: List[int]) -> List[int]:
        """Generate neighbor that tries to reduce constraint violations."""
        n = len(tour)
        violations = self._find_constraint_violations(tour)
        
        if violations:
            violation_type = random.choice(list(violations.keys()))
            if violations[violation_type]:
                if violation_type == 'node_compatibility':
                    idx1, idx2 = random.choice(violations['node_compatibility'])
                    if idx2 == n - 1:
                        idx2 = 0
                    new_tour = tour.copy()
                    new_tour[idx1], new_tour[idx2] = new_tour[idx2], new_tour[idx1]
                    return new_tour
                
                elif violation_type == 'road_constraint':
                    idx, = random.choice(violations['road_constraint'])
                    new_tour = tour.copy()
                    j = random.randint(0, n-1)
                    while j == idx:
                        j = random.randint(0, n-1)
                    new_tour[idx], new_tour[j] = new_tour[j], new_tour[idx]
                    return new_tour
                
                elif violation_type == 'time_constraint':
                    city, time_idx = random.choice(violations['time_constraint'])
                    new_tour = tour.copy()
                    if city in new_tour:
                        current_idx = new_tour.index(city)
                        new_tour[current_idx], new_tour[time_idx] = new_tour[time_idx], new_tour[current_idx]
                    return new_tour
        
        return self._generate_random_neighbor(tour)
    
    def _generate_random_neighbor(self, tour: List[int]) -> List[int]:
        """Generate random neighbor using various moves."""
        n = len(tour)
        move_type = random.random()
        
        if move_type < 0.6:
            i, j = random.sample(range(n), 2)
            i, j = min(i, j), max(i, j)
            new_tour = tour.copy()
            new_tour[i:j+1] = reversed(new_tour[i:j+1])
        
        elif move_type < 0.8:
            i, j = random.sample(range(n), 2)
            new_tour = tour.copy()
            new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
        
        else:
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            while j == i:
                j = random.randint(0, n-1)
            new_tour = tour.copy()
            city = new_tour.pop(i)
            if j < i:
                new_tour.insert(j, city)
            else:
                new_tour.insert(j-1, city)
        
        return new_tour
    
    def compare_methods(self, n_runs: int = 5) -> Dict[str, Any]:
        """Compare all methods including ACO."""
        results = {}
        
        print("\n" + "="*60)
        print("COMPREHENSIVE METHOD COMPARISON")
        print("="*60)
        
        # Naive simulated annealing
        naive_sa_costs = []
        for _ in range(min(3, n_runs)):
            tour, cost, _ = self.simulated_annealing(
                max_iterations=100,
                cooling_rate=0.995
            )
            naive_sa_costs.append(cost)
        
        results['naive_simulated_annealing'] = {
            'avg_cost': np.mean(naive_sa_costs) if naive_sa_costs else float('inf'),
            'best_cost': min(naive_sa_costs) if naive_sa_costs else float('inf'),
            'worst_cost': max(naive_sa_costs) if naive_sa_costs else float('inf')
        }
        
        # Constraint-aware simulated annealing
        constraint_sa_costs = []
        for _ in range(min(3, n_runs)):
            tour, cost, _ = self.constraint_aware_simulated_annealing(
                max_iterations=100,
                cooling_rate=0.995,
                use_constraint_aware_moves=True
            )
            constraint_sa_costs.append(cost)
        
        results['constraint_aware_simulated_annealing'] = {
            'avg_cost': np.mean(constraint_sa_costs) if constraint_sa_costs else float('inf'),
            'best_cost': min(constraint_sa_costs) if constraint_sa_costs else float('inf'),
            'worst_cost': max(constraint_sa_costs) if constraint_sa_costs else float('inf')
        }
        
        # Ant Colony Optimization
        aco_costs = []
        for _ in range(min(2, n_runs)):  # ACO is slower
            tour, cost, _ = self.constraint_aware_ant_colony(
                n_ants=5,
                n_iterations=100,
                use_constraint_awareness=True
            )
            aco_costs.append(cost)
        
        results['constraint_aware_ant_colony'] = {
            'avg_cost': np.mean(aco_costs) if aco_costs else float('inf'),
            'best_cost': min(aco_costs) if aco_costs else float('inf'),
            'worst_cost': max(aco_costs) if aco_costs else float('inf')
        }
        
        # Print comprehensive comparison
        if naive_sa_costs:
            print(f"\nNaive Simulated Annealing:")
            print(f"  Best: {results['naive_simulated_annealing']['best_cost']:.2f}")
            print(f"  Avg:  {results['naive_simulated_annealing']['avg_cost']:.2f}")
        
        if constraint_sa_costs:
            print(f"\nConstraint-Aware Simulated Annealing:")
            print(f"  Best: {results['constraint_aware_simulated_annealing']['best_cost']:.2f}")
            print(f"  Avg:  {results['constraint_aware_simulated_annealing']['avg_cost']:.2f}")
        
        if naive_sa_costs and constraint_sa_costs:
            improvement_sa = ((results['naive_simulated_annealing']['avg_cost'] - 
                             results['constraint_aware_simulated_annealing']['avg_cost']) / 
                             results['naive_simulated_annealing']['avg_cost'] * 100)
            print(f"\nSA Improvement: {improvement_sa:.1f}%")
        
        if aco_costs:
            print(f"\nConstraint-Aware Ant Colony:")
            print(f"  Best: {results['constraint_aware_ant_colony']['best_cost']:.2f}")
            print(f"  Avg:  {results['constraint_aware_ant_colony']['avg_cost']:.2f}")
            
            # Compare ACO with best SA
            best_avg = min(
                results['constraint_aware_simulated_annealing']['avg_cost'] if constraint_sa_costs else float('inf'),
                results['naive_simulated_annealing']['avg_cost'] if naive_sa_costs else float('inf')
            )
            aco_vs_best = ((best_avg - results['constraint_aware_ant_colony']['avg_cost']) / 
                          best_avg * 100) if best_avg > 0 else 0
            print(f"\nACO vs Best Other Method: {aco_vs_best:.1f}% improvement")
        
        print("\n" + "="*60)
        
        return results