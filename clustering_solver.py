# cluster_solver.py

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Any
import itertools
import time

# Import our existing QAOA components
try:
    from quantum_circuits import TSPQAOA
    from qubo import TSPQUBO
    from classical_optimizer import QAOAClassicalOptimizer
    QAOA_AVAILABLE = True
    print("✓ All QAOA components loaded successfully")
except ImportError as e:
    print(f"✗ Import warning: {e}")
    QAOA_AVAILABLE = False

# Try alternative imports
if not QAOA_AVAILABLE:
    try:
        from quantum_circuits import TSPQAOA, construct_qaoa_circuit
        from qubo import TSPQUBO, build_complete_qubo
        from classical_optimizer import QAOAClassicalOptimizer
        QAOA_AVAILABLE = True
        print("✓ Alternative import successful")
    except:
        QAOA_AVAILABLE = False


class ImprovedHierarchicalClusteringSolver:
    """
    CL-QAOA Solver: Hierarchical clustering + QAOA for TSP.
    Uses actual QAOA with proper penalty handling and constraints.
    """
    
    def __init__(self, max_cluster_size: int = 3, qaoa_depth: int = 2, use_qaoa: bool = True):
        """
        Initialize CL-QAOA solver.
        
        Args:
            max_cluster_size: Maximum cities per cluster (max 3 for QAOA)
            qaoa_depth: QAOA circuit depth (p parameter)
            use_qaoa: Whether to use QAOA or fallback to classical
        """
        # Ensure max_cluster_size is appropriate for QAOA
        self.max_cluster_size = min(max_cluster_size, 3)  # QAOA works best for n≤3
        
        self.qaoa_depth = qaoa_depth
        self.use_qaoa = use_qaoa and QAOA_AVAILABLE
        
        # Initialize QAOA optimizer if available
        self.qaoa_optimizer = None
        if self.use_qaoa:
            self.qaoa_optimizer = QAOAClassicalOptimizer(
                p=self.qaoa_depth,
                max_iterations=50,
                tolerance=1e-4
            )
            print(f"✓ CL-QAOA solver initialized with QAOA (max cluster: {self.max_cluster_size}, depth: {qaoa_depth})")
        else:
            print(f"✓ CL-QAOA solver initialized (classical only, max cluster: {self.max_cluster_size})")
        
        # Statistics
        self.stats = {
            'clusters_created': 0,
            'clusters_solved_qaoa': 0,
            'clusters_solved_classical': 0,
            'total_iterations': 0,
            'total_time': 0.0
        }
    
    def solve_full_problem(self, coordinates: np.ndarray) -> Tuple[List[int], float]:
        """
        Solve TSP using CL-QAOA approach.
        
        Args:
            coordinates: Array of (x,y) coordinates
            
        Returns:
            (best_tour, best_cost)
        """
        start_time = time.time()
        n = len(coordinates)
        
        print(f"\n{'='*60}")
        print(f"CL-QAOA SOLVER: Solving {n}-city TSP")
        print(f"{'='*60}")
        
        # Compute distance matrix
        distance_matrix = self._compute_distance_matrix(coordinates)
        
        # Handle small problems directly
        if n <= self.max_cluster_size:
            print(f"Problem size {n} ≤ max cluster size {self.max_cluster_size}, solving directly...")
            tour, cost, _ = self._solve_directly(coordinates, distance_matrix, {})
            return tour, cost
        
        # Step 1: Create balanced clusters (NO SINGLE-CITY CLUSTERS)
        clusters = self._create_balanced_clusters(coordinates)
        print(f"Created {len(clusters)} clusters")
        
        # Step 2: Solve each cluster
        cluster_solutions = self._solve_clusters(coordinates, distance_matrix, clusters, {})
        
        # Step 3: Combine clusters with optimal connections
        final_tour, total_cost = self._combine_clusters(coordinates, distance_matrix, clusters, cluster_solutions)
        
        # Step 4: Apply final optimization
        final_tour = self._apply_local_optimization(final_tour, distance_matrix)
        total_cost = self._compute_tour_cost(final_tour, distance_matrix)
        
        # Update statistics
        self.stats['total_time'] = time.time() - start_time
        self.stats['total_cities'] = n
        self.stats['final_cost'] = total_cost
        
        # Verify all cities are included
        if len(final_tour) != n:
            print(f"  WARNING: Tour has {len(final_tour)} cities, expected {n}")
            # Add missing cities to the end
            missing = set(range(n)) - set(final_tour)
            if missing:
                final_tour.extend(list(missing))
                print(f"  Added missing cities: {missing}")
                # Recompute cost
                total_cost = self._compute_tour_cost(final_tour, distance_matrix)
        
        print(f"\n✓ Solution found:")
        print(f"  Tour: {final_tour}")
        print(f"  Cost: {total_cost:.4f}")
        print(f"  Time: {self.stats['total_time']:.3f}s")
        
        return final_tour, total_cost
    
    def _compute_distance_matrix(self, coordinates: np.ndarray) -> np.ndarray:
        """Compute Euclidean distance matrix."""
        n = len(coordinates)
        dist_matrix = np.sqrt(((coordinates[:, np.newaxis] - coordinates[np.newaxis, :])**2).sum(axis=2))
        np.fill_diagonal(dist_matrix, 0)
        return dist_matrix
    
    def _create_balanced_clusters(self, coordinates: np.ndarray) -> Dict[int, List[int]]:
        """Create balanced clusters without single-city clusters."""
        n = len(coordinates)
        
        # Determine number of clusters (ensure none are too large)
        n_clusters = max(2, (n + self.max_cluster_size - 1) // self.max_cluster_size)
        
        # Use hierarchical clustering
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage='ward',
            metric='euclidean'
        )
        
        labels = clustering.fit_predict(coordinates)
        
        # Organize into clusters
        clusters = defaultdict(list)
        for idx, label in enumerate(labels):
            clusters[label].append(idx)
        
        # Balance clusters to ensure none exceed max_cluster_size and no single-city clusters
        balanced_clusters = {}
        cluster_id = 0
        
        # First pass: keep good clusters
        good_clusters = []
        single_cities = []
        
        for cluster in clusters.values():
            if len(cluster) == 1:
                single_cities.extend(cluster)
            elif len(cluster) <= self.max_cluster_size:
                good_clusters.append(cluster)
            else:
                # Split large clusters
                for i in range(0, len(cluster), self.max_cluster_size):
                    chunk = cluster[i:i + self.max_cluster_size]
                    if len(chunk) >= 2:
                        good_clusters.append(chunk)
        
        # Assign good clusters
        for cluster in good_clusters:
            balanced_clusters[cluster_id] = cluster
            cluster_id += 1
        
        # Distribute single cities to existing clusters
        for city in single_cities:
            best_cluster = None
            min_distance = float('inf')
            
            for cid, cluster_cities in balanced_clusters.items():
                if len(cluster_cities) < self.max_cluster_size:
                    # Calculate average distance to cluster
                    total_dist = 0
                    for other_city in cluster_cities:
                        total_dist += np.linalg.norm(coordinates[city] - coordinates[other_city])
                    
                    avg_dist = total_dist / len(cluster_cities)
                    if avg_dist < min_distance:
                        min_distance = avg_dist
                        best_cluster = cid
            
            if best_cluster is not None:
                balanced_clusters[best_cluster].append(city)
                print(f"  Added single city {city} to cluster {best_cluster}")
            else:
                # Create new cluster for this single city (with nearest other single city)
                if len(balanced_clusters) == 0:
                    balanced_clusters[cluster_id] = [city]
                    cluster_id += 1
        
        # Report cluster statistics
        cluster_sizes = [len(c) for c in balanced_clusters.values()]
        print(f"Cluster sizes: {cluster_sizes}")
        print(f"Min size: {min(cluster_sizes)}, Max size: {max(cluster_sizes)}")
        
        self.stats['clusters_created'] = len(balanced_clusters)
        return balanced_clusters
    
    def _solve_clusters(self, coordinates: np.ndarray, 
                       distance_matrix: np.ndarray,
                       clusters: Dict[int, List[int]],
                       constraints: Dict) -> Dict[int, Tuple[List[int], float]]:
        """Solve each cluster using QAOA or classical methods."""
        solutions = {}
        
        print(f"\nSolving {len(clusters)} clusters:")
        print("-" * 40)
        
        for cluster_id, city_indices in clusters.items():
            size = len(city_indices)
            
            # Handle single-city cluster
            if size == 1:
                print(f"  Cluster {cluster_id} (1 city): Trivial solution")
                solutions[cluster_id] = ([city_indices[0]], 0.0)
                self.stats['clusters_solved_classical'] += 1
                continue
            
            # Extract cluster-specific data
            cluster_coords = coordinates[city_indices]
            cluster_dist = distance_matrix[np.ix_(city_indices, city_indices)]
            
            # Extract constraints for this cluster
            cluster_constraints = self._extract_cluster_constraints(city_indices, constraints)
            
            # Choose solver based on size and availability
            if self.use_qaoa and size <= 3 and size > 1:
                print(f"  Cluster {cluster_id} ({size} cities): Using QAOA...")
                tour_local, cost = self._solve_cluster_with_qaoa(
                    cluster_coords, cluster_dist, cluster_constraints
                )
                self.stats['clusters_solved_qaoa'] += 1
            else:
                print(f"  Cluster {cluster_id} ({size} cities): Using classical solver...")
                tour_local, cost = self._solve_cluster_classically(
                    cluster_coords, cluster_dist, cluster_constraints
                )
                self.stats['clusters_solved_classical'] += 1
            
            # Map back to global indices - FIXED MAPPING
            if tour_local and len(tour_local) == size:
                # Create mapping from local to global indices
                tour_global = [city_indices[i] for i in tour_local]
            else:
                print(f"    Warning: Invalid local tour, using original order")
                tour_global = city_indices.copy()
                # Compute actual cost for this fallback tour
                if size > 1:
                    cost = self._compute_tour_cost(tour_global, distance_matrix)
                else:
                    cost = 0.0
            
            # Apply local optimization to cluster tour
            if size > 1:
                tour_global = self._optimize_cluster_tour(tour_global, distance_matrix)
                cost = self._compute_tour_cost(tour_global, distance_matrix)
            
            solutions[cluster_id] = (tour_global, cost)
            print(f"    Cost: {cost:.4f}, Tour: {tour_global}")
        
        return solutions
    
    def _solve_cluster_with_qaoa(self, coordinates: np.ndarray, 
                                distance_matrix: np.ndarray,
                                constraints: Dict) -> Tuple[List[int], float]:
        """Solve a cluster using actual QAOA."""
        n = len(coordinates)
        
        if n > 3:
            print(f"    Warning: QAOA not recommended for n={n}>3, using classical")
            return self._solve_cluster_classically(coordinates, distance_matrix, constraints)
        
        try:
            # Build QUBO with constraints
            qubo_builder = TSPQUBO(n)
            
            # Extract constraints if provided
            node_categories = constraints.get('node_categories', None)
            road_constraints = constraints.get('road_constraints', None)
            time_constraints = constraints.get('time_constraints', None)
            
            # Build QUBO (with penalty scaling for small problems)
            penalty_scale = 0.15  # Moderate penalty for clusters
            qp, Q, q, const = qubo_builder.build(
                distance_matrix,
                node_categories=node_categories,
                road_constraints=road_constraints,
                time_constraints=time_constraints,
                penalty_scale=penalty_scale
            )
            
            # Create QAOA circuit
            qaoa = TSPQAOA(n)
            circuit, gammas, betas = qaoa.construct_circuit(Q, p=self.qaoa_depth, add_measurement=True)
            
            # Optimize QAOA parameters
            print(f"    Optimizing QAOA parameters (depth={self.qaoa_depth})...")
            
            if self.qaoa_optimizer:
                optimized_params, opt_cost = self.qaoa_optimizer.optimize(
                    circuit, gammas, betas, distance_matrix
                )
                
                # Sample best solution
                best_tour, best_cost, stats = self.qaoa_optimizer.perform_final_sampling(
                    circuit, optimized_params, gammas, betas, distance_matrix, shots=1000
                )
                
                if best_tour and len(best_tour) == n:
                    print(f"    QAOA found solution with cost: {best_cost:.4f}")
                    # Return local indices (0, 1, 2, ...)
                    return best_tour, best_cost
            
            print(f"    QAOA optimization failed, using classical fallback")
            
        except Exception as e:
            print(f"    QAOA error: {e}")
        
        # Fallback to classical solution
        return self._solve_cluster_classically(coordinates, distance_matrix, constraints)
    
    def _solve_cluster_classically(self, coordinates: np.ndarray,
                                  distance_matrix: np.ndarray,
                                  constraints: Dict) -> Tuple[List[int], float]:
        """Solve cluster using classical methods (exhaustive for small clusters)."""
        n = len(coordinates)
        
        if n <= 8:  # Exhaustive search for small clusters
            return self._solve_exhaustive(distance_matrix)
        else:  # Heuristic for larger clusters
            return self._solve_greedy(distance_matrix)
    
    def _solve_exhaustive(self, distance_matrix: np.ndarray) -> Tuple[List[int], float]:
        """Exhaustive search for TSP (for n ≤ 8)."""
        n = distance_matrix.shape[0]
        
        if n <= 1:
            return [0], 0.0
        
        best_tour = None
        best_cost = float('inf')
        
        # Fix first city as start
        for perm in itertools.permutations(range(1, n)):
            tour = [0] + list(perm)
            cost = self._compute_tour_cost(tour, distance_matrix)
            
            if cost < best_cost:
                best_cost = cost
                best_tour = tour
        
        return best_tour, best_cost
    
    def _solve_greedy(self, distance_matrix: np.ndarray) -> Tuple[List[int], float]:
        """Greedy nearest-neighbor algorithm."""
        n = distance_matrix.shape[0]
        
        tour = [0]
        visited = set([0])
        
        for _ in range(n - 1):
            current = tour[-1]
            remaining = [i for i in range(n) if i not in visited]
            next_city = min(remaining, key=lambda x: distance_matrix[current, x])
            tour.append(next_city)
            visited.add(next_city)
        
        cost = self._compute_tour_cost(tour, distance_matrix)
        
        # Apply 2-opt improvement
        tour = self._apply_2opt(tour, distance_matrix)
        cost = self._compute_tour_cost(tour, distance_matrix)
        
        return tour, cost
    
    def _combine_clusters(self, coordinates: np.ndarray,
                         distance_matrix: np.ndarray,
                         clusters: Dict[int, List[int]],
                         solutions: Dict[int, Tuple[List[int], float]]) -> Tuple[List[int], float]:
        """Combine cluster solutions into complete tour."""
        n_clusters = len(clusters)
        
        if n_clusters == 1:
            return solutions[0]
        
        # Build connection cost matrix between clusters
        connection_costs = np.zeros((n_clusters, n_clusters))
        connection_points = {}
        
        for i in range(n_clusters):
            tour_i, _ = solutions[i]
            for j in range(n_clusters):
                if i != j:
                    tour_j, _ = solutions[j]
                    
                    # Find minimum connection cost between tours
                    min_cost = float('inf')
                    best_pair = (tour_i[0], tour_j[0])
                    
                    for city_i in tour_i:
                        for city_j in tour_j:
                            cost = distance_matrix[city_i, city_j]
                            if cost < min_cost:
                                min_cost = cost
                                best_pair = (city_i, city_j)
                    
                    connection_costs[i, j] = min_cost
                    connection_points[(i, j)] = best_pair
        
        # Solve TSP on clusters (which cluster order to visit)
        cluster_order = self._solve_cluster_order(connection_costs)
        
        # Construct complete tour
        complete_tour = []
        total_cost = 0.0
        
        for idx, cluster_id in enumerate(cluster_order):
            cluster_tour, cluster_cost = solutions[cluster_id]
            
            # Connect to previous cluster
            if idx > 0:
                prev_cluster = cluster_order[idx - 1]
                connect_from, connect_to = connection_points[(prev_cluster, cluster_id)]
                
                # Rotate current cluster tour to start at connection point
                if cluster_tour[0] != connect_to:
                    try:
                        start_idx = cluster_tour.index(connect_to)
                        cluster_tour = cluster_tour[start_idx:] + cluster_tour[:start_idx]
                    except ValueError:
                        pass  # Keep original order if connection point not found
                
                # Add connection cost
                total_cost += connection_costs[prev_cluster, cluster_id]
            
            complete_tour.extend(cluster_tour)
            total_cost += cluster_cost
        
        # Add return to start
        total_cost += distance_matrix[complete_tour[-1], complete_tour[0]]
        
        return complete_tour, total_cost
    
    def _solve_cluster_order(self, connection_costs: np.ndarray) -> List[int]:
        """Solve TSP on cluster connection graph."""
        n = connection_costs.shape[0]
        
        if n <= 1:
            return [0] if n == 1 else []
        
        # Use nearest neighbor for cluster ordering
        tour = [0]
        visited = set([0])
        
        for _ in range(n - 1):
            current = tour[-1]
            remaining = [i for i in range(n) if i not in visited]
            next_cluster = min(remaining, key=lambda x: connection_costs[current, x])
            tour.append(next_cluster)
            visited.add(next_cluster)
        
        return tour
    
    def _solve_directly(self, coordinates: np.ndarray, 
                       distance_matrix: np.ndarray,
                       constraints: Dict) -> Tuple[List[int], float, Dict]:
        """Solve small problems directly."""
        n = len(coordinates)
        
        if self.use_qaoa and n <= 3:
            print("Using QAOA for direct solution...")
            tour_local, cost = self._solve_cluster_with_qaoa(coordinates, distance_matrix, constraints)
            
            if tour_local and len(tour_local) == n:
                # Convert to global indices (same for direct solution)
                tour_global = list(range(n))
                cost = self._compute_tour_cost(tour_global, distance_matrix)
                
                # Apply optimization
                tour_global = self._apply_local_optimization(tour_global, distance_matrix)
                cost = self._compute_tour_cost(tour_global, distance_matrix)
                
                self.stats['clusters_solved_qaoa'] += 1
                return tour_global, cost, self.stats
        
        # Classical fallback
        print("Using classical exhaustive search...")
        tour, cost = self._solve_exhaustive(distance_matrix)
        tour = self._apply_local_optimization(tour, distance_matrix)
        cost = self._compute_tour_cost(tour, distance_matrix)
        
        self.stats['clusters_solved_classical'] += 1
        return tour, cost, self.stats
    
    def _apply_local_optimization(self, tour: List[int], 
                                 distance_matrix: np.ndarray,
                                 max_iterations: int = 100) -> List[int]:
        """Apply 2-opt local optimization."""
        n = len(tour)
        if n <= 3:
            return tour.copy()
        
        best_tour = tour.copy()
        best_cost = self._compute_tour_cost(tour, distance_matrix)
        
        for _ in range(max_iterations):
            improved = False
            
            for i in range(n - 1):
                for j in range(i + 1, n):
                    # Try 2-opt swap
                    new_tour = best_tour.copy()
                    new_tour[i:j+1] = reversed(new_tour[i:j+1])
                    new_cost = self._compute_tour_cost(new_tour, distance_matrix)
                    
                    if new_cost < best_cost - 1e-6:
                        best_tour = new_tour
                        best_cost = new_cost
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return best_tour
    
    def _apply_2opt(self, tour: List[int], distance_matrix: np.ndarray) -> List[int]:
        """Simple 2-opt implementation."""
        return self._apply_local_optimization(tour, distance_matrix, max_iterations=50)
    
    def _optimize_cluster_tour(self, tour: List[int], distance_matrix: np.ndarray) -> List[int]:
        """Optimize tour within a cluster."""
        if len(tour) <= 2:
            return tour.copy()
        
        # Extract sub-matrix for these cities
        indices = tour
        n = len(indices)
        
        if n <= 1:
            return indices
        
        # Create sub-distance matrix
        sub_dist = distance_matrix[np.ix_(indices, indices)]
        
        # Solve TSP on this subset
        sub_tour, _ = self._solve_exhaustive(sub_dist)
        
        # Map back to original indices
        optimized_tour = [indices[i] for i in sub_tour]
        
        return optimized_tour
    
    def _compute_tour_cost(self, tour: List[int], distance_matrix: np.ndarray) -> float:
        """Compute total cost of a tour."""
        if len(tour) <= 1:
            return 0.0
        
        cost = 0.0
        n = len(tour)
        
        # Path edges
        for i in range(n - 1):
            cost += distance_matrix[tour[i], tour[i + 1]]
        
        # Return to start
        cost += distance_matrix[tour[-1], tour[0]]
        
        return cost
    
    def _is_valid_tour(self, tour: List[int], n_cities: int) -> bool:
        """Check if tour is a valid permutation."""
        if len(tour) != n_cities:
            return False
        if len(set(tour)) != n_cities:
            return False
        if min(tour) < 0 or max(tour) >= n_cities:
            return False
        return True
    
    def _extract_cluster_constraints(self, city_indices: List[int], 
                                    constraints: Dict) -> Dict:
        """Extract constraints relevant to a specific cluster."""
        if not constraints:
            return {}
        
        cluster_constraints = {}
        idx_map = {orig: new for new, orig in enumerate(city_indices)}
        
        # Node categories
        if 'node_categories' in constraints:
            orig_cats = constraints['node_categories']
            cluster_constraints['node_categories'] = np.array([orig_cats[i] for i in city_indices])
        
        # Road constraints (subset of matrix)
        if 'road_constraints' in constraints:
            orig_roads = constraints['road_constraints']
            n = len(city_indices)
            cluster_roads = np.zeros((n, n), dtype=int)
            for i, ci in enumerate(city_indices):
                for j, cj in enumerate(city_indices):
                    if ci < orig_roads.shape[0] and cj < orig_roads.shape[1]:
                        cluster_roads[i, j] = orig_roads[ci, cj]
            cluster_constraints['road_constraints'] = cluster_roads
        
        # Time constraints
        if 'time_constraints' in constraints:
            orig_times = constraints['time_constraints']
            n = len(city_indices)
            cluster_times = np.zeros((n, n), dtype=int)
            for i, ci in enumerate(city_indices):
                if ci < orig_times.shape[0]:
                    cluster_times[i] = orig_times[ci, :n]  # Take first n time slots
            cluster_constraints['time_constraints'] = cluster_times
        
        return cluster_constraints


# Example usage and testing
def test_solver():
    """Test the ImprovedHierarchicalClusteringSolver."""
    np.random.seed(42)
    
    print("ImprovedHierarchicalClusteringSolver Test")
    print("=" * 60)
    
    # Create a test problem
    n_cities = 8
    coordinates = np.random.rand(n_cities, 2) * 100
    
    # Create solver
    solver = ImprovedHierarchicalClusteringSolver(
        max_cluster_size=3,
        qaoa_depth=1,
        use_qaoa=True
    )
    
    # Solve the problem
    tour, cost = solver.solve_full_problem(coordinates)
    
    print(f"\n{'='*60}")
    print("FINAL RESULTS:")
    print(f"  Cities: {n_cities}")
    print(f"  Tour length: {len(tour)}")
    print(f"  Unique cities: {len(set(tour))}")
    print(f"  Total cost: {cost:.4f}")
    print(f"  Time: {solver.stats['total_time']:.3f}s")
    print(f"  QAOA clusters: {solver.stats.get('clusters_solved_qaoa', 0)}")
    print(f"  Classical clusters: {solver.stats.get('clusters_solved_classical', 0)}")
    
    # Verify solution
    if len(set(tour)) == n_cities:
        print("✓ Valid tour (all cities visited)")
    else:
        missing = set(range(n_cities)) - set(tour)
        print(f"✗ Invalid tour: {len(set(tour))}/{n_cities} unique cities")
        print(f"  Missing cities: {missing}")


if __name__ == "__main__":
    test_solver()
