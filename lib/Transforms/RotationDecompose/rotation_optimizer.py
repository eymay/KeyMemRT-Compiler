#!/usr/bin/env python3
"""
Enhanced Addition Graph MIP Optimizer (Fixed)

This script provides optimization for addition graph problems using MIP.
It accepts input via stdin or file and outputs results via stdout or file.
The output includes decomposition of each target number using the base set.

Usage:
  cat input.json | python3 mip_optimizer.py > output.json
  python3 mip_optimizer.py --input input.json --output output.json
"""

import json
import sys
import argparse
import time
from collections import defaultdict, deque
import itertools
import pulp

class AdditionGraphMIPOptimizer:
    def __init__(self, target_numbers, max_chain_length=5):
        """
        Initialize the optimizer with target numbers and maximum chain length.
        
        Args:
            target_numbers: List of integers to be reached
            max_chain_length: Maximum number of additions allowed to reach a target
        """
        self.target_numbers = sorted(list(set(target_numbers)))  # Remove duplicates
        self.max_chain_length = max_chain_length
        self.max_number = max(target_numbers) if target_numbers else 0
        self.min_number = min(target_numbers) if target_numbers else 1
        
        # Pre-compute all possible addition chains up to max_chain_length
        self.precompute_addition_chains()
        
    def precompute_addition_chains(self):
        """
        Precompute all possible addition chains for all potential base numbers
        up to max_chain_length.
        """
        # Consider all numbers up to max_number as potential base numbers
        candidate_numbers = list(range(1, self.max_number + 1))
        
        # Store for each target, which base sets can reach it and in how many steps
        self.reachability = defaultdict(list)
        
        # For each potential base set of size 1 (individual numbers)
        for base_num in candidate_numbers:
            # Perform BFS to find all reachable numbers within max_chain_length
            steps = {base_num: 0}
            queue = deque([(base_num, 0)])  # (number, steps)
            
            while queue:
                current, current_steps = queue.popleft()
                
                # Stop if we've reached the maximum chain length
                if current_steps >= self.max_chain_length:
                    continue
                
                # Add current number to itself
                new_num = current + base_num
                if new_num <= 2 * self.max_number and new_num not in steps:
                    steps[new_num] = current_steps + 1
                    queue.append((new_num, current_steps + 1))
                    
                    # Record that this target can be reached with this base number
                    if new_num in self.target_numbers:
                        self.reachability[new_num].append((base_num, current_steps + 1))
                
                # Also check if the base number alone can reach any targets
                if base_num in self.target_numbers and base_num not in self.reachability[base_num]:
                    self.reachability[base_num].append((base_num, 0))
        
        # For each pair of potential base numbers
        for base_num1, base_num2 in itertools.combinations(candidate_numbers, 2):
            # Perform BFS to find all reachable numbers with this pair
            steps = {base_num1: 0, base_num2: 0}
            queue = deque([(base_num1, 0), (base_num2, 0)])  # (number, steps)
            
            while queue:
                current, current_steps = queue.popleft()
                
                # Stop if we've reached the maximum chain length
                if current_steps >= self.max_chain_length:
                    continue
                
                # Try adding each base number to the current number
                for base_num in [base_num1, base_num2]:
                    new_num = current + base_num
                    if new_num <= 2 * self.max_number and new_num not in steps:
                        steps[new_num] = current_steps + 1
                        queue.append((new_num, current_steps + 1))
                        
                        # Record that this target can be reached with this base set
                        if new_num in self.target_numbers:
                            self.reachability[new_num].append(((base_num1, base_num2), current_steps + 1))
                            
        # Print some statistics
        covered_targets = sum(1 for t in self.target_numbers if self.reachability[t])
        
    def solve_with_mip(self, base_size, include_non_targets=True, verbose=False):
        """
        Solve the addition graph problem using Mixed Integer Programming.
        
        Args:
            base_size: Size of the base set
            include_non_targets: Whether to include numbers not in target set
            time_limit: Time limit for solver in seconds
            verbose: Whether to print progress information
            
        Returns:
            best_base_set: The optimal base set
            total_steps: Total steps to reach all targets
            unreachable: Set of target numbers that couldn't be reached
        """
        start_time = time.time()
        
        # Determine candidate pool
        if include_non_targets:
            candidate_pool = list(range(1, self.max_number + 1))
        else:
            candidate_pool = self.target_numbers
            
        # Create a new LP problem
        prob = pulp.LpProblem("AdditionGraphOptimizer", pulp.LpMinimize)
        
        # Decision variables
        # x[i] = 1 if candidate i is in the base set, 0 otherwise
        x = pulp.LpVariable.dicts("x", candidate_pool, cat=pulp.LpBinary)
        
        # y[j] = 1 if target j is reached, 0 otherwise
        y = pulp.LpVariable.dicts("y", self.target_numbers, cat=pulp.LpBinary)
        
        # z[j][k] = 1 if target j is reached using addition chain k, 0 otherwise
        z = {}
        for target in self.target_numbers:
            z[target] = {}
            for idx, (base_nums, steps) in enumerate(self.reachability[target]):
                if isinstance(base_nums, tuple):  # Pair of base numbers
                    z[target][idx] = pulp.LpVariable(f"z_{target}_{idx}", cat=pulp.LpBinary)
                else:  # Single base number
                    z[target][idx] = pulp.LpVariable(f"z_{target}_{idx}", cat=pulp.LpBinary)
        
        # Objective: Minimize total steps + penalty for unreached targets
        unreachable_penalty = 1000  # High penalty for unreachable targets
        objective = pulp.lpSum(
            [unreachable_penalty * (1 - y[target]) for target in self.target_numbers] + 
            [z[target][idx] * steps for target in self.target_numbers 
             for idx, (base_nums, steps) in enumerate(self.reachability[target])]
        )
        prob += objective
        
        # Constraint: Base set size
        prob += pulp.lpSum([x[i] for i in candidate_pool]) == base_size
        
        # Constraint: A target is reached if at least one of its addition chains is used
        for target in self.target_numbers:
            if self.reachability[target]:  # If there are chains for this target
                prob += y[target] <= pulp.lpSum([z[target][idx] for idx in range(len(self.reachability[target]))])
            else:
                # If no chains can reach this target, it's always unreachable
                prob += y[target] == 0
        
        # Constraint: An addition chain can only be used if all its base numbers are in the base set
        for target in self.target_numbers:
            for idx, (base_nums, _) in enumerate(self.reachability[target]):
                if isinstance(base_nums, tuple):  # Pair of base numbers
                    prob += z[target][idx] <= x[base_nums[0]]
                    prob += z[target][idx] <= x[base_nums[1]]
                else:  # Single base number
                    prob += z[target][idx] <= x[base_nums]
        
        # Set time limit if supported by the solver
        if pulp.apis.coin_api.COIN_CMD().available():
            # solver = pulp.apis.coin_api.COIN_CMD(timeLimit=time_limit)
            prob.solve(solver)
        else:
            # Fallback to default solver
            prob.solve()
        
        # Extract the solution
        if prob.status == pulp.LpStatusOptimal:
            # Get the selected base set
            base_set = [i for i in candidate_pool if pulp.value(x[i]) > 0.5]
            
            # Count total steps and unreachable targets
            total_steps = 0
            unreachable = set()
            
            for target in self.target_numbers:
                if pulp.value(y[target]) < 0.5:
                    unreachable.add(target)
                else:
                    # Find which addition chain was used
                    for idx, (_, steps) in enumerate(self.reachability[target]):
                        if pulp.value(z[target][idx]) > 0.5:
                            total_steps += steps
                            break
            
            if verbose:
                print(f"Optimization completed in {time.time() - start_time:.2f} seconds.")
                print(f"Optimal base set: {base_set}")
                print(f"Total steps: {total_steps}")
                print(f"Coverage: {(len(self.target_numbers) - len(unreachable))/len(self.target_numbers)*100:.1f}% "
                      f"({len(self.target_numbers) - len(unreachable)}/{len(self.target_numbers)})")
                if unreachable:
                    print(f"Unreachable targets: {unreachable}")
            
            return base_set, total_steps, unreachable
        else:
            if verbose:
                print(f"No optimal solution found within the time limit. Status: {pulp.LpStatus[prob.status]}")
            return None, float('inf'), set(self.target_numbers)
    
    def find_addition_chain(self, base_set, target):
        """
        Find the shortest addition chain to reach a target using base set elements.
        
        Args:
            base_set: The base set elements
            target: The target to reach
            
        Returns:
            chain: List of tuples (a, b, a+b) representing addition steps
        """
        if target in base_set:
            return []  # No additions needed
        
        # BFS to find the shortest path
        visited = set(base_set)
        queue = deque([(num, []) for num in base_set])
        
        while queue:
            current, chain = queue.popleft()
            
            if current == target:
                return chain
            
            # Try adding each base number
            for base_num in base_set:
                new_num = current + base_num
                
                if new_num > 2 * self.max_number:
                    continue
                    
                if new_num not in visited:
                    visited.add(new_num)
                    new_chain = chain + [(current, base_num, new_num)]
                    queue.append((new_num, new_chain))
                    
                    if new_num == target:
                        return new_chain
        
        return None  # Target not reachable
                
    def get_decomposition(self, base_set):
        """
        Get decomposition of each target number using the base set.
        
        Args:
            base_set: The base set to use
            
        Returns:
            decompositions: Dictionary mapping each target to its decomposition
        """
        decompositions = {}
        
        for target in self.target_numbers:
            # If target is in base set, it's already decomposed
            if target in base_set:
                decompositions[target] = [target]
                continue
                
            # Find the addition chain
            chain = self.find_addition_chain(base_set, target)
            
            if chain is None:
                decompositions[target] = None  # Unreachable
                continue
                
            # Convert to flattened base numbers list
            decomp = self.extract_base_numbers(chain, base_set)
            decompositions[target] = decomp
                
        return decompositions
    
    def extract_base_numbers(self, chain, base_set):
        """
        Extract the list of base numbers used in an addition chain.
        
        Args:
            chain: The addition chain
            base_set: The base set
            
        Returns:
            base_numbers: List of base numbers used
        """
        if not chain:
            return []
            
        # Create a dependency graph: which numbers are needed to create each number
        dependencies = {}
        
        # Initialize with base numbers
        for base_num in base_set:
            dependencies[base_num] = []
        
        # Build dependencies from the chain
        for a, b, result in chain:
            if a in base_set:
                dependencies[result] = dependencies.get(result, []) + [a]
            else:
                dependencies[result] = dependencies.get(result, []) + dependencies.get(a, [])
                
            if b in base_set:
                dependencies[result] = dependencies.get(result, []) + [b]
            else:
                dependencies[result] = dependencies.get(result, []) + dependencies.get(b, [])
        
        # Get the final target number (result of the last addition)
        target = chain[-1][2]
        
        # Return the base numbers used
        return sorted(dependencies[target])
    
    def analyze_path_to_target(self, base_set, target):
        """
        Analyze and return the path to reach a specific target from the base set.
        
        Args:
            base_set: The base set to use
            target: The target number to reach
            
        Returns:
            path: List of steps to reach the target, or None if unreachable
        """
        if target in base_set:
            return [target]
            
        # Build the graph using BFS
        steps = {}
        parents = {}
        queue = deque()
        
        # Initialize with base set
        for num in base_set:
            steps[num] = 0
            queue.append(num)
        
        # BFS to find path
        found = False
        while queue and not found:
            current = queue.popleft()
            current_steps = steps[current]
            
            # Stop if we've reached the maximum chain length
            if current_steps >= self.max_chain_length:
                continue
                
            # Try adding each base number to the current number
            for addend in base_set:
                new_num = current + addend
                
                # Skip if the new number is already processed
                if new_num in steps:
                    continue
                    
                steps[new_num] = current_steps + 1
                parents[new_num] = (current, addend)
                queue.append(new_num)
                
                # Check if we've found our target
                if new_num == target:
                    found = True
                    break
        
        # If the target is unreachable
        if target not in parents:
            return None
            
        # Reconstruct the path
        path = []
        current = target
        while current not in base_set:
            parent, addend = parents[current]
            path.append((parent, addend, current))
            current = parent
            
        path.reverse()
        return path

def process_input(input_data, verbose=False):
    """
    Process input data and run the optimization.
    
    Args:
        input_data: Dictionary with input parameters
        verbose: Whether to print verbose output
        
    Returns:
        Dictionary with optimization results
    """
    # Extract parameters from input data
    target_numbers = input_data.get('target_indices', [])
    base_size = input_data.get('base_size', 3)
    max_chain_length = input_data.get('max_chain_length', 5)
    time_limit = input_data.get('time_limit', 300)
    include_non_targets = input_data.get('include_non_targets', True)

    # Create optimizer
    optimizer = AdditionGraphMIPOptimizer(target_numbers, max_chain_length)
    
    # Run optimization
    base_set, total_steps, unreachable = optimizer.solve_with_mip(
        base_size, include_non_targets, verbose
    )
    
    # Get decompositions for each target number
    decompositions = optimizer.get_decomposition(base_set) if base_set else {}
    
    # Convert dict to a list format for JSON
    decomposition_list = []
    for target, decomp in decompositions.items():
        decomposition_list.append({
            'target': target,
            'decomposition': decomp
        })
    
    # Convert unreachable set to list for JSON serialization
    unreachable_list = list(unreachable) if unreachable is not None else []
    
    # Prepare result
    result = {
        'base_set': base_set if base_set is not None else [],
        'total_steps': total_steps if total_steps != float('inf') else -1,
        'unreachable': unreachable_list,
        'coverage_percent': (len(target_numbers) - len(unreachable_list)) / len(target_numbers) * 100 if target_numbers else 0,
        'decompositions': decomposition_list
    }
    
    return result

def main():
    """
    Main function to handle command line arguments and stdin/stdout.
    """
    parser = argparse.ArgumentParser(description='Addition Graph MIP Optimizer')
    parser.add_argument('--input', help='Input JSON file (default: stdin)')
    parser.add_argument('--output', help='Output JSON file (default: stdout)')
    parser.add_argument('--verbose', action='store_true', help='Print verbose output')
    args = parser.parse_args()
    
    # Read input
    if args.input:
        with open(args.input, 'r') as f:
            input_data = json.load(f)
    else:
        # Read from stdin
        input_data = json.load(sys.stdin)
    
    # Process input
    result = process_input(input_data, args.verbose)
    
    # Write output
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
    else:
        # Write to stdout
        json.dump(result, sys.stdout, indent=2)

if __name__ == '__main__':
    main()

