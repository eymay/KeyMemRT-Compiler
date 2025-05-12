#!/usr/bin/env python3
"""
Enhanced Addition Graph MIP Optimizer with Saturated BSGS

This script provides optimization for addition graph problems using MIP and
saturated BSGS for continuous ranges.
It accepts input via stdin or file and outputs results via stdout or file.

Usage:
  cat input.json | python3 rotation_optimizer.py > output.json
  python3 rotation_optimizer.py --input input.json --output output.json
"""

import json
import sys
import argparse
import time
from collections import defaultdict, deque
import itertools
import math
import pulp

def identify_ranges(target_numbers, min_range_length=3, verbose=False):
    """
    Identify contiguous ranges in the target numbers.
    
    Args:
        target_numbers: List of integers to be reached
        min_range_length: Minimum length of range to identify
        verbose: Whether to print detailed information
        
    Returns:
        List of (start, end) tuples representing ranges
    """
    if not target_numbers:
        return []
    
    # Sort the targets
    sorted_targets = sorted(target_numbers)
        
    ranges = []
    start = sorted_targets[0]
    prev = start
    
    for i in range(1, len(sorted_targets)):
        curr = sorted_targets[i]
        if curr != prev + 1:
            # End of a range
            if prev - start + 1 >= min_range_length:
                ranges.append((start, prev))
            start = curr
        prev = curr
        
    # Check the last range
    if prev - start + 1 >= min_range_length:
        ranges.append((start, prev))
        
    if verbose:
        print(f"Found {len(ranges)} contiguous ranges with length >= {min_range_length}")
        for start, end in ranges:
            print(f"  Range: {start}-{end} ({end-start+1} elements)")
            
    return ranges

def apply_saturated_bsgs(ranges, base_size, verbose=False):
    """
    Apply BSGS with controlled saturation to stay within budget.
    
    Args:
        ranges: List of (start, end) tuples representing contiguous ranges
        base_size: Maximum size of the base set
        verbose: Whether to print detailed information
        
    Returns:
        base_elements: List of base elements
        decompositions: Dictionary mapping target -> decomposition
    """
    import math
    
    # Step 1: Estimate the number of elements needed for standard BSGS
    std_bsgs_elements = 0
    for start, end in ranges:
        range_size = end - start + 1
        m = int(math.ceil(math.sqrt(range_size)))
        elements_needed = len(range(1, m)) + len(range(1, (range_size // m) + 1))
        if start > 1:  # Add offset if needed
            elements_needed += 1
        std_bsgs_elements += elements_needed
    
    if verbose:
        print(f"Standard BSGS would use approximately {std_bsgs_elements} elements")
    
    # Step 2: Determine how to partition ranges based on budget
    partitioned_ranges = []
    
    # Calculate saturation level
    saturation = base_size / std_bsgs_elements if std_bsgs_elements > 0 else 1.0
    
    if verbose:
        print(f"Budget: {base_size} elements, Saturation level: {saturation:.2f}")
    
    # If we have plenty of budget, split ranges more aggressively
    if saturation >= 1.5:
        # We can afford to create more subranges for better performance
        target_split_ratio = saturation
        
        if verbose:
            print(f"High budget available, splitting ranges more aggressively")
        
        for start, end in ranges:
            range_size = end - start + 1
            
            # For very small ranges, don't partition
            if range_size < 15:
                partitioned_ranges.append((start, end))
                continue
            
            # Calculate how many subranges to create based on saturation
            # The higher the saturation, the more subranges we create
            num_splits = min(int(math.sqrt(range_size) * target_split_ratio / 2), range_size // 10)
            num_splits = max(1, num_splits)  # At least 1 split
            
            if verbose and num_splits > 1:
                print(f"Splitting range {start}-{end} into {num_splits} subranges")
            
            # Create the subranges
            subrange_size = range_size / num_splits
            for i in range(num_splits):
                subrange_start = start + int(i * subrange_size)
                subrange_end = start + int((i + 1) * subrange_size) - 1
                if i == num_splits - 1:  # Last subrange gets any remainder
                    subrange_end = end
                partitioned_ranges.append((subrange_start, subrange_end))
    elif saturation < 0.8:
        # We need to be more conservative with subranges
        if verbose:
            print(f"Limited budget, optimizing range splitting")
        
        # Sort ranges by size (largest first)
        sorted_ranges = sorted(ranges, key=lambda r: r[1] - r[0] + 1, reverse=True)
        
        # Allocate budget proportionally to range sizes
        total_elements = sum(end - start + 1 for start, end in sorted_ranges)
        
        for start, end in sorted_ranges:
            range_size = end - start + 1
            
            # For very small ranges, don't partition
            if range_size < 10:
                partitioned_ranges.append((start, end))
                continue
            
            # Allocate budget proportionally to range size
            range_budget = base_size * (range_size / total_elements)
            
            # Calculate how many splits we can afford
            m = int(math.ceil(math.sqrt(range_size)))
            elems_per_subrange = m + (range_size // m)
            affordable_splits = max(1, int(range_budget / elems_per_subrange))
            
            if verbose and affordable_splits > 1:
                print(f"Splitting range {start}-{end} into {affordable_splits} subranges")
            
            # Create the subranges
            subrange_size = range_size / affordable_splits
            for i in range(affordable_splits):
                subrange_start = start + int(i * subrange_size)
                subrange_end = start + int((i + 1) * subrange_size) - 1
                if i == affordable_splits - 1:  # Last subrange gets any remainder
                    subrange_end = end
                partitioned_ranges.append((subrange_start, subrange_end))
    else:
        # We're close to the budget, use standard BSGS
        partitioned_ranges = ranges
    
    # Step 3: Apply BSGS to each (sub)range
    base_elements = []
    decompositions = {}
    
    for start, end in partitioned_ranges:
        range_elements, range_decomps = apply_bsgs_to_range(start, end, verbose)
        base_elements.extend(range_elements)
        decompositions.update(range_decomps)
    
    # Remove duplicates while preserving order
    base_elements = list(dict.fromkeys(base_elements))
    
    # Step 4: If we exceed budget, prioritize elements
    if len(base_elements) > base_size:
        if verbose:
            print(f"Generated {len(base_elements)} elements, but budget is {base_size}")
            print(f"Prioritizing most important elements")
        
        # First, keep elements that are direct targets (most efficient)
        direct_targets = [e for e in base_elements if e in decompositions and decompositions[e] == [e]]
        
        # For the remaining budget, prioritize elements used most frequently
        remaining_budget = base_size - len(direct_targets)
        
        if remaining_budget > 0:
            # Count usage of each element
            element_usage = {}
            for target, decomp in decompositions.items():
                if decomp and target not in direct_targets:
                    for elem in decomp:
                        if elem not in direct_targets:
                            element_usage[elem] = element_usage.get(elem, 0) + 1
            
            # Sort remaining elements by usage
            other_elements = [e for e in base_elements if e not in direct_targets]
            sorted_elements = sorted(other_elements, key=lambda x: element_usage.get(x, 0), reverse=True)
            
            # Select top elements by usage
            selected_elements = sorted_elements[:remaining_budget]
            
            # Combine direct targets and selected elements
            base_elements = direct_targets + selected_elements
        else:
            # If we can't fit all direct targets, prioritize smallest ones
            # (which are more often used in combination with others)
            base_elements = sorted(direct_targets)[:base_size]
        
        # We need to recalculate decompositions for the reduced base set
        decompositions = {}
    
    if verbose:
        print(f"Final base set has {len(base_elements)} elements")
        
    return base_elements, decompositions

def apply_bsgs_to_range(start, end, verbose=False):
    """
    Apply standard Baby-Step Giant-Step to a range of consecutive integers.
    
    Args:
        start: Start of the range (inclusive)
        end: End of the range (inclusive)
        verbose: Whether to print detailed information
        
    Returns:
        base_elements: List of base elements to include
        decompositions: Dictionary mapping target -> base elements used
    """
    import math
    
    n = end - start + 1
    
    # Calculate optimal step size (approximately sqrt(n))
    m = int(math.ceil(math.sqrt(n)))
    
    # Baby steps: 1, 2, 3, ..., m-1
    baby_steps = list(range(1, m))
    
    # Giant steps: m, 2m, 3m, ...
    giant_steps = [m * i for i in range(1, (n // m) + 1)]
    
    # Combine baby and giant steps
    base_elements = []
    
    # For efficiency, handle ranges not starting at 1
    if start > 1:
        # If the range starts at a value > 1, we create shifted steps
        offset = start - 1
        
        # Create shifted steps directly to avoid using offset separately
        shifted_baby_steps = [offset + step for step in baby_steps]
        shifted_giant_steps = [offset + step for step in giant_steps]
        
        base_elements = shifted_baby_steps + shifted_giant_steps
    else:
        # Standard BSGS elements
        base_elements = baby_steps + giant_steps
    
    # Sort and remove duplicates
    base_elements = sorted(list(set(base_elements)))
    
    if verbose:
        if start > 1:
            print(f"BSGS for range {start}-{end}:")
            print(f"  Shifted baby steps: {shifted_baby_steps}")
            print(f"  Shifted giant steps: {shifted_giant_steps}")
        else:
            print(f"BSGS for range {start}-{end}:")
            print(f"  Baby steps: {baby_steps}")
            print(f"  Giant steps: {giant_steps}")
        print(f"  Total base elements: {len(base_elements)}")
    
    # Calculate decompositions for each target in the range
    decompositions = {}
    for target in range(start, end + 1):
        # If target is directly in the base set, use it
        if target in base_elements:
            decompositions[target] = [target]
            continue
        
        # Otherwise, find the optimal decomposition using at most 2 steps
        best_decomp = None
        
        # Try all possible 2-element combinations
        for elem1 in base_elements:
            if elem1 > target:
                continue
                
            if target - elem1 in base_elements:
                best_decomp = [elem1, target - elem1]
                best_decomp = sorted(best_decomp, reverse=True)
                break
        
        if best_decomp:
            decompositions[target] = sorted(best_decomp, reverse=True)
        else:
            # This should rarely happen with proper BSGS elements
            # Fall back to a greedy approach
            remaining = target
            decomp = []
            
            # Use largest elements that fit
            while remaining > 0:
                best_elem = 0
                for elem in base_elements:
                    if elem <= remaining and elem > best_elem:
                        best_elem = elem
                
                if best_elem == 0:
                    # Can't represent this number
                    break
                    
                decomp.append(best_elem)
                remaining -= best_elem
            
            if remaining == 0:
                decompositions[target] = sorted(decomp, reverse=True)
    
    return base_elements, decompositions

def apply_bsgs_to_range(start, end, verbose=False):
    """
    Apply standard Baby-Step Giant-Step to a range of consecutive integers.
    
    Args:
        start: Start of the range (inclusive)
        end: End of the range (inclusive)
        verbose: Whether to print detailed information
        
    Returns:
        base_elements: List of base elements to include
        decompositions: Dictionary mapping target -> base elements used
    """
    import math
    
    n = end - start + 1
    
    # Calculate optimal step size (approximately sqrt(n))
    m = int(math.ceil(math.sqrt(n)))
    
    # Baby steps: 1, 2, 3, ..., m-1
    baby_steps = list(range(1, m))
    
    # Giant steps: m, 2m, 3m, ...
    giant_steps = [m * i for i in range(1, (n // m) + 1)]
    
    # Combine baby and giant steps
    base_elements = []
    
    # For efficiency, handle ranges not starting at 1
    if start > 1:
        # If the range starts at a value > 1, we create shifted steps
        offset = start - 1
        
        # Create shifted steps directly to avoid using offset separately
        shifted_baby_steps = [offset + step for step in baby_steps]
        shifted_giant_steps = [offset + step for step in giant_steps]
        
        base_elements = shifted_baby_steps + shifted_giant_steps
    else:
        # Standard BSGS elements
        base_elements = baby_steps + giant_steps
    
    # Sort and remove duplicates
    base_elements = sorted(list(set(base_elements)))
    
    if verbose:
        if start > 1:
            print(f"BSGS for range {start}-{end}:")
            print(f"  Shifted baby steps: {shifted_baby_steps}")
            print(f"  Shifted giant steps: {shifted_giant_steps}")
        else:
            print(f"BSGS for range {start}-{end}:")
            print(f"  Baby steps: {baby_steps}")
            print(f"  Giant steps: {giant_steps}")
        print(f"  Total base elements: {len(base_elements)}")
    
    # Calculate decompositions for each target in the range
    decompositions = {}
    for target in range(start, end + 1):
        # If target is directly in the base set, use it
        if target in base_elements:
            decompositions[target] = [target]
            continue
        
        # Otherwise, find the optimal decomposition using at most 2 steps
        best_decomp = None
        
        # Try all possible 2-element combinations
        for elem1 in base_elements:
            if elem1 > target:
                continue
                
            if target - elem1 in base_elements:
                best_decomp = [elem1, target - elem1]
                break
        
        if best_decomp:
            decompositions[target] = sorted(best_decomp)
        else:
            # This should rarely happen with proper BSGS elements
            # Fall back to a greedy approach
            remaining = target
            decomp = []
            
            # Use largest elements that fit
            while remaining > 0:
                best_elem = 0
                for elem in base_elements:
                    if elem <= remaining and elem > best_elem:
                        best_elem = elem
                
                if best_elem == 0:
                    # Can't represent this number
                    break
                    
                decomp.append(best_elem)
                remaining -= best_elem
            
            if remaining == 0:
                decompositions[target] = sorted(decomp)
    
    return base_elements, decompositions

class AdditionGraphMIPOptimizer:
    def __init__(self, target_numbers, max_chain_length=5, verbose=False):
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
        self.verbose = verbose
        
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
        if self.verbose:
            print(f"Precomputation completed. {covered_targets}/{len(self.target_numbers)} targets can be reached.")

    def optimize_with_saturated_bsgs_and_mip(self, base_size, min_range_length=3):
        """
        Use saturated BSGS for continuous ranges and MIP for the rest.
        
        Args:
            base_size: Maximum size of the base set
            min_range_length: Minimum length of continuous range to use BSGS
            
        Returns:
            base_set: The optimized base set
            decompositions: Decompositions for all targets
            unreachable: Set of unreachable targets
        """
        start_time = time.time()
        
        # Step 1: Identify continuous ranges
        ranges = identify_ranges(self.target_numbers, min_range_length, self.verbose)
        
        # Step 2: Apply saturated BSGS to ranges
        base_elements, decompositions = apply_saturated_bsgs(ranges, base_size, self.verbose)
        
        # Step 3: Identify targets covered by BSGS
        range_covered = set()
        for start, end in ranges:
            range_covered.update(range(start, end + 1))
        
        # Step 4: Find targets not covered by BSGS
        remaining_targets = [t for t in self.target_numbers if t not in range_covered]
        
        if self.verbose:
            print(f"BSGS covers {len(range_covered)} targets")
            print(f"Remaining targets: {len(remaining_targets)}")
            print(f"BSGS uses {len(base_elements)} out of {base_size} elements")
        
        # Step 5: If we have remaining elements in budget, use MIP for remaining targets
        available_slots = base_size - len(base_elements)
        
        if remaining_targets and available_slots > 0:
            if self.verbose:
                print(f"Running MIP for remaining targets with {available_slots} available slots")
                
            # Create MIP optimizer for ALL targets (to allow BSGS elements to help reach other targets)
            mip_optimizer = AdditionGraphMIPOptimizer(
                self.target_numbers, self.max_chain_length, self.verbose)
                
            # Find additional base elements with pre-selected BSGS elements
            additional_base_set, _, _ = mip_optimizer.solve_with_mip(
                base_size=available_slots,
                include_non_targets=True,
                pre_selected_elements=base_elements)
                
            if additional_base_set:
                # Get decompositions for ALL targets using the combined set
                all_elements = base_elements + additional_base_set
                combined_decomps = mip_optimizer.get_decomposition(all_elements)
                    
                # Update base elements and decompositions
                base_elements = all_elements
                decompositions = combined_decomps
        
        # Calculate total steps
        total_steps = sum(len(decomp) - 1 if decomp else 0 for decomp in decompositions.values())
        
        # Count unreachable targets
        unreachable = set(t for t in self.target_numbers if t not in decompositions or decompositions[t] is None)
        
        if self.verbose:
            print(f"Optimization completed in {time.time() - start_time:.2f} seconds")
            print(f"Final base set has {len(base_elements)} elements")
            coverage = (len(self.target_numbers) - len(unreachable)) / len(self.target_numbers) * 100
            print(f"Coverage: {coverage:.1f}% "
                  f"({len(self.target_numbers) - len(unreachable)}/{len(self.target_numbers)})")
            
        return base_elements, decompositions, unreachable, total_steps
        
    def solve_with_mip(self, base_size, include_non_targets=True, pre_selected_elements=None):
        """
        Solve the addition graph problem using Mixed Integer Programming.
        
        Args:
            base_size: Size of the base set
            include_non_targets: Whether to include numbers not in target set
            pre_selected_elements: List of elements that must be in the base set
            
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
            candidate_pool = self.target_numbers.copy()
        
        # Handle pre-selected elements
        if pre_selected_elements:
            # Make sure pre-selected elements are in the candidate pool
            for elem in pre_selected_elements:
                if elem not in candidate_pool:
                    candidate_pool.append(elem)
            candidate_pool = sorted(list(set(candidate_pool)))
            
            # Adjust base_size if we have pre-selected elements
            actual_base_size = base_size + len(pre_selected_elements)
        else:
            pre_selected_elements = []
            actual_base_size = base_size
            
        # Create a new LP problem
        prob = pulp.LpProblem("AdditionGraphOptimizer", pulp.LpMinimize)
        
        # Decision variables
        # x[i] = 1 if candidate i is in the base set, 0 otherwise
        x = pulp.LpVariable.dicts("x", candidate_pool, cat=pulp.LpBinary)
        
        # Fix pre-selected elements
        for elem in pre_selected_elements:
            prob += x[elem] == 1
            
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
        prob += pulp.lpSum([x[i] for i in candidate_pool]) == actual_base_size
        
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
            solver = pulp.apis.coin_api.COIN_CMD(timeLimit=300)
            prob.solve(solver)
        else:
            # Fallback to default solver
            prob.solve()
        
        # Extract the solution
        if prob.status == pulp.LpStatusOptimal:
            # Get the selected base set (excluding pre-selected elements)
            base_set = [i for i in candidate_pool if pulp.value(x[i]) > 0.5 and i not in pre_selected_elements]
            
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
            
            if self.verbose:
                print(f"MIP optimization completed in {time.time() - start_time:.2f} seconds.")
                print(f"Base set: {base_set}")
                print(f"Total steps: {total_steps}")
                coverage = (len(self.target_numbers) - len(unreachable)) / len(self.target_numbers) * 100
                print(f"Coverage: {coverage:.1f}% "
                      f"({len(self.target_numbers) - len(unreachable)}/{len(self.target_numbers)})")
            
            return base_set, total_steps, unreachable
        else:
            if self.verbose:
                print(f"No optimal solution found. Status: {pulp.LpStatus[prob.status]}")
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
        return sorted(dependencies[target],reverse=True)

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
    saturated_bsgs = input_data.get('saturated_bsgs', True)
    min_range_length = input_data.get('min_range_length', 3)

    # Create optimizer
    optimizer = AdditionGraphMIPOptimizer(target_numbers, max_chain_length, verbose)
    
    # Run appropriate optimization method
    if saturated_bsgs:
        if verbose:
            print("Using saturated BSGS with MIP optimization")
            
        base_set, decompositions, unreachable, total_steps = optimizer.optimize_with_saturated_bsgs_and_mip(
            base_size, min_range_length)
    else:
        # Run standard MIP optimization
        if verbose:
            print("Using standard MIP optimization")
            
        base_set, total_steps, unreachable = optimizer.solve_with_mip(
            base_size, include_non_targets=True)
            
    # Get decompositions for each target number
    if base_set:
        decompositions = optimizer.get_decomposition(base_set)
    else:
        decompositions = {}
    
    # Convert dict to a list format for JSON
    decomposition_list = []
    for target in target_numbers:
        decomp = decompositions.get(target)
        decomposition_list.append({
            'target': target,
            'decomposition': decomp
        })
    print("Decomposition list")
    print(decomposition_list)
    
    # Convert unreachable set to list for JSON serialization
    unreachable_list = list(unreachable) if unreachable is not None else []
    
    # Prepare result
    result = {
        'base_set': base_set if base_set is not None else [],
        'base_set_size': len(base_set) if base_set is not None else 0,
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
    parser.add_argument('--saturated-bsgs', action='store_true', 
                    help='Use saturated BSGS to better utilize memory')
    parser.add_argument('--min-range', type=int, default=3, 
                    help='Minimum length of range to apply BSGS')
    args = parser.parse_args()
    
    # Read input
    if args.input:
        with open(args.input, 'r') as f:
            input_data = json.load(f)
    else:
        # Read from stdin
        input_data = json.load(sys.stdin)
    
    # Add command line options to input data
    if args.saturated_bsgs:
        input_data['saturated_bsgs'] = True
    
    if args.min_range is not None:
        input_data['min_range_length'] = args.min_range
    
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
