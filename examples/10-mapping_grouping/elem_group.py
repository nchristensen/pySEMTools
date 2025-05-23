#!/usr/bin/env python

import os
import sys
sys.path.append("/home/hpc/iwst/iwst115h/forked_git/pySEMTools") # add path to pynek tools
from typing import cast
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    num_elements_per_section = os.environ.get("num_elements_per_section")
    pattern_factor = os.environ.get("pattern_factor")
    num_pattern_cross_sections = os.environ.get("num_pattern_cross_sections")

    # Check if any variable is missing
    if None in [num_elements_per_section, pattern_factor, num_pattern_cross_sections]:
        print("Error: Missing required environment variables.", file=sys.stderr)
        sys.exit(1)

    num_elements_per_section = int(cast(str, num_elements_per_section))
    pattern_factor = int(cast(str, pattern_factor))
    num_pattern_cross_sections = int(cast(str, num_pattern_cross_sections))
else:
    num_elements_per_section = None
    pattern_factor = None
    num_pattern_cross_sections = None

num_elements_per_section = comm.bcast(num_elements_per_section, root=0)
pattern_factor = comm.bcast(pattern_factor, root=0)
num_pattern_cross_sections = comm.bcast(num_pattern_cross_sections, root=0)

import numpy as np
from pysemtools.io.reorder_data import (
    generate_data, calculate_face_differences, calculate_face_averages, calculate_face_normals,
    face_mappings, compare_mappings, compare_methods, reduce_face_normals, directions_face_normals,
    reorder_assigned_data, correct_axis_signs
)

elem_pattern_crosssections = num_elements_per_section * pattern_factor; # No. of elements in each pattern cross-section
total_elements = elem_pattern_crosssections * num_pattern_cross_sections # Total elements in the file.
elements_per_rank = total_elements // size  # No. of elements in each rank.
ranks_per_section = num_elements_per_section // elements_per_rank # No. of ranks per each cross-section
ranks_per_pattern_section = ranks_per_section * pattern_factor # No. of ranks per each pattern cross-section
leftover = total_elements % size  # Elements that don’t fit evenly in the rank (if any)

def validate_rank_size(total_elements: int, size: int, num_elements_per_section: int) -> bool:
   
    # Criterion 1: elements_per_rank should be exactly divisible
    elements_per_rank = total_elements // size
    if total_elements % size != 0:
        print(f"Error: Criteria 1 Failed,  {total_elements} % {size} != 0 (elements_per_rank = {elements_per_rank})") if rank == 0 else None
        return False
    
    # Criterion 2: ranks_per_section should be exactly divisible
    ranks_per_section = num_elements_per_section // elements_per_rank
    if num_elements_per_section % elements_per_rank != 0:
        print(f"Error: Criteria 2 Failed, {num_elements_per_section} % {elements_per_rank} != 0 (ranks_per_section = {ranks_per_section})") if rank == 0 else None
        return False
    
    print(f"Both Criterias Passed: elements_per_rank = {elements_per_rank}, ranks_per_section = {ranks_per_section}") if rank == 0 else None
    return True

def find_valid_sizes(total_elements: int, num_elements_per_section: int, min_size=200, max_size=2000):

    valid_sizes = []

    for size in range(min_size, max_size + 1):
        if total_elements % size == 0:
            elements_per_rank = total_elements // size
            if num_elements_per_section % elements_per_rank == 0:
                valid_sizes.append(size)

    if not valid_sizes:
        print("No valid sizes found. Consider adjusting your min/max limits.")
    
    print("Valid size options:", valid_sizes)
        
    return valid_sizes

if rank == 0:
    check = validate_rank_size(total_elements, size, num_elements_per_section)
    print(f"Validation passed! Running main computation with size {size}...\n")
else:
    check = None 
    
check = comm.bcast(check, root=0)

def element_grouping() -> None:
    
    # fname_3d = "/home/woody/iwst/iwst115h/Work/Checkpoint_files/field0.f00024"
    fname_3d = "/home/woody/iwst/iwst115h/Work/Checkpoint_files/field0.f00000"
    
    # fname_out = "/home/woody/iwst/iwst115h/Work/writefiles/unstruct_new/field_out_0.f00024"
    fname_out = "/home/woody/iwst/iwst115h/Work/writefiles/field/fieldout0.f00000"
    
    assigned_data = generate_data(fname_3d, fname_out)

    # Methods for cal. the difference and normal vectors
    assigned_r_diff, assigned_s_diff, assigned_t_diff = calculate_face_differences(assigned_data)
    assigned_r_avg_diff, assigned_s_avg_diff, assigned_t_avg_diff = calculate_face_averages(assigned_data)
    assigned_normals = calculate_face_normals(assigned_data)

    # Output:
    # print(f"Rank {rank} has Face differences:", assigned_r_diff.shape, assigned_s_diff.shape, assigned_t_diff.shape)
    # print(f"Rank {rank} has Average differences:", assigned_r_avg_diff.shape, assigned_s_avg_diff.shape, assigned_t_avg_diff.shape)
    # print(f"Rank {rank} has Normals shape:", assigned_normals.shape)

    # Method-1: Face_differences
    mappings_fd = face_mappings(assigned_r_diff, assigned_s_diff, assigned_t_diff)
    # Method-2: Face_averages
    mappings_fa = face_mappings(assigned_r_avg_diff, assigned_s_avg_diff, assigned_t_avg_diff)
    

    if rank == 0:
        ref_mapping_fd = mappings_fd[0]
    else:
        ref_mapping_fd = None
        
    ref_mapping_fd = comm.bcast(ref_mapping_fd, root=0)
    
    compare_mappings(mappings_fa, ref_mapping_fd, "mappings_fa_before_reorder")
    compare_mappings(mappings_fd, ref_mapping_fd, "mappings_fd_before_reorder")
    compare_methods(mappings_fd, mappings_fa)

    # Reordering of subset        
    assigned_data_new = reorder_assigned_data(assigned_data, mappings_fd, ref_mapping_fd)
    r_diff_new, s_diff_new, t_diff_new = calculate_face_differences(assigned_data_new)

    if rank == 0:
        extract_reference_r = r_diff_new[0]
        extract_reference_s = s_diff_new[0]
        extract_reference_t = t_diff_new[0]
    else:
        extract_reference_r = None
        extract_reference_s = None
        extract_reference_t = None
        
    extract_reference_r = comm.bcast(extract_reference_r, root=0)
    extract_reference_s = comm.bcast(extract_reference_s, root=0)
    extract_reference_t = comm.bcast(extract_reference_t, root=0)
    
    assigned_data_final = correct_axis_signs(assigned_data_new, extract_reference_r, extract_reference_s, extract_reference_t, r_diff_new, s_diff_new, t_diff_new)

    r_diff_final, s_diff_final, t_diff_final = calculate_face_differences(assigned_data_final)
    final_mappings_fd = face_mappings(r_diff_final, s_diff_final, t_diff_final)

    r_avg_diff_final, s_avg_diff_final, t_avg_diff_final = calculate_face_averages(assigned_data_final)
    final_mappings_fa = face_mappings(r_avg_diff_final, s_avg_diff_final, t_avg_diff_final)
    
    assigned_normals = calculate_face_normals(assigned_data_final)
    averaged_normals =  reduce_face_normals(assigned_normals)
    final_mappings_fn, flow_directions = directions_face_normals(averaged_normals)
    flow_directions = np. array(flow_directions)

    # Compare all methods:
    compare_mappings(final_mappings_fa, ref_mapping_fd, "mappings_fa_after_reorder")
    compare_mappings(final_mappings_fd, ref_mapping_fd, "mappings_fd_after_reorder")
    compare_methods(final_mappings_fd, final_mappings_fa)
    compare_mappings(final_mappings_fn, ref_mapping_fd, "mappings_fn_after_reorder")
        
    def truncate_reduce(data, decimals):
        trial = data[:, :, :, :, 0:2]  # Select specific slices
        # trial = trial.astype(np.float64)
        factor = 10.0 ** decimals
        return np.floor(trial * factor) / factor
    
    if rank == 0:
        idx_ranges = []
        start_idx = 0
        # Distribute leftovers among initial ranks
        for i in range(size):
            adjusted_chunk_size = elements_per_rank + (1 if i < leftover else 0)  # Distribute leftovers
            end_idx = start_idx + adjusted_chunk_size
            idx_ranges.append((start_idx, end_idx))
            start_idx = end_idx 
    else:
        idx_ranges = None
    idx_ranges = comm.bcast(idx_ranges, root=0)

    full_rank_list = list(range(ranks_per_pattern_section))        
    ndim1, ndim2, ndim3, ndim4 = assigned_data_final.shape[1:]

    if rank in full_rank_list:
        assigned_data_final_array  = np.empty((elements_per_rank, ndim1, ndim2, ndim3, ndim4))
    else:
        assigned_data_final_array = None
        
    matched_pairs = []  # store tuples of (source_global_idx, target_global_idx)
    bag = {i: [] for i in range(elements_per_rank)}

    #send data to the ranks
    if rank in full_rank_list:
        start_index, end_index = idx_ranges[0]
        assigned_data_final_array[start_index:end_index] = assigned_data_final
        data_to_send = assigned_data_final_array[start_index:end_index]
        target_ranks = [rank + ranks_per_pattern_section * i for i in range(1, num_pattern_cross_sections)]
        valid_targets = [target for target in target_ranks if target < size]
            
        for target in valid_targets:
            comm.Send(data_to_send, dest=target, tag=rank)
    
    elif rank not in full_rank_list:
        start_index, end_index = idx_ranges[rank]
        size_to_receive = end_index - start_index
        temp_buffer = np.empty((size_to_receive, ndim1, ndim2, ndim3, ndim4))
        source_rank = rank % ranks_per_pattern_section
        
        # comm.Recv(temp_buffer, source=0, tag=99)
        comm.Recv(temp_buffer, source=source_rank, tag=source_rank)
            
        rounded_data = truncate_reduce(assigned_data_final, decimals=6)
            
        rounded_data_received = truncate_reduce(temp_buffer, decimals=6)
            
        # Compare with local rounded_data
        matched_indices = []
        if np.allclose(rounded_data, rounded_data_received, atol=1e-4):
            for i in range(elements_per_rank):
                source_global_idx = source_rank * elements_per_rank + i
                target_global_idx = rank * elements_per_rank + i
                # Append to bag
                if source_global_idx not in bag:
                    bag[source_global_idx] = []
                bag[source_global_idx].append(target_global_idx)
        else:
            print(f"[NO MATCH] Rank {rank} did not match with {source_rank}")
            
    all_bag_data = comm.gather(bag, root=0)

    if rank == 0:
        combined_bag = {}

        for partial_bag in all_bag_data:
            for src_idx, targets in partial_bag.items():
                if src_idx not in combined_bag:
                    combined_bag[src_idx] = []
                combined_bag[src_idx].extend(targets)

        # Build final output: list of lists like [src, tgt1, tgt2, ...]
        final_bag = [[src_idx] + sorted(set(targets)) for src_idx, targets in sorted(combined_bag.items())]
        bag_array = np.array(final_bag, dtype=int)

        print("\nFinal grouping of elements:")
        for row in bag_array:
            print(row)
            
if check:    
    element_grouping()    
elif rank ==0:
    print(f"Error:: Invalid size: {size}")
    suggested_sizes  = find_valid_sizes(total_elements, num_elements_per_section)