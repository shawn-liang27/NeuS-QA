import re

def resolve_target_window(target_window):
    inner = target_window.strip()[1:-1]
    parts = inner.split(',')
    result = []
    for part in parts:
        part = part.strip()
        match = re.search(r'([+-])\s*(\d+)', part)
        if match:
            sign, num = match.groups()
            result.append(int(sign + num))
        else:
            result.append(0)
    return result

# def group_with_gaps(nums, max_gaps=2):
#     if not nums:
#         return []

#     groups = []
#     current_group = [nums[0]]

#     for i in range(1, len(nums)):
#         diff = nums[i] - nums[i-1]
        
#         if diff <= max_gaps + 1:
#             if diff > 1:
#                 current_group.extend(range(nums[i-1] + 1, nums[i] + 1))
#             else:
#                 current_group.append(nums[i])
#         else:
#             groups.append(current_group)
#             current_group = [nums[i]]

#     groups.append(current_group)
#     return groups

def group_with_gaps(nums, max_gaps):
    if not nums:
        return []

    # Sort to ensure we are checking temporal neighbors
    nums = sorted(list(nums))
    
    groups = []
    current_group = [nums[0]]

    for i in range(1, len(nums)):
        # Calculate the jump between this detection and the previous one
        diff = nums[i] - nums[i-1]
        
        # If the jump is smaller than our allowed gap, they belong to the same event
        if diff <= max_gaps:
            current_group.append(nums[i])
        else:
            # Otherwise, finish this group and start a new one
            groups.append(current_group)
            current_group = [nums[i]]

    groups.append(current_group)
    return groups

def intersection_with_gaps(indices, max_gaps=8):
    non_empty = [s for s in indices if s]
    if len(non_empty) == 1:
        return sorted(list(non_empty[0]))

    combined = sorted(list(set().union(*indices)))
    
    if not combined:
        return []

    groups = group_with_gaps(combined, max_gaps)
    largest_set = max(groups, key=lambda g: (max(g) - min(g)))

    return sorted(largest_set)


def find_soft_handover(p_indices, q_indices, tolerance_frames):
    """
    Finds frames from both sets that are within 'tolerance_frames' of each other.
    """
    handover_points = set()
    p_list = sorted(list(p_indices))
    q_list = sorted(list(q_indices))
    
    if not p_list or not q_list:
        return set()

    i, j = 0, 0
    while i < len(p_list) and j < len(q_list):
        p_val = p_list[i]
        q_val = q_list[j]
        
        if abs(p_val - q_val) <= tolerance_frames:
            handover_points.add(p_val)
            handover_points.add(q_val)
            if p_val < q_val: i += 1
            else: j += 1
        elif p_val < q_val:
            i += 1
        else:
            j += 1
            
    return handover_points

def reconcile_sparse_ltl(video_info, p_indices, q_indices, automaton_foi, target_window, BRIDGE_MULT=3, CONTEXT_SECONDS=5, TOLERANCE=5):

    fps = video_info["fps"]
    sample_rate = video_info.get("sample_rate", 1)
    frame_count = video_info["frame_count"]
    
    MAX_GAPS = BRIDGE_MULT * fps
    WINDOW = CONTEXT_SECONDS * fps      # 300
    tolerance = TOLERANCE * fps

    if q_indices:
        handover_all = find_soft_handover(p_indices, q_indices, tolerance)

        if handover_all:
            handover_sorted = sorted(list(handover_all))
            handover_clusters = group_with_gaps(handover_sorted, max_gaps=MAX_GAPS)
            valid_segments = set()
            
            for cluster in handover_clusters:
                t_start = min(cluster)
                t_end = max(cluster)
                
                event_p = {idx for idx in p_indices if (t_start - WINDOW) <= idx <= t_end}
                event_q = {idx for idx in q_indices if t_start <= idx <= (t_end + WINDOW)}
                
                valid_segments.update(event_p)
                valid_segments.update(event_q)

            final_set = valid_segments | set(automaton_foi)
        else:
            final_set = set(automaton_foi)
    else:
        final_set = p_indices | set(automaton_foi)
    if not final_set:
        return [-1]

    final_foi_groups = group_with_gaps(sorted(list(final_set)), max_gaps=MAX_GAPS)

    return final_foi_groups

def reconcile_dynamic_ltl(video_info, all_detections, automaton_foi, target_window_before, target_window_after,
                          BRIDGE_MULT=5, CONTEXT_SECONDS=5, TOLERANCE=5):
    """
    Handles N-stage sequential handovers (0->1, 1->2, ... N-1->N)
    to identify Frames of Interest (FOI) across a complex LTL chain.
    """
    fps = video_info["fps"]
    frame_count = video_info["frame_count"]
    
    # Configuration
    MAX_GAPS = BRIDGE_MULT * fps
    WINDOW = CONTEXT_SECONDS * fps
    tolerance = TOLERANCE * fps

    # Start with the ground truth from the automaton
    final_set = set(automaton_foi)
    valid_segments = set()

    # Iterate through each temporal transition (Stage i -> Stage i+1)
    # If all_detections has 3 sets, we check (0,1) and (1,2)
    if len(all_detections) == 1:
        stage_1 = sorted(list(all_detections[0]))
        segments = group_with_gaps(stage_1, max_gaps=MAX_GAPS)
        for cluster in segments:
            t_start = min(cluster)
            t_end = max(cluster)
            
            event_before = {idx for idx in stage_1 if (t_start - max(WINDOW, int(target_window_before / 2) * fps)) <= idx <= t_end}
            event_after = {idx for idx in stage_1 if t_start <= idx <= (t_end + max(WINDOW, int(target_window_after / 2) * fps))}
            
            valid_segments.update(event_before)
            valid_segments.update(event_after)
    else:
        for i in range(len(all_detections) - 1):
            p_indices = all_detections[i]
            q_indices = all_detections[i+1]

            if not p_indices or not q_indices:
                continue

            # Find where Stage i hands over to Stage i+1
            handover_all = find_soft_handover(p_indices, q_indices, tolerance)

            if handover_all:
                handover_sorted = sorted(list(handover_all))
                # Group handover points into clusters to find distinct event boundaries
                handover_clusters = group_with_gaps(handover_sorted, max_gaps=MAX_GAPS)
                
                for cluster in handover_clusters:
                    t_start = min(cluster)
                    t_end = max(cluster)
                    
                    # Capture context around the transition
                    # event_p: Context before and during the handover
                    event_p = {idx for idx in p_indices if (t_start - max(WINDOW, target_window_before)) <= idx <= t_end}
                    # event_q: Context during and after the handover
                    event_q = {idx for idx in q_indices if t_start <= idx <= (t_end + max(WINDOW, target_window_after))}
                    
                    valid_segments.update(event_p)
                    valid_segments.update(event_q)

    # Union the segments identified by handovers with the automaton's path
    final_set.update(valid_segments)

    # Fallback: If no sequential handovers were found, take the union of all hits
    if not final_set:
        final_set = set().union(*all_detections)
    
    if not final_set:
        return [-1]

    # Group the final indices to create continuous video ranges
    final_foi_groups = group_with_gaps(sorted(list(final_set)), max_gaps=MAX_GAPS)
    
    # Convert clusters to (start, end) tuples for the cropper
    return [(min(g), max(g)) for g in final_foi_groups]