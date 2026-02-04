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

def group_with_gaps(nums, max_gaps=2):
    if not nums:
        return []

    groups = []
    current_group = [nums[0]]

    for i in range(1, len(nums)):
        diff = nums[i] - nums[i-1]
        
        if diff <= max_gaps + 1:
            if diff > 1:
                current_group.extend(range(nums[i-1] + 1, nums[i] + 1))
            else:
                current_group.append(nums[i])
        else:
            groups.append(current_group)
            current_group = [nums[i]]

    groups.append(current_group)
    return groups

def intersection_with_gaps(indices, max_gaps=8): 
    non_empty = [set(s) for s in indices if s]
    
    if not non_empty:
        return []
    if len(non_empty) == 1:
        combined = sorted(list(non_empty[0]))
    else:
        intersected = set.intersection(*non_empty)
        combined = sorted(list(intersected))

    if not combined:
        return []

    groups = group_with_gaps(combined, max_gaps)
    return max(groups, key=len) if groups else []


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

def reconcile_sparse_ltl(video_info, p_indices, q_indices, automaton_foi, target_window, BRIDGE_MULT=30, CONTEXT_SECONDS=10, TOLERANCE=8):

    fps = video_info["fps"]
    sample_rate = video_info.get("sample_rate", 1)
    frame_count = video_info["frame_count"]
    
    MAX_GAPS = BRIDGE_MULT * fps
    WINDOW = CONTEXT_SECONDS * fps      # 300
    tolerance = 8 * fps

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
    
    tw_before, tw_after = resolve_target_window(target_window)
    final_ranges = []
    
    for foi_group in final_foi_groups:
        group_min, group_max = min(foi_group), max(foi_group)
        
        start_ext = max(0, int(group_min + tw_before * fps))
        end_ext = min(frame_count - 1, int(group_max + tw_after * fps))
        
        final_ranges.append((start_ext, end_ext))
        
    final_ranges.sort()
    merged = []
    if final_ranges:
        curr_start, curr_end = final_ranges[0]
        for next_start, next_end in final_ranges[1:]:
            if next_start <= curr_end + MAX_GAPS:
                curr_end = max(curr_end, next_end)
            else:
                merged.append((curr_start, curr_end))
                curr_start, curr_end = next_start, next_end
        merged.append((curr_start, curr_end))

    return merged