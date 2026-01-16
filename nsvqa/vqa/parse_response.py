import re
import random

def parse_multi_choice_response(response, len_candidates):
    candidates=["A", "B", "C", "D", "E", "F", "G", "H"]

    candidates = candidates[:len_candidates]

    if not response:
        return random.choice(candidates) 

    s = response.strip()
    
    # 1. Remove common prefixes (from utils.py)
    answer_prefixes = [
        "The best answer is", "The correct answer is", "The answer is",
        "The answer", "The best option is", "The correct option is",
        "Best answer:", "Best option:",
    ]
    for prefix in answer_prefixes:
        s = s.replace(prefix, "")

    if len(s) < 5 and s.endswith((".", ")")):
        s = s[:-1]

    # 2. Extract letter
    # If the response is very long (>10 words) and has no clear letter, fail/random guess
    if len(s.split()) > 10 and not re.search("[ABCDE]", s):
        return random.choice(candidates)

    matches = re.search(r"[ABCDE]", s)
    
    if matches is None:
        return random.choice(candidates)
        
    return matches[0] # Returns 'A', 'B', etc.