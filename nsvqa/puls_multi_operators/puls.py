from nsvqa.puls_multi_operators.llm import *
from nsvqa.puls_multi_operators.prompts import *
import json
import os
import re
import logging

def clean_and_parse_json(raw_str):
    start = raw_str.find('{')
    end = raw_str.rfind('}') + 1
    json_str = raw_str[start:end]
    return json.loads(json_str)

def normalize_proposition(prop):
    # 1. Lowercase and handle common symbols first
    prop = prop.lower().replace("&", "_and_").replace("+", "_plus_")
    
    # 2. Replace dashes and spaces with underscores
    prop = re.sub(r"[\s\-\.]+", "_", prop)
    
    # 3. Remove any remaining characters that aren't letters, numbers, or underscores
    prop = re.sub(r'[^a-z0-9_]', '', prop)
    
    # 4. Clean up leading/trailing underscores
    prop = prop.strip("_")
    
    # Now "S&P 500" becomes "s_and_p_500" 
    # and "S.P. 500" becomes "s_p_500"
    if not prop or prop[0].isdigit():
        prop = "n_" + prop
    return prop

def process_specification(specification, propositions):
    # new_propositions = []
    # for prop in propositions:
    #     prop_cleaned = re.sub(r"^[^a-zA-Z]+|[^a-zA-Z]+$", "", prop)
    #     prop_cleaned = re.sub(r"\s+", "_", prop_cleaned)
    #     prop_cleaned = prop_cleaned.replace("'", "").replace("-", "_").lower()
    #     prop_cleaned = re.sub(r'[^a-zA-Z0-9_]', '', prop_cleaned)
    #     new_propositions.append(prop_cleaned)
    def is_balanced(s):
        count = 0
        for char in s:
            if char == '(': count += 1
            elif char == ')': count -= 1
            if count < 0: return False
        return count == 0

    new_propositions = [normalize_proposition(p) for p in propositions]

    replacements = sorted(
        list(zip(propositions, new_propositions)),
        key=lambda x: len(x[0]),
        reverse=True
    )
    for original, new in replacements:
        quoted_original = f'"{original}"'
        if quoted_original in specification:
            # Replace the quoted version with a single-quoted new label
            specification = specification.replace(quoted_original, f'"{new}"')
        elif original in specification:
            # Replace the unquoted version and add quotes
            specification = specification.replace(original, f'"{new}"')
            
    if " UNTIL " not in specification:
        if not is_balanced(specification):
            # Fallback: Strip and flatten for unary/binary safety
            # We keep the logical words but remove the broken structure
            specification = specification.replace("(", "").replace(")", "").strip()
    elif " UNTIL " in specification:
        # 2. Ambiguity Checker for UNTIL
        until_count = specification.count(" UNTIL ")
        if until_count > 1 and specification.count("(") < until_count:
                logging.warning(f"Generated Specification is invalid {specification}, Rebuilding using Right Nested Parentheses Rule")
                clean_parts = [p.strip().strip("()") for p in specification.split(" UNTIL ")]
                # Rebuild right-nested: (A UNTIL (B UNTIL (C)))
                nested = clean_parts[-1]
                nested = f"({clean_parts[-1]})"
                for p in reversed(clean_parts[:-1]):
                    nested = f"({p} UNTIL {nested})"
                specification = nested

    replacements = {
        "AND": "&",
        "OR": "|",
        "UNTIL": "U",
        "ALWAYS": "G",
        "EVENTUALLY": "F",
        "NOT": "!"
    }
    for word, symbol in replacements.items():
        specification = specification.replace(word, symbol)

    open_p = specification.count("(")
    close_p = specification.count(")")
    
    if open_p > close_p:
        specification += ")" * (open_p - close_p)
    elif close_p > open_p:
        specification = ("(" * (close_p - open_p)) + specification

    return new_propositions, specification


def PULS(prompt, id, save_dir, candidates=None, openai_key=None):
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
    save_dir = os.path.join(save_dir, id)
    os.makedirs(save_dir, exist_ok=True)
    llm = LLM(save_dir=save_dir)

    if candidates:
        cleaned_candidates = [c.split('. ', 1)[-1] for c in candidates]
    raw_results = llm.run_puls(prompt, cleaned_candidates)

    final_output = {}

    cleaned_props, processed_spec = process_specification(
        raw_results["specification"], 
        raw_results["proposition"]
    )
    
    final_output["proposition"] = cleaned_props
    final_output["specification"] = processed_spec
    final_output["raw_output"] = raw_results

    if not cleaned_props or not processed_spec:
        final_output["is_valid"] = False 
    else:
        final_output["is_valid"] = True

    saved_path = llm.save_history(id)
    final_output["saved_path"] = saved_path

    return final_output
