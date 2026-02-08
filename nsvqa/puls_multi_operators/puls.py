from nsvqa.puls.llm import *
from nsvqa.puls.prompts import *
import json
import os
import re
import logging

def clean_and_parse_json(raw_str):
    start = raw_str.find('{')
    end = raw_str.rfind('}') + 1
    json_str = raw_str[start:end]
    return json.loads(json_str)

def process_specification(specification, propositions):
    new_propositions = []
    for prop in propositions:
        prop_cleaned = re.sub(r"^[^a-zA-Z]+|[^a-zA-Z]+$", "", prop)
        prop_cleaned = re.sub(r"\s+", "_", prop_cleaned)
        prop_cleaned = prop_cleaned.replace("'", "").replace("-", "_").lower()
        prop_cleaned = re.sub(r'[^a-zA-Z0-9_]', '', prop_cleaned)
        new_propositions.append(prop_cleaned)

    replacements = sorted(
        list(zip(propositions, new_propositions)),
        key=lambda x: len(x[0]),
        reverse=True
    )
    for original, new in replacements:
        if specification.count(original) == 1:
            specification = specification.replace(original, f'"{new}"')
            
    specification = re.sub(r'EVENTUALLY\s+("?[a-zA-Z0-9_]+"?)(?!\s*\))', r'EVENTUALLY (\1)', specification)

    # 2. Ambiguity Checker for UNTIL
    until_count = specification.count(" UNTIL ")
    if until_count > 1:
        # Check if the operators are already "enclosed" by parentheses
        # If the number of open parens is less than the number of binary splits needed
        if specification.count("(") < until_count:
            logging.warning(f"Generated Specification is invalid {specification}, Rebuilding using Right Nested Parentheses Rule")
            clean_parts = [p.strip().strip("()") for p in specification.split(" UNTIL ")]
            
            # Rebuild right-nested: (A UNTIL (B UNTIL (C)))
            nested = clean_parts[-1]
            if not nested.startswith("("):
                nested = f"({nested})"
            
            for p in reversed(clean_parts[:-1]):
                nested = f"({p} UNTIL {nested})"
            specification = nested

    open_p = specification.count("(")
    close_p = specification.count(")")
    
    if open_p > close_p:
        specification += ")" * (open_p - close_p)
    elif close_p > open_p:
        specification = ("(" * (close_p - open_p)) + specification

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

    return new_propositions, specification

def PULS(prompt, id, save_dir, openai_key=None):
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
    save_dir = os.path.join(save_dir, id)
    os.makedirs(save_dir, exist_ok=True)
    llm = LLM(save_dir=save_dir)

    full_prompt = find_prompt(prompt)
    llm_output = llm.prompt(full_prompt)
    parsed = clean_and_parse_json(llm_output)

    final_output = {}

    cleaned_props, processed_spec = process_specification(parsed["specification"], parsed["proposition"])
    if not cleaned_props or not processed_spec:
        final_output["is_valid"] = False 
    else:
        final_output["is_valid"] = True 
    
    final_output["proposition"] = cleaned_props
    final_output["specification"] = processed_spec

    saved_path = llm.save_history(id)
    final_output["saved_path"] = saved_path

    return final_output