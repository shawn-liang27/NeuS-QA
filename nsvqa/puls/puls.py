from nsvqa.puls.llm import *
from nsvqa.puls.prompts import *
import json
import os
import re

def clean_and_parse_json(raw_str):
    start = raw_str.find('{')
    end = raw_str.rfind('}') + 1
    json_str = raw_str[start:end]
    return json.loads(json_str)

def process_specification(specification, propositions):
    mapping = {}
    new_propositions = []
    
    for prop in propositions:
        original_prop = prop
        
        # 1. Strip leading/trailing non-alphanumerics (including quotes)
        prop = re.sub(r"^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$", "", prop)
        
        if prop.startswith("subtitle"):
            # Remove "subtitle", then strip any internal quotes or extra spaces
            raw_text = prop.replace("subtitle", "").strip("_'\" ")
            clean_text = re.sub(r"\s+", "_", raw_text)
            # GATE: Remove any remaining single or double quotes inside the text
            clean_text = re.sub(r"['\"]", "", clean_text)
            clean_text = re.sub(r'[^a-zA-Z0-9_]', '', clean_text)
            clean_text = re.sub(r"_+", "_", clean_text)
            
            # Formatted without internal single quotes
            prop_cleaned = f"subtitle_{clean_text.lower()}"
        else:
            # GATE: Strip all internal quotes for standard propositions
            prop_cleaned = prop.replace("'", "").replace('"', "").replace("-", "_").lower()
            prop_cleaned = re.sub(r"\s+", "_", prop_cleaned)
            prop_cleaned = re.sub(r'[^a-zA-Z0-9_]', '', prop_cleaned)
            prop_cleaned = re.sub(r"_+", "_", prop_cleaned)

        new_propositions.append(prop_cleaned)
        mapping[original_prop] = prop_cleaned

    # Replace Logic Keywords
    logic_map = {
        "AND": "&", "OR": "|", "UNTIL": "U", 
        "ALWAYS": "G", "EVENTUALLY": "F", "NOT": "!"
    }
    for word, symbol in logic_map.items():
        specification = specification.replace(word, symbol)

    # Sort by length descending to prevent partial matching
    replacements = sorted(mapping.items(), key=lambda x: len(x[0]), reverse=True)
    
    for original, new in replacements:
        # This regex looks for the original proposition potentially wrapped in quotes
        # and replaces the whole thing with a clean "new_prop" (single outer quotes only)
        pattern = re.escape(original)
        # Replacing matching segments with "new" wrapped in double quotes
        specification = re.sub(rf'[\'\"\\ ]*{pattern}[\'\"\\ ]*', f' "{new}" ', specification)
    
    # Final cleanup of whitespace
    specification = " ".join(specification.split())

    return new_propositions, specification

def PULS(prompt, id, save_dir, openai_key=None):
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
    save_dir = os.path.join(save_dir, id)
    os.makedirs(save_dir, exist_ok=True)
    llm = LLM(save_dir=save_dir)
    
    raw_results = llm.run_puls(prompt)
    print(f'[DEBUG] RAW PULS: {raw_results}')
    final_output = {}

    cleaned_props, processed_spec = process_specification(
        raw_results["specification"], 
        raw_results["proposition"]
    )

    final_output = {
        "proposition": cleaned_props,
        "specification": processed_spec
    }
    saved_path = llm.save_history(id)
    final_output["saved_path"] = saved_path

    return final_output
