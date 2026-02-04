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

    # specification = specification.replace("U", "& F")
    # if 'G "' in specification:
    #     specification = specification.replace('G "', 'F "')

    return new_propositions, specification

def PULS(prompt, id, save_dir, openai_key=None):
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
    save_dir = os.path.join(save_dir, id)
    os.makedirs(save_dir, exist_ok=True)
    llm = LLM(save_dir=save_dir)
    
    raw_results = llm.run_puls(prompt)

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
