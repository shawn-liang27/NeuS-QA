from openai import OpenAI
import datetime
import json
import os
from nsvqa.puls_multi_operators.prompts import *
import logging

class LLM:
    def __init__(self, model="gpt-4o", history=None, save_dir=""):
        """Initialize LLM"""
        self.client = OpenAI()
        self.model = model
        if history:
            self.history = history
        else:
            self.history = []
        self.save_dir = save_dir
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    def run_puls(self, question, candidates):
        """Executes the two-stage PULS chain with matching few-shot styles."""
        # --- STAGE 1: Extraction ---
        self.history = [{"role": "system", "content": PROPOSITION_EXTRACTOR_SYSTEM}]
        prop_query = f"Question: \"{question}\""
        prop_res = self.prompt(prop_query)
        
        props_data = json.loads(prop_res)
        propositions = props_data.get("proposition", [])
        
        propositions = [p for p in propositions if p.strip()]
        if not propositions:
            if not candidates:
                return {
                    "proposition": [],
                    "specification": "",
                    "is_valid" : False,
                    "message" : {
                        "question": question,
                        "candidates": candidates,
                        "log" : f"fail to extract propositions, no candidates to fill"
                    }
                }
            
            self.history = [{"role": "system", "content": CANDIDATE_EXTRACTOR_SYSTEM}]

            prop_query = f"Candidates: {candidates} | Question: {question}"
            candidates_res = self.prompt(prop_query)
            candidates_data = json.loads(candidates_res)
            candidates_props = candidates_data.get("proposition", [])
            q_type = candidates_data.get('type', "SELECTION")

            if not candidates_props:
                return {
                    "proposition": [],
                    "specification": "",
                    "is_valid" : False,
                    "message" : {
                        "question": question,
                        "candidates": candidates,
                        "log" : f"fail to extract propositions, candidates exists, failed to extract from candidates"
                    }
                }
            # # Build Nested eventually: (F(A) AND (F(B) AND F(C)))
            # # This ensures all candidate entities are searched for in the video trace
            # nested_f = f'EVENTUALLY ("{propositions[-1]}")'
            # for p in reversed(propositions[:-1]):
            #     nested_f = f'(EVENTUALLY ("{p}") AND {nested_f})'
            if q_type == "SEQUENCE":
                spec_parts = [f'EVENTUALLY ("{p}")' for p in candidates_props]
                specification = " AND ".join(spec_parts)
            elif q_type == "SELECTION":
                spec_parts = [f'EVENTUALLY ("{p}")' for p in candidates_props]
                # specification = " OR ".join(spec_parts)
                specification = " OR ".join(spec_parts)
            else:
                logging.critical(f"Unidentified Question Type: {q_type}, defaulting to 'OR' Chaining Specification")
                spec_parts = [f'EVENTUALLY ("{p}")' for p in candidates_props]
                specification = " OR ".join(spec_parts)
            return {
                "proposition": candidates_props,
                "specification": specification,
                "is_valid": True,
                "message" : {
                        "question": question,
                        "candidates": candidates,
                        "log" : f"fail to extract propositions, candidates successfully filled in, specification built",
                        "question_type": q_type
                    }
            }

        # --- STAGE 2: Specification Generation ---
        self.history.append({"role": "system", "content": TL_GENERATOR_SYSTEM})
        
        spec_query = f"Question: \"{question}\" | Propositions: {propositions}"
        spec_res = self.prompt(spec_query)
        
        try:
            spec_data = json.loads(spec_res)
            specification = spec_data.get("specification", f"({propositions[0]})")
        except:
            specification = f"({propositions[0]})"
            return {
                "proposition": propositions,
                "specification": specification,
                "is_valid": False,
                "message" : {
                    "question": question,
                    "candidates": candidates,
                    "log" : f"propositions extracted sucessfully, specification generation failed"
                }
            }

        return {
            "proposition": propositions,
            "specification": specification,
            "is_valid": True,
            "message" : {
                "question": question,
                "candidates": candidates,
                "log" : f"propositions extracted sucessfully, specification generation successfully"
            }
        }

    def prompt(self, p):
        """Sends a message to GPT-4o with history and JSON constraints."""
        self.history.append({"role": "user", "content": p})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.history,
            response_format={"type": "json_object"},
            temperature=0, 
        )
        
        content = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": content})
        return content

    def save_history(self, id, suffix=""):
        """Save conversation history to a JSON file and return the save path"""
        if not self.save_dir:
            return None
        if suffix:
            filename = f"conversation_history_target_{suffix}_{id}.json"
        else:
            filename = f"conversation_history_target_{id}.json"

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name, extension = os.path.splitext(filename)
        timestamped_filename = f"{base_name}_{timestamp}{extension}"

        save_path = os.path.join(self.save_dir, timestamped_filename)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=4, ensure_ascii=False)
        return save_path

