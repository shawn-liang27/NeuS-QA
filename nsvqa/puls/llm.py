from openai import OpenAI
import datetime
import json
import os
from nsvqa.puls.prompts import *

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

    def prompt(self, p):
        """Send a prompt to the LM and update conversation history"""
        user_message = {"role": "user", "content": [{"type": "text", "text": p}]}
        self.history.append(user_message)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.history,
            store=False,
        )
        assistant_response = response.choices[0].message.content
        assistant_message = {"role": "assistant", "content": [{"type": "text", "text": assistant_response}]}
        self.history.append(assistant_message)

        return assistant_response

    def run_puls(self, question):
        """Executes the two-stage PULS chain with matching few-shot styles."""
        # --- STAGE 1: Extraction ---
        self.history = [{"role": "system", "content": PROPOSITION_EXTRACTOR_SYSTEM}]
        # Format input to match examples
        prop_query = f"Question: \"{question}\""
        prop_res = self.prompt(prop_query)
        
        try:
            props_data = json.loads(prop_res)
            propositions = props_data.get("proposition", ["scene appears"])
        except:
            propositions = ["scene appears"]

        # --- STAGE 2: Logic Generation ---
        self.history.append({"role": "system", "content": TL_GENERATOR_SYSTEM})
        
        # STYLE MATCHING: Format this string exactly like the 'EXAMPLES' above
        spec_query = f"Question: \"{question}\" | Props: {propositions}"
        spec_res = self.prompt(spec_query)
        
        try:
            spec_data = json.loads(spec_res)
            specification = spec_data.get("specification", f"({propositions[0]})")
        except:
            specification = f"({propositions[0]})"

        return {
            "proposition": propositions,
            "specification": specification
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

