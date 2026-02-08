from nsvqa.nsvs.model_checker.frame_validator_multi_operators import FrameValidatorMulti
from nsvqa.nsvs.model_checker.stormpy import StormModelChecker
import re

class PropertyChecker:
    def __init__(self, proposition, specification, model_type, tl_satisfaction_threshold, detection_threshold):
        self.proposition = proposition
        self.tl_satisfaction_threshold = tl_satisfaction_threshold
        self.specification = self.generate_specification(specification)
        self.model_type = model_type
        self.detection_threshold = detection_threshold

        self.model_checker = StormModelChecker(
            proposition_set=self.proposition,
            ltl_formula=self.specification
        )
        self.frame_validator = FrameValidatorMulti(
            ltl_formula=self.specification,
            threshold_of_probability=self.detection_threshold
        )
        self.stages = self._decompose_specification(specification)
        
        print(f'[DEBUG] The specification stored in MODEL CHECKER IS {self.specification}')
    def generate_specification(self, specification_raw):
        return f"P>={self.tl_satisfaction_threshold:.2f} [ {specification_raw} ]"

    def validate_frame(self, frame_of_interest):
        return self.frame_validator.validate_frame(frame_of_interest)

    def check_automaton(self, automaton):
        return self.model_checker.check_automaton(
            transitions=automaton.transitions,
            states=automaton.states,
            model_type=self.model_type
        )

    def validate_tl_specification(self, specification):
        return self.model_checker.validate_tl_specification(specification)

    def _decompose_specification(self, spec):
        # Instead of stripping all parens, we split by the 'top-level' UNTILs
        # In Stormpy format, these are the 'U' operators that aren't inside nested parens
        # For simplicity, if we follow your 'Right-Nested' rule:
        
        stages = []
        # Logic: Find the strings between the 'U's
        # "((A & B | C ) U (D U E))" -> ["A & B | C", "D U E"]
        # Then we recursively handle "D U E" -> ["D", "E"]
        
        parts = self._recursive_until_split(spec) 
        
        for part in parts:
            # Group all propositions found in this temporal slice
            found = [p for p in self.proposition if p in part]
            if found:
                stages.append(found)
        return stages

    def get_num_stages(self):
        """Returns the dynamic length for all_detections."""
        return len(self.stages)

    def check_split(self, prop_label):
        """
        Returns the stage index (int) for a given proposition label.
        Example: If 'car' is in stage 2, returns 2.
        """
        # Clean the prop_label if necessary to match your internal props list
        for idx, stage_props in enumerate(self.stages):
            if prop_label in stage_props:
                return idx
        return None  # If the detected object isn't part of the logics
    
    def _recursive_until_split(self, spec):
        spec = spec.strip()
        # Remove outer-most surrounding parentheses if they enclose the whole spec
        if spec.startswith('(') and spec.endswith(')'):
            # Only strip if they are a matching pair for the whole string
            count = 0
            is_pair = True
            for i in range(len(spec)-1):
                if spec[i] == '(': count += 1
                elif spec[i] == ')': count -= 1
                if count == 0:
                    is_pair = False
                    break
            if is_pair:
                return self._recursive_until_split(spec[1:-1])

        # Find the top-level ' U ' or ' & ' (when used for sequences)
        depth = 0
        split_idx = -1
        for i in range(len(spec)):
            if spec[i] == '(': depth += 1
            elif spec[i] == ')': depth -= 1
            elif depth == 0:
                # Look for " U " (Until) or " & " (And)
                if spec[i:i+3] == " U ":
                    split_idx = i
                    break
                elif spec[i:i+3] == " & " and "EVENTUALLY" in spec:
                    split_idx = i
                    break

        if split_idx != -1:
            left = spec[:split_idx].strip()
            right = spec[split_idx+3:].strip()
            return [left] + self._recursive_until_split(right)
        else:
            return [spec]