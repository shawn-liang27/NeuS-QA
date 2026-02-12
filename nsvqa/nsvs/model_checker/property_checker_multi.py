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
        print(f'[DEBUG] The symbolic rule stored in FrameValidator is {self.frame_validator.symbolic_verification_rule}')
        print(f'[DEBUG] Stages decomposition: {self.stages}')
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
        # Determine if this is a sequence discovery (Relaxed) or order verification (Strict)
        # If there is no 'U' (Until), we collapse into a single stage for global evidence
        if " U " not in spec:
            # One stage containing all unique propositions mentioned in the spec
            all_props = [p for p in self.proposition if p in spec]
            return [all_props] if all_props else []

        # If 'U' is present, proceed with multi-stage decomposition for handover logic
        stages = []
        parts = self._recursive_until_split(spec) 
        
        for part in parts:
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
        # 1. Remove outer-most balanced parentheses
        if spec.startswith('(') and spec.endswith(')'):
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

        # 2. Find ONLY the top-level ' U ' (Until)
        depth = 0
        split_idx = -1
        for i in range(len(spec)):
            if spec[i] == '(': depth += 1
            elif spec[i] == ')': depth -= 1
            elif depth == 0:
                # We ONLY split on Until to define temporal stages
                if spec[i:i+3] == " U ":
                    split_idx = i
                    break

        # 3. Recurse if a temporal boundary was found
        if split_idx != -1:
            left = spec[:split_idx].strip()
            right = spec[split_idx+3:].strip()
            return [left] + self._recursive_until_split(right)
        else:
            return [spec]