import enum
import re
from nsvqa.nsvs.video.video_frame import VideoFrame

class SymbolicFilterRule(enum.Enum):
    NOT_PROPS = "not"
    AND_PROPS = "and"
    OR_PROPS = "or"

class FrameValidatorMulti:
    def __init__(
        self,
        ltl_formula: str,
        threshold_of_probability: float = 0.5,
    ):
        self.threshold_of_probability = threshold_of_probability

        # 1. Strip outer probability wrappers like P>=0.5 [ ... ]
        if '[' in ltl_formula and ']' in ltl_formula:
            ltl_formula = ltl_formula[ltl_formula.find('[') + 1:ltl_formula.rfind(']')]
        
        # 2. Extract every propositional block separated by temporal operators (U or F)
        # This regex splits by 'U' or 'F' while ignoring surrounding parentheses
        parts = re.split(r'\s+U\s+|\s+F\s+', ltl_formula)
        
        all_and = []
        all_or = []
        all_not = []
        
        # 3. Process each part to extract its atomic propositions
        for part in parts:
            rule = self.get_symbolic_rule_from_ltl_formula(part)
            
            if rule.get(SymbolicFilterRule.AND_PROPS):
                all_and.extend(rule[SymbolicFilterRule.AND_PROPS])
            if rule.get(SymbolicFilterRule.OR_PROPS):
                all_or.extend(rule[SymbolicFilterRule.OR_PROPS])
            if rule.get(SymbolicFilterRule.NOT_PROPS):
                # Handle single strings or lists of NOT propositions
                not_val = rule[SymbolicFilterRule.NOT_PROPS]
                if isinstance(not_val, list):
                    all_not.extend(not_val)
                else:
                    all_not.append(not_val)
                
        self.symbolic_verification_rule = {
            SymbolicFilterRule.AND_PROPS: all_and,
            SymbolicFilterRule.OR_PROPS: all_or,
            SymbolicFilterRule.NOT_PROPS: all_not if all_not else None
        }

    def validate_frame(self, frame: VideoFrame):
        """Validate frame."""
        thresholded_objects = frame.thresholded_detected_objects(self.threshold_of_probability)
        if len(thresholded_objects) > 0:
            return self.symbolic_verification(frame)
        else:
            return False

    def symbolic_verification(self, frame: VideoFrame):
        """Symbolic verification."""
        # 1. Negative constraints: If any 'NOT' proposition is detected, frame is invalid
        not_props = self.symbolic_verification_rule.get(SymbolicFilterRule.NOT_PROPS)
        if not_props:
            for prop in not_props:
                if frame.object_of_interest.get(prop):
                    if frame.object_of_interest[prop].get_detected_probability() >= self.threshold_of_probability:
                        return False

        # 2. Disjunctive constraints (OR): If an OR group is present, at least one must be true
        or_props = self.symbolic_verification_rule.get(SymbolicFilterRule.OR_PROPS)
        if or_props:
            or_satisfied = False
            for group in or_props:
                for prop in group:
                    if frame.object_of_interest.get(prop):
                        if frame.object_of_interest[prop].get_detected_probability() >= self.threshold_of_probability:
                            or_satisfied = True
                            break
                if or_satisfied: break
            if not or_satisfied:
                return False

        # 3. Conjunctive constraints (AND): Uses a majority voting logic as per the original code
        and_props = self.symbolic_verification_rule.get(SymbolicFilterRule.AND_PROPS)
        if and_props:
            for group in and_props:
                bad = 0
                total = len(group)
                for prop in group:
                    prob = 0
                    if frame.object_of_interest.get(prop):
                        prob = frame.object_of_interest[prop].get_detected_probability()
                    if prob < self.threshold_of_probability:
                        bad += 1
                # If more than half the propositions in the AND group are detected, we keep the frame
                if total > 2 * bad:
                    return True
        
        # 4. Final check for simple positive propositions
        has_positive_props = bool(and_props or or_props)
        return not has_positive_props

    def get_symbolic_rule_from_ltl_formula(self, ltl_formula: str) -> dict:
        """Helper to extract AND/OR/NOT groups from a non-temporal fragment."""
        symbolic_verification_rule = {}

        # Handle NOT (!)
        if "!" in ltl_formula:
            match = re.search(r'!\s*(?:\((.*?)\)|([^\s\)]+))', ltl_formula)
            if match:
                not_tl = (match.group(1) or match.group(2)).strip()
                symbolic_verification_rule[SymbolicFilterRule.NOT_PROPS] = not_tl
        else:
            symbolic_verification_rule[SymbolicFilterRule.NOT_PROPS] = None

        # Clean symbols and parentheses
        clean_formula = re.sub(r"[!GF]", "", ltl_formula).strip()
        clean_formula = re.sub(r"[()]", " ", clean_formula).strip()

        # Split and clean functions
        split_and_clean_and = lambda expr: [p.strip().strip('"') for p in re.split(r"\s*&\s*", expr) if p.strip()]
        split_and_clean_or = lambda expr: [p.strip().strip('"') for p in re.split(r"\s*\|\s*", expr) if p.strip()]

        if "|" in clean_formula:
            symbolic_verification_rule[SymbolicFilterRule.OR_PROPS] = [split_and_clean_or(clean_formula)]
            symbolic_verification_rule[SymbolicFilterRule.AND_PROPS] = []
        else:
            symbolic_verification_rule[SymbolicFilterRule.AND_PROPS] = [split_and_clean_and(clean_formula)]
            symbolic_verification_rule[SymbolicFilterRule.OR_PROPS] = []

        return symbolic_verification_rule