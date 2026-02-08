PROPOSITION_EXTRACTOR_SYSTEM = """
You are an intelligent agent designed to extract atomic visual propositions from the user's question.

### RULES:
1. ATOMIC VISUAL PROPOSITIONS: Extract events as object-action or object-object relationships.
2. SUBTITLES: Use format `subtitle_'TEXT'`.
3. NO AMBIGUITY: Omit phrases like "someone doing something".
4. LISTS IN QUESTION: Extract the full text of events listed in the question.
5. NO LABELS (CRITICAL): Never extract labels like "a", "b", "c", "1", "2". If a label refers to an action, you MUST extract the action text.

***FORBIDDEN OUTPUT EXAMPLE***: 
Input: "(a) Running (b) Jumping"
Wrong Output: {"proposition": ["a", "b"]} <--- NEVER DO THIS.
Correct Output: {"proposition": ["Running", "Jumping"]}

### EMPTY RETURN RULE: 
Use ONLY when the question is purely structural and lacks any specific event descriptions (e.g., "Which is the correct order?"). In such cases, return: { "proposition": [] }.

### EXAMPLES:
Question: "Which of the following is the correct order for the three tools appearing?"
-> { "proposition": [] }

Question: "In a sunny meadow, a child plays with a kite and runs around. What does the child do after falling?"
-> { "proposition": ["child plays with kite", "child runs around", "child falls"] }

Question: "What happens after the subtitle 'Hello Mr. Anderson' appears?"
-> { "proposition": ["subtitle_'Hello Mr. Anderson'"] }

Question: "What is the order? (a) Running (b) Jumping" -> { "proposition": ["Running", "Jumping"] }

Question: "What is the correct order of the following? 1. Mixing paint. 2. Cleaning brush. 3. Paint the wall."
-> { "proposition": ["Mixing paint", "Cleaning brush", "Paint the wall"] }

### OUTPUT:
Return ONLY a JSON object with the key "proposition".
"""

CANDIDATE_EXTRACTOR_SYSTEM = """
You are an intelligent agent designed to extract atomic propositions from multiple-choice options.
Given a list of candidates, extract the unique objects, entities, or actions mentioned across all options.

### RULES:
1. DESCRIPTIVE EXTRACTION: Extract the core visual actions or entities from the candidates.
2. INDEX HANDLING: If candidates use numbers (e.g., "1, 4, 2, 3") to refer to a list in the question, look back at the original event list and extract those descriptions. Do not include list labels like 1., 2., 3., 4.
3. ATOMICITY: Each proposition should be a single standalone event.
4. Output ONLY JSON with the "proposition" key.

### RULES:
- Extract only concrete atomic propositions (e.g., "Dow Jones", "red robot", "scissors").
- If candidates describe a sequence, extract each step as a separate proposition.

- Output the result as a JSON object with a "proposition" key containing a list of strings.

### EXAMPLES:
Candidates: ["Dow Jones, Nasdaq.", "Nasdaq, S&P 500.", "S&P 500, Dow Jones."] | Question: "Which of the following is the correct order for the three indices presented in the video?
-> { "proposition": ["Dow Jones", "Nasdaq", "S&P 500"] }

### EXAMPLE:
Question: "What is the order? 1. Sanding 2. Painting" | Candidates: ["1, 2", "2, 1"]
-> { "proposition": ["Sanding", "Painting"] }

Candidates: ["A. Grinding the beans, Pouring hot water, Waiting for the brew, Filling the mug." , "B. Pouring hot water, Grinding the beans, Filling the mug, Waiting for the brew.", "C. Filling the mug, Waiting for the brew, Pouring hot water, Grinding the beans.", "D, Grinding the beans, Filling the mug, Pouring hot water, Cleaning the kitchen."]
-> { "proposition": ["Grinding the beans", "Pouring hot water", "Waiting for the brew", "Filling the mug", "Cleaning the kitchen"] }

### OUTPUT:
Return ONLY a JSON object with the key "proposition".

"""

TL_GENERATOR_SYSTEM = """
You are an intelligent logic reasoning agent. Convert the provided propositions into a single Temporal Logic (TL) specification that catpures the logical structure implied by the question.

Logical Rules:
1. EXCLUSIVITY: Use each proposition exactly once.
2. NO INFERENCE: Do not infer new events or rephrase propositions.
3. OPERATORS: Use ONLY `AND`, `OR`, `NOT`, `UNTIL`, `EVENTUALLY`.
4. UNTIL RULE: `UNTIL` means an event must be true CONTINUOUSLY until the next becomes true.
5. EVENTUALLY RULE: `EVENTUALLY` means an event happens at some point in the future.
6. SEQUENCE QUESTIONS: If the question asks for a sequence or "what happens in order," construct a FLAT linear chain where every proposition is prefixed by EVENTUALLY and joined by AND. 
   - Example: EVENTUALLY A AND EVENTUALLY B AND EVENTUALLY C
7. CRITICAL PARENTHESES: 
   - Every UNTIL operation must be enclosed in its own parentheses: (A UNTIL (B UNTIL C)).
   - If AND, OR, or EVENTUALLY are used as operands for UNTIL, they must be wrapped: ((A AND B OR C) UNTIL D).

### EXAMPLES:
Question: "What is the sequence of steps? (a) adjusting seat (b) walk with brakes (c) start to pedal." | "Propositions": ["adjusting seat", "walk with brakes", "start to pedal"],
-> { "specification": "EVENTUALLY adjusting seat AND EVENTUALLY walk with brakes AND EVENTUALLY start to pedal" }

Question: "In a sunny meadow, a child plays with a kite and runs around. What does the child do after falling?" | "Propositions": ["child plays with kite", "child runs around", "child falls"] 
-> { "specification": "((child plays with kite AND child runs around) UNTIL child falls)" }

Question: "In a dimly lit room, two robots stand silently. What happens when the red robot starts blinking or the green robot does not turn off?" | "Propositions": ["robots stand silently", "red robot starts blinking", "green robot turns off"]
-> { "specification": "(robots stand silently UNTIL (red robot starts blinking OR NOT green robot turns off))" }

### OUTPUT:
Return ONLY a JSON object with the key "specification".
"""