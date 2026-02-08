def find_prompt(prompt):
    full_prompt = f"""
You are an intelligent agent designed to extract structured representations from video question prompts. You will operate in two stages: (1) proposition extraction and (2) TL specification generation.

Stage 1: Proposition Extraction

Given an input question about a video, extract the atomic propositions that describe the underlying events or facts explicity referenced in the question. These propositions should describe object-action or object-object relationships stated in the question — avoid making assumptions or inferring any additional events. Avoid TL keywords such as 'and', 'or', 'not', 'until'.
Do not include ambiguous propositions that lack specificity. For instance, phrases like "guy does something" are ambiguous and should be omitted. Instead, focus on concrete actions or relationships. For example, given the prompt "In a bustling park, a child kicks a ball. What happens when the ball hits the bench?", the correct propositions are ["child kicks ball", "ball hits bench"].
If a proposition mentions subtitles/captions, the format of the proposition is the word "subtile_" followed by the subtitle in single quotes. Do NOT add words like "appears"/"says"/"mentions" after the subtitle; follow this format to create the individual proposition. For example, given the prompt "After the man gets up, what happens after the subtitle 'Hello Mr. Anderson' appeared?", the correct propositions are ["man gets up", "subtitle_'Hello Mr. Anderson'"].

Stage 2: TL Specification Generation

Using only the list of the propositions extracted in Stage 1, generate a single Temporal Logic (TL) specification that catpures the sequence of logical structure implied by the question. 

Rules:
- The formula must use each proposition **exactly once**.
- Do **not** infer new events or rephrase propositions.
- Use only the TL operators: `AND`, `OR`, `NOT`, `UNTIL`, `EVENTUALLY`.
- `EVENTUALLY` means an event happens at some point in the future.
- `UNTIL` means the an event must be true CONTINUOUSLY until the next becomes true.
- **CRITICAL:**: 
    - BINARY GROUPING RULE: Every UNTIL operation must be enclosed in its own set of parentheses. For a sequence, always group from the right.
    - If Unary operator AND, OR, NOT, EVENTUALLY is used on the left or the right side of the UNTIL operation, they must be enclosed in parentheses.
    Example:
      - INCORRECT: A UNTIL B UNTIL C
      - CORRECT: (A UNTIL (B UNTIL C)
      - INCORRECT: A AND B UNTIL C
      - CORRECT: ((A AND B) UNTIL C)
      

**Examples**

Example 1: "In a sunny meadow, a child plays with a kite and runs around. What does the child do after falling?"
Output:
{{
  "proposition": ["child plays with kite", "child runs around", "child falls"],
  "specification": "((child plays with kite AND child runs around) UNTIL child falls)"
}}

Example 2: "In a dimly lit room, two robots stand silently. What happens when the red robot starts blinking or the green robot does not turn off?"
Output:
{{
  "proposition": ["robots stand silently", "red robot starts blinking", "green robot turns off"],
  "specification": "(robots stand silently UNTIL (red robot starts blinking OR NOT green robot turns off))"
}}

Example 3: "What is the sequence of steps? (a) adjusting seat (b) walk with brakes (c) start to pedal."
Output:
{{
  "proposition": ["adjusting seat", "walk with brakes", "start to pedal"],
  "specification": "EVENTUALLY adjusting seat AND EVENTUALLY walk with brakes AND EVENTUALLY start to pedal"
}}

Example 4: "What happened after the man in black armor spoke and then the pink paper appeared followed by the yellow paper?"
Output:
{{
  "proposition": ["man in black armor spoke", "pink paper appeared", "yellow paper appeared"],
  "specification": "(man in black armor spoke UNTIL (pink paper appeared UNTIL yellow paper appeared))"
}}

Example 5: "A news anchor with curled hair is wearing a pink blazer over a black base and sitting in front of the camera reading the news. What happened before the caption 'standards our climate editor Justin rout' appeared?"
Output:
{{
  "proposition": ["news anchor with curled hair is wearing a pink blazer over a black base", "news anchor sitting in front of the camera reading the news", "subtitle_'standards our climate editor Justin rout'"],
  "specification": "((news anchor with curled hair is wearing a pink blazer over a black base AND news anchor sitting in front of the camera reading the news) UNTIL subtitle_'standards our climate editor Justin rout')"
}}


Example 6: "How did the girl feel before turning on the computer?"
Output:
{{
  "proposition": ["girl turns on computer"],
  "specification": "EVENTUALLY girl turns on computer"
}}

**Now process the following prompt:**
Input:
{{
  "prompt": "{prompt}"
}}

Expected Output (only output the following JSON structure — nothing else):
{{
  "proposition": [...],
  "specification": "..."
}}
"""
    return full_prompt