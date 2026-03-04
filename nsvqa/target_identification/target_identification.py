from nsvqa.puls.llm import LLM
import json
import os 

TARGET_GROUNDING_SYSTEM = """
You are a Video Temporal Grounding Agent. Your task is to expand an identified temporal window to capture the visual evidence required to answer a specific question.

### BUFFER RULES:
  - "after": Use [start_time, end_time + 5]
  - "before": Use [start_time - 5, end_time]
  - "during": Use [start_time - 3, end_time + 3]
  * Note: All target windows MUST fully contain the original [start_time, end_time].

### QUESTION FORMATTING:
Input:
Question: "q" 
Specification: "p"
Identified temporal window for the specification: [start_time, end_time]
Candidates: Candidates: {c}

### EXAMPLE
Input:
Question: "What happened after the man sat down?"
Specification: (man sits down)
Identified temporal window for the specification: [start_time, end_time]
Candidates: 
"- He picked up a book"
"- He fell asleep"

Output:
{
  "target_frame_window": "[start_time, end_time + 5]",
  "explanation": "Since the question asks for events 'after' the specification, I included the original window and added a 5s buffer to capture the subsequent action."
}

### OUTPUT:
Return ONLY a valid JSON object. Follow this pattern:
-> {"target_frame_window": "[s_new, e_new]", "explanation": "..."}
"""

def get_target_window_prompt(question, specification, candidates):
    user_query =f"""
Input:
Question: "{question}" |
Specification: "{specification}" |
Window: [start_time, end_time] |
Candidate:
{chr(10).join(f"- {candidate}" for candidate in candidates)}

Output:
"""

    # 3. Structure the messages for the API
    messages = [
        {"role": "system", "content": TARGET_GROUNDING_SYSTEM},
        # You can insert few-shot examples here as user/assistant pairs if needed
        {"role": "user", "content": user_query}
    ]

    return messages

def clean_and_parse_json(raw_str):
    start = raw_str.find("{")
    end = raw_str.rfind("}") + 1
    json_str = raw_str[start:end]
    return json.loads(json_str)


def process_datapoint(llm, question, candidates, specification, video_id):
    messages = get_target_window_prompt(question, specification, candidates)
    llm.history.extend(messages)
    # Get LLM response
    response = llm.client.chat.completions.create(
        model=llm.model,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0
    )
    
    raw_content = response.choices[0].message.content
    parsed = json.loads(raw_content)
    llm.history.append({"role": "assistant", "content": raw_content})

    target_frame_window = parsed["target_frame_window"]
    explanation = parsed["explanation"]

    # Save the conversation history with timestamp
    history_path = llm.save_history(video_id, suffix="target")

    return {
        "frame_window": target_frame_window,
        "explanation": explanation,
        "saved_path": history_path,
    }


def identify_target(question, candidates, specification, conversation_history, video_id, save_dir):
    # Read the conversation history
    history_path = conversation_history
    with open(history_path, "r") as f:
        history = json.load(f)

    save_dir = os.path.join(save_dir, video_id)
    os.makedirs(save_dir, exist_ok=True)
    llm = LLM(history=history, save_dir=save_dir)

    # Get target identification results
    result = process_datapoint(llm, question, candidates, specification, video_id)
    return result

