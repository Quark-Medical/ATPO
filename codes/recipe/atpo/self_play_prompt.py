def self_play_prompt(json_data):
    prompt = """
# Instruction: Generate a medical diagnostic conversation JSON
I will provide a medical diagnosis case as JSON input, which includes the patient's initial information "initial_info". In the initial information, there is the patient's initial statement, the diagnostic question "question", four optional choices "options", and the correct option "answer_idx", as well as the patient's objective factual information "atomic_facts".
Your task is to strictly follow the rules below to simulate a multi-turn conversation between a doctor and a patient. The purpose of the conversation is for the doctor to collect enough information through questioning, ultimately selecting the most likely diagnosis from the options, and to choose correctly as much as possible. Your final output must be a single, completely correct JSON object.

## 1. Role Definition
Doctor (role: "assistant"): A professional AI doctor. Your task is to logically ask questions to collect information, perform differential diagnosis, and ultimately make the best possible decision.
Patient (role: "user"): An AI patient. Your knowledge is strictly limited to the provided "atomic_facts".

## 2. Core Rules (Must Follow Strictly)
### Doctor ("assistant") Behavior Guidelines:
2.1 Execution Steps:
1) Internal thinking: At the start of each turn, internally evaluate—based only on the existing simulated conversation history and the question information (including "question", "options")—whether you are confident enough to make the correct choice (Note: It is strictly forbidden to use any other unmentioned information such as "atomic_facts" or "answer_idx"). If the information is insufficient, prepare to simulate asking the patient a question; if sufficient, make the final choice directly.
2) If asking: First internal thinking—determine what crucial information is missing to make the most accurate diagnosis (based on the simulated conversation history and question info), and design ONE question to ask in the "content" field. In the "think" field, explain what key information is missing, why this question can help narrow down the diagnosis, and why asking this question first makes sense.
3) If choosing: In the "think" field, explain how you arrived at your final conclusion using only information revealed during the conversation; put the final choice in the "content" field.

2.2 Notes:
1) Format requirements:
- A questioning turn's "content" must be in the format: "Question: [question text]", with only the question itself; "think" field briefly describes the thought process and reason for asking.
- A choosing turn's "content" must be in the format: "Final Answer: [option letter]", single choice only; "think" field briefly describes thought process and reason for choice.
2) Basis for reasoning: The basis in the "think" field must rely only on information already revealed in the conversation and question; it is forbidden to use any "atomic_facts" or "answer_idx" not yet mentioned in the conversation. This is the most important rule.
3) Person reference: Must refer to the patient in third person during interaction (e.g., "Does the patient have any weight changes?").
4) Question count: At most ONE question per turn.

### Patient ("user") Behavior Guidelines:
2.3 Execution Steps:
1) Start: The first turn of the conversation must use the "initial_info" content provided in the input.
2) Reply: In subsequent turns, respond concisely to each doctor question based on the "atomic_facts". Do not answer unrelated questions not in "atomic_facts".
2.4 Notes:
1) Knowledge boundary (critical): All of the patient's answers must come solely from the "atomic_facts" list. This is an absolute restriction.
2) Missing info: If the doctor's question is beyond the scope of "atomic_facts", the patient must answer "The patient cannot answer this question". Do not fabricate or infer information.
3) Person reference: Must refer to the patient in third person during interaction (e.g., "Yes, the patient has lost weight.").

## 3. Output Format Requirements
3.1 Your only output must be a complete JSON object in English.
3.2 The JSON object must contain a "message" key, whose value is an array of conversation turns.
3.3 Total conversation turns (user + assistant) should be adjusted according to the difficulty level of the diagnostic question, not fewer than 3 turns and not more than 8 turns.
3.4 Output structure example (note: the turn count here is only reference):
{
  "message": [
    {"role": "user", "content": "[Patient's opening statement]"},
    {"role": "assistant", "think": "[Reasoning process before asking question]", "content": "Question: [Doctor's question]"},
    {"role": "user", "content": "[Patient's answer based on 'atomic_facts']"},
    {"role": "assistant", "think": "[Full reasoning and explanation before making final answer]", "content": "Final Answer: [option letter]"}
  ]
}

## 4. Reference Example
### Example 1:
Input:
{
  "initial_info":"A 59-year-old woman came to see a doctor because of increasing weakness and fatigue over the past 3 months.\n question: Which of the following is the most likely primary mechanism underlying this woman's temporal muscle atrophy? \n option:{\"A\": \"Cytochrome c-mediated activation of proteases\", \"B\": \"Lipase-mediated degradation of triglycerides\", \"C\": \"TNF-α-mediated activation of caspases\", \"D\": \"Proteasomal degradation of ubiquitinated proteins\"} ",
  "answer_idx": "D",
  "atomic_facts":["1. She has had a 10.5-kg weight loss during this period.", "2. The weight loss is equivalent to 23 lbs.", "3. Her BMI is 16 kg/m2.", "4. She appears thin.", "5. She has bilateral temporal wasting."]
}

Output:
{
  "message": [
    {
      "role": "user",
      "content": "A 59-year-old woman came to see a doctor because of increasing weakness and fatigue over the past 3 months.\n Question: Which of the following is the most likely primary mechanism underlying this woman's temporal muscle atrophy? \n Option:{\"A\": \"Cytochrome c-mediated activation of proteases\", \"B\": \"Lipase-mediated degradation of triglycerides\", \"C\": \"TNF-α-mediated activation of caspases\", \"D\": \"Proteasomal degradation of ubiquitinated proteins\"}"
    },
    {
      "role": "assistant",
      "think": "The patient complained of weakness, a nonspecific symptom. To understand the underlying mechanism, I needed to learn more about her medical condition, as weight loss is a key indicator of a wasting disease.",
      "content": "Question: Hello, in addition to weakness, has the patient had any recent weight changes?"
    },
    {
      "role": "user",
      "content": "Yes, the patient lost a lot of weight, about 10.5 kg."
    },
    {
      "role": "assistant",
      "think": "The patient confirmed significant weight loss. Questions pointed to 'temporal muscle atrophy,' a visual sign. I needed to confirm this sign through questioning, as it's directly related to muscle wasting.",
      "content": "Question: Do you notice any difference in the patient's appearance, such as body shape or face?"
    },
    {
      "role": "user",
      "content": "The patient looked very thin, and her family members thought she looked unrecognizable, especially her temples, which were sunken."
    },
    {
      "role": "assistant",
      "think": "The patient's description of 'sunken temples' is precisely 'bilateral temporal wasting', which, combined with her significant weight loss (10.5 kg) and emaciated appearance (BMI 16), strongly suggests muscle wasting caused by cancer cachexia. Its core pathological mechanism is the degradation of muscle proteins via the ubiquitin-proteasome pathway. Therefore, the most likely mechanism is 'proteasomal degradation of ubiquitinated proteins.'",
      "content": "Final Answer: D"
    }
  ]
}

# Please start simulating the conversation based on the input below (note: the number of turns in the example is for reference only; the actual simulated conversation turns should be adjusted according to the diagnostic difficulty, not fewer than 3 turns and not more than 8 turns.)
Input:
{json_data}
"""

    return prompt
