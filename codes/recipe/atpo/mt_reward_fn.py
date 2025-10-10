import re
import json
import torch
from typing import Dict, List, Union, Any

def parse_turns_from_response_mask(response_mask: torch.Tensor, response_ids: torch.Tensor, tokenizer) -> List[Dict]:
    turns = []
    response_mask_list = response_mask.tolist()
    
    i = 0
    turn_id = 0
    
    while i < len(response_mask_list):
        if response_mask_list[i] == 1:
            start_pos = i
            
            while i < len(response_mask_list) and response_mask_list[i] == 1:
                i += 1
            end_pos = i - 1 
            
            turn_response_ids = response_ids[start_pos:i]
            response_text = tokenizer.decode(turn_response_ids, skip_special_tokens=True) 
            
            is_final_turn = True
            for j in range(i, len(response_mask_list)):
                if response_mask_list[j] == 1:
                    is_final_turn = False
                    break
            
            turns.append({
                'turn_id': turn_id,
                'start_position': start_pos,
                'end_position': end_pos,
                'response': response_text,
                'is_final_turn': is_final_turn,
                'length': end_pos - start_pos + 1
            })
            
            turn_id += 1
        else:
            i += 1
    
    return turns

def calculate_question_reward(details, is_final_turn, verifier_response, human_response, verify_score, effective_score):
    verifier_reward = 0.0
    effective_reward = 0.0
    if is_final_turn:
        details["warning_info"] = "Question asked in final turn"
    else:
        if verifier_response:
            if verifier_response == "<Multiple>":
                details['is_multiple'] = True
            elif verifier_response == "<Repeated>":
                details['is_repeated'] = True
            elif verifier_response == "<ERROR_RESPONSE>":
                details['is_error_response'] = True
            elif verifier_response == "<Timeout>":
                details['is_timeout'] = True
                verifier_reward = verify_score / 2
            elif verifier_response == "<HUMAN_Timeout>":
                details['is_human_timeout'] = True
                verifier_reward = verify_score
            elif verifier_response == "<Normal>":
                verifier_reward = verify_score
                if human_response and human_response != "The patient cannot answer this question.":
                    details["is_effective"] = True
                    effective_reward = effective_score
                else:
                    details["is_reject_human"] = True
            else:
                raise ValueError(f'Unrecognized verifier_response: {verifier_response}')
    return verifier_reward, effective_reward


def calculate_single_turn_reward(model_response: str, correct_answer: str, is_final_turn: bool, turn_id: int, 
                    verifier_response: str = None, human_response: str = None, format_score: float = 0.1, 
                    verify_score: float = 0.4, effective_score: float = 0.5,
                    correctness_score: float = 1.0, **kwargs) -> Dict[str, Union[float, str]]:
    
    format_reward = 0.0
    verifier_reward = 0.0
    effective_reward = 0.0
    correctness_reward = 0.0
    
    details = {
        "has_valid_format": False,  
        "response_type": None,     
        "question_content": None,  
        "final_answer": None,      
        "verifier_result": verifier_response, 
        "is_multiple": False,      
        "is_repeated": False,     
        "is_timeout": False,        
        "is_effective": False,      
        "is_correct": False,        
        "is_reject_human": False,     
        "is_human_timeout": False,      
        "is_error_response": False,     
        "warning_info": None
    }
    
    think_pattern = r'<think>(.*?)</think>(.*?)$'
    match = re.search(think_pattern, model_response, re.DOTALL | re.IGNORECASE)
    
    if match:
        think_content = match.group(1).strip()
        after_think = match.group(2).strip()

        if after_think.startswith("Question:") or after_think.startswith("Final Answer:"):
            details["has_valid_format"] = True
            format_reward += format_score 
            
            if after_think.startswith("Question:"):
                details["response_type"] = "question"
                details["question_content"] = after_think[9:].strip() 
                verifier_reward, effective_reward = calculate_question_reward(details, is_final_turn, verifier_response, human_response, verify_score, effective_score)
            
            elif after_think.startswith("Final Answer:"):
                details["response_type"] = "final_answer"
                details["final_answer"] = after_think[13:].replace('<|im_end|>', '').strip()  
                verifier_reward = verify_score     
                effective_reward = effective_score
                
                if not is_final_turn:
                    raise ValueError(f"Final answer given in non-final turn")
                if correct_answer is not None:
                    model_answer = details["final_answer"].strip().upper()
                    correct_answer_upper = str(correct_answer).strip().upper()

                    if model_answer == correct_answer_upper:
                        details["is_correct"] = True
                        correctness_reward += correctness_score
                    else:
                        details["is_correct"] = False
                        correctness_reward += 0.0
        else:
            details["warning_info"] = "Content after </think> does not start with 'Question:' or 'Final Answer:'"
    else:
        details["warning_info"] = "Response does not contain valid <think></think> format"
    
    process_reward = format_reward + verifier_reward + effective_reward
    
    outcome_reward = correctness_reward
    
    return {
        "process_reward": process_reward,
        "outcome_reward": outcome_reward,
        "format_reward": format_reward,
        "verifier_reward": verifier_reward,
        "effective_reward": effective_reward,
        "correctness_reward": correctness_reward,
        "details": details
    }


def mt_reward_fn(data_source: str, solution_str: str, ground_truth: str, extra_info: Dict = None, **kwargs):
    pass