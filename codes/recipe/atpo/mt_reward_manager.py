import torch
import numpy as np
from collections import defaultdict
from typing import Dict, Any, Optional, Callable, Tuple, List
from verl import DataProto
from verl.workers.reward_manager.abstract import AbstractRewardManager
from recipe.atpo.mt_reward_fn import parse_turns_from_response_mask, calculate_single_turn_reward

class MultiTurnRewardManager(AbstractRewardManager):
    def __init__(
        self, 
        tokenizer,
        num_examine: int,
        compute_score: Optional[Callable] = None,
        reward_fn_key: str = "data_source",
        reward_coefficients: Optional[Dict] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine 
        self.compute_score = compute_score or None
        self.reward_fn_key = reward_fn_key
        self.reward_params = reward_coefficients or {
            'format_score': 0.1,
            'verify_score': 0.4,
            'effective_score': 0.5,
            'correctness_score': 1.0,
            'exceed_max_turn_penalty': -1.0
        }

    def __call__(self, data: DataProto, return_dict: bool = False) -> Dict[str, torch.Tensor]:
        
        process_reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        outcome_reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        for i in range(len(data)):
            data_item = data[i]

            response_ids = data_item.batch["responses"]
            response_mask = data_item.batch["response_mask"] 
            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            verifier_responses = data_item.non_tensor_batch['verifier_responses']
            num_turn = data_item.non_tensor_batch['__num_turns__']
            turns_info = parse_turns_from_response_mask(response_mask, response_ids, self.tokenizer)

            expected_verifier_count = len(turns_info) - 1 if len(turns_info) > 0 else 0
            assert len(turns_info) == num_turn, f"Turn Count Mismatch: expected {num_turn}, got {len(turns_info)}"
            assert len(verifier_responses) == expected_verifier_count, f"Verifier Count Mismatch: expected {expected_verifier_count}, got {len(verifier_responses)}"
            
            if not turns_info:
                continue

            human_responses = self._extract_human_responses(response_ids, response_mask, self.tokenizer)

            process_rewards = []
            turn_end_positions = []
            details_list = []
            outcome_reward = 0

            for turn_idx, turn_info in enumerate(turns_info):
                model_response = turn_info['response']
                is_final_turn = turn_info['is_final_turn']

                verifier_response = None
                human_response = None
                if turn_idx < len(verifier_responses):
                    verifier_response = verifier_responses[turn_idx]
                if turn_idx < len(human_responses):
                    human_response = human_responses[turn_idx]

                correct_answer = None
                if is_final_turn:
                    if isinstance(ground_truth, dict) and 'answer' in ground_truth:
                        correct_answer = ground_truth['answer']
                    else:
                        correct_answer = str(ground_truth)
                    
                turn_reward = calculate_single_turn_reward(
                    model_response=model_response,
                    correct_answer=correct_answer,
                    is_final_turn=is_final_turn,
                    turn_id=turn_idx,
                    verifier_response=verifier_response,
                    human_response=human_response,
                    **self.reward_params
                )

                end_position = turn_info['end_position']
                turn_end_positions.append(end_position)
                process_reward = turn_reward['process_reward']
                process_reward_tensor[i, end_position] = float(process_reward)
                process_rewards.append(process_reward)
                details_list.append(turn_reward['details'])
                
                if is_final_turn:
                    if turn_reward['details']['response_type'] == 'final_answer':
                        outcome_reward = turn_reward['outcome_reward']
                    else:
                        outcome_reward = self.reward_params['exceed_max_turn_penalty']
                    outcome_reward_tensor[i, end_position] = float(outcome_reward)

            reward_extra_info['process_rewards_sum'].append(sum(process_rewards))
            reward_extra_info['turn_end_positions'].append(turn_end_positions)
            reward_extra_info['turn_process_rewards'].append(process_rewards)
            process_rewards_mean = sum(process_rewards) / (len(process_rewards) + 1e-7)
            reward_extra_info['process_rewards_mean'].append(process_rewards_mean)
            reward_extra_info['outcome_reward'].append(outcome_reward)

            self.update_metrics(details_list, reward_extra_info, len(turns_info))

        reward_extra_info['adv_compute_info'] = []
        need_keys = ['turn_end_positions', 'turn_process_rewards', 'outcome_reward']
        for i in range(len(outcome_reward_tensor)):
            ret = {}
            for key in need_keys:
                ret[key] = reward_extra_info[key][i]
            reward_extra_info['adv_compute_info'].append(ret)
        reward_extra_info.pop('turn_end_positions')
        reward_extra_info.pop('turn_process_rewards')

        if return_dict:
            return {
                "reward_tensor": outcome_reward_tensor,
                "reward_extra_info": reward_extra_info
            }
        else:
            return outcome_reward_tensor

    def update_metrics(self, details_list, reward_extra_info, turn_num):
        num_incorrect_format = 0
        num_multiple = 0
        num_repeated = 0
        num_verify_timeout = 0
        num_effective = 0
        num_reject_human = 0
        num_human_timeout = 0
        num_error_response = 0
        for details in details_list[:-1]:
            if details['has_valid_format'] == False:
                num_incorrect_format += 1
            if details['is_multiple'] == True:
                num_multiple += 1
            if details['is_repeated'] == True:
                num_repeated += 1
            if details['is_timeout'] == True:
                num_verify_timeout += 1
            if details['is_effective'] == True:
                num_effective += 1
            if details['is_reject_human'] == True:
                num_reject_human += 1
            if details['is_human_timeout'] == True:
                num_human_timeout += 1
            if details['is_error_response'] == True:
                num_error_response += 1
        assert num_incorrect_format + num_multiple + num_repeated + num_verify_timeout + num_effective + num_reject_human + num_human_timeout + num_error_response == turn_num - 1, "metric length sum mismatch, {}, {}, {}, {}, {}, {}, {}, {}, turn_num={}".format(
            num_incorrect_format, num_multiple, num_repeated, num_verify_timeout,
                num_effective, num_reject_human, num_human_timeout, num_error_response, turn_num)

        reward_extra_info['turn_count'].append(turn_num)
        reward_extra_info['incorrect_format_rate'].append(
            num_incorrect_format / (turn_num - 1) if turn_num > 1 else 0.0
        )
        reward_extra_info['multiple_rate'].append(
            num_multiple / (turn_num - 1) if turn_num > 1 else 0.0
        )
        reward_extra_info['repeated_rate'].append(
            num_repeated / (turn_num - 1) if turn_num > 1 else 0.0
        )
        reward_extra_info['verify_timeout_rate'].append(
            num_verify_timeout / (turn_num - 1) if turn_num > 1 else 0.0
        )
        reward_extra_info['effective_rate'].append(
            num_effective / (turn_num - 1) if turn_num > 1 else 0.0
        )
        reward_extra_info['human_reject_rate'].append(
            num_reject_human / (turn_num - 1) if turn_num > 1 else 0.0
        )
        reward_extra_info['human_timeout_rate'].append(
            num_human_timeout / (turn_num - 1) if turn_num > 1 else 0.0
        )
        reward_extra_info['assistant_error_rate'].append(
            num_error_response / (turn_num - 1) if turn_num > 1 else 0.0
        )
        reward_extra_info['is_valid'].append(1 if details_list[-1]['response_type'] == 'final_answer' else 0)
        reward_extra_info['is_correct'].append(1 if details_list[-1]['is_correct'] else 0)


    def _extract_human_responses(self, response_ids: torch.Tensor, response_mask: torch.Tensor, tokenizer) -> List[str]:
        human_responses = []
        response_mask_list = response_mask.tolist()
        
        i = 0
        while i < len(response_mask_list):
            if response_mask_list[i] == 0:
                start_pos = i
                while i < len(response_mask_list) and response_mask_list[i] == 0:
                    i += 1
                end_pos = i          
                if end_pos != len(response_mask_list): 
                    human_response_ids = response_ids[start_pos:end_pos]
                    human_text = tokenizer.decode(human_response_ids, skip_special_tokens=True)
                    human_text = human_text.replace('user\n', '').replace('\nassistant', '')
                    human_responses.append(human_text.strip())
            else:
                i += 1
                
        return human_responses