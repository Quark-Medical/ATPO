# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
LangGraph React Agent Loop.

This implementation is exact same as `ToolAgentLoop`.

Ref: https://langchain-ai.github.io/langgraph/tutorials/workflows/
"""
import re
import json
import random
import asyncio
import numpy as np
from collections import defaultdict
from typing import Any, Literal, List, Dict, Tuple
from langchain_core.outputs import ChatResult
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, convert_to_messages
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from recipe.atpo.chat_model import (
    ChatModel,
    MaxTokenExceededError,
    convert_to_agent_output,
)
from recipe.atpo.agent_loop import AgentLoopBase, AgentLoopOutput
from recipe.atpo.api_request_async import request_vllm_async
from recipe.atpo.tree_search_manager import TreeSearchManager, TreeNode
from recipe.atpo.mt_reward_fn import calculate_single_turn_reward

def extract_answer(text):
    pattern = r'<think>(.*?)</think>(.*?)<\|im_end\|>'
    if text.count('<think>') == 1 and text.count('</think>') == 1:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            answer_content = match.group(2).strip()
            return answer_content
        else:
            return None
    else:
        return None

def parse_assistant_response(assistant_response: str):
    try:
        match = re.search(r'Question:\s*([\s\S]+)', assistant_response)
        if match:
            return match.group(1).strip()
        else:
            return "[ERROR] Your question format is incorrect."
    except:
        return "[ERROR] Your question format is incorrect."


def get_conversation_history(messages: list[BaseMessage]) -> str:
    conversation_history = "{\n"
    for message in messages[1:-1]:
        if message.type == "human":
            conversation_history += f"  \"user\": \"{message.content}\",\n"
        elif message.type == "ai":
            conversation_history += f"  \"assistant\": \"{parse_assistant_response(extract_answer(message.content))}\",\n"
    conversation_history += "}"
    return conversation_history


def render_prompt(assistant_response: str):
    render_template = """<|im_start|>user
{}<|im_end|>
<|im_start|>assistant<think>

</think>

"""
    return render_template.format(assistant_response)

def get_verifier_system_prompt(doctor_question, conversation_history):
    prompt_template = """You are a question analyzer. Your task is to categorize the doctor's current question by examining it against the conversation history.

INSTRUCTIONS:
Respond with EXACTLY ONE of these three tags - nothing else:
    - <Multiple>: Use this tag if the current question contains multiple distinct questions that should be addressed separately
    - <Repeated>: Use this tag if the current question is highly similar to any previous single question already asked by assistant in the conversation history
    - <Normal>: Use this tag if neither of the above conditions apply

Conversation history:
{conversation_history}

Current question to analyze:
{doctor_question}

Your response (one tag only):"""
    return prompt_template.format(conversation_history=conversation_history, doctor_question=doctor_question)

def get_user_system_prompt(atomic_facts, doctor_question):
    prompt_template = """You are a medical information assistant. Your role is to help doctors by providing information strictly from patient data.

INSTRUCTIONS:
1. Search through the provided atomic facts for information that directly answers the doctor's question
2. If you find relevant atomic facts, provide the answer using ONLY that information
3. Do NOT add any medical analysis, inference, interpretation, or external knowledge
4. Do NOT make assumptions or draw conclusions beyond what is explicitly stated
5. If no atomic fact directly answers the question, respond with exactly this phrase: "The patient cannot answer this question."

Patient atomic facts:
{atomic_facts}

Doctor's question:
{doctor_question}

Your response:"""
    return prompt_template.format(atomic_facts=atomic_facts, doctor_question=doctor_question)

async def call_critic_model(trajectory: list[BaseMessage], is_terminal: bool, config: dict) -> float:
    # This is a placeholder for a real critic model.
    # For now, we return a random score to simulate variance.
    if is_terminal: return 0.0
    assert trajectory[-1].type == "human", "Critic last message must be human, but got {}".format(','.join([x.type for x in trajectory]))
    critic_worker = config['configurable']['critic_worker']
    actor_model = config['configurable']['model']
    kwargs = {"is_critic": True}
    try:
        request_id_base, prompt_ids, response_mask = await actor_model.preprocess(trajectory, **kwargs)   
    except MaxTokenExceededError as e:
        print('critic hit MaxTokenExceededError, set critic_value=0.0')
        return 0.0
    response_ids = prompt_ids[-5:]
    prompt_ids = prompt_ids[:-5]
    assert sum(response_ids) == sum([151645, 198, 151644, 77091, 198]), 'critic response_ids {}, except [151645, 198, 151644, 77091, 198]'.format(','.join(response_ids))
    critic_input = {
        'request_id': request_id_base,
        'prompt_ids': prompt_ids,
        'response_ids': response_ids
    }
    value_out = await critic_worker.predict(critic_input)
    value_result = await value_out['result_future'] 
    # print('value_result', value_result)
    return sum(value_result) / len(value_result)

async def call_api(state: dict, config: dict):
    atomic_facts = config["configurable"]["atomic_facts"]
    assistant_response = state["messages"][-1].content
    if isinstance(atomic_facts, list):
        atomic_facts = '\n'.join(atomic_facts)

    parsed_assistant_response = parse_assistant_response(extract_answer(assistant_response))
    if '[ERROR]' in parsed_assistant_response:
        return {"messages": [HumanMessage(content=parsed_assistant_response, 
                                    additional_kwargs={"verifier_response": '<ERROR_RESPONSE>'})]}
    conversation_history = get_conversation_history(state['messages'])

    user_system_prompt = get_user_system_prompt(atomic_facts, parsed_assistant_response)
    rendered_user_system_prompt = render_prompt(user_system_prompt)

    verifier_system_prompt = get_verifier_system_prompt(parsed_assistant_response, conversation_history)
    rendered_verifier_system_prompt = render_prompt(verifier_system_prompt)

    api_response = await request_vllm_async(rendered_user_system_prompt, config['configurable']['apikey'], model='med-qwen3-8b-instruct-test', env='test')
    verifier_response = await request_vllm_async(rendered_verifier_system_prompt, config['configurable']['apikey'], model='med-qwen3-8b-instruct-test', env='test')
    verifier_response = verifier_response.strip() if verifier_response else verifier_response

    if verifier_response:
        if verifier_response in ['<Multiple>', '<Repeated>', '<Normal>']:
            verifier_response = verifier_response
        else:
            verifier_response = '<Timeout>'
            print("[Warning] Unrecognized verifier_response")
    else:
        verifier_response = '<Timeout>'
        print("[Warning] calling verifier failed!")

    if api_response:
        return {"messages": [HumanMessage(
            content=api_response,
            additional_kwargs={"verifier_response": verifier_response}
        )]}
    else:
        print("[Warning] calling user api failed!")
        return {"messages": [HumanMessage(
            content="Sorry, please repeat your question.", 
            additional_kwargs={"verifier_response": '<HUMAN_Timeout>'}
        )]}


def should_continue(state: dict, config: dict) -> Literal["user", "end"]:
    max_assistant_turns = config["configurable"]["max_assistant_turns"]
    num_assistant_turns = 0
    for message in state["messages"]:
        if message.type == "ai":
            num_assistant_turns += 1

    last_message = state["messages"][-1]

    # LLM call failed, e.g: max response length exceeded
    if last_message.type == "human":
        print('Finish loop with last message type is human')
        return "end"

    # max assistant turns exceeded
    if max_assistant_turns and num_assistant_turns >= max_assistant_turns:
        print('Finish loop with max assistant turns exceeded')
        return "end"

    # make a choice
    if 'Final Answer:' in last_message.content:
        print('Finish loop with make a choice')
        return "end"

    return "user"



class TreeSearchRunner:
    """Manages the tree search process with unified termination handling."""
    def __init__(self, initial_messages, params, model, config, reward_coefficients, statistic_metrics):
        self.root = TreeNode(messages=initial_messages)
        self.params = params
        self.model = model
        self.config = config
        self.reward_coefficients = reward_coefficients
        self.statistic_metrics = statistic_metrics
        self.completed_trajectories: List[TreeNode] = []
        self.active_leaves: List[TreeNode] = [self.root]

    async def run(self):
        """Execute the tree search with unified lifecycle management."""
        max_depth = self.config["configurable"]["max_assistant_turns"]
        if self.params["call_critic_enabled"]:
            root_critic_value = await call_critic_model(self.root.messages, self.root.is_terminal, self.config)
            self.root.critic_state_value = root_critic_value
        
        while len(self.completed_trajectories) < self.params["M_trajectories"]:
            if not self.active_leaves:
                break
            
            expansions = await self._expand_leaves_with_assistant()
            
            if not expansions:
                break

            new_leaves_for_evaluation = await self._process_assistant_expansions(expansions)
            
            if not new_leaves_for_evaluation:
                break

            await self._evaluate_nodes(new_leaves_for_evaluation)

            self._prune_and_update_active_leaves(new_leaves_for_evaluation)

        assert len(self.active_leaves) == 0, print('excepted active_leaves is equal 0')
        await self._calculate_advantages_and_returns_for_trajectories(self.completed_trajectories)

        return self.completed_trajectories[:self.params["M_trajectories"]]

    async def _expand_leaves_with_assistant(self) -> Dict[TreeNode, List[BaseMessage]]:
        tasks = {}
        for leaf in self.active_leaves:
            num_candidates = self.params["N_candidates"]
            if (len(self.completed_trajectories) + len(self.active_leaves) * num_candidates) > self.params["M_trajectories"]:
                 num_candidates = 1

            task = self.model.agenerate(
                [leaf.messages], n=num_candidates, sampling_params=self.config["configurable"]["sampling_params"]
            )
            tasks[leaf] = task
    
        results = await asyncio.gather(*tasks.values(), return_exceptions=True) 
        
        expansions = {}
        updated_active_leaves = []

        for leaf, result in zip(tasks.keys(), results):
            if isinstance(result, MaxTokenExceededError):
                leaf.is_terminal = True
                self.completed_trajectories.append(leaf)
            elif isinstance(result, AssertionError):
                print("--- AssertionError ---")
                print(f"AssertionError: {result}")
                import traceback
                print(traceback.print_exc())
            else:
                expansions[leaf] = [gen.message for gen in result.generations[0]]
                updated_active_leaves.append(leaf)

        self.active_leaves = updated_active_leaves
        return expansions

    async def _process_assistant_expansions(self, expansions: Dict[TreeNode, List[BaseMessage]]) -> List[TreeNode]:
        user_response_tasks = {}
        all_new_nodes = []

        for parent, assistant_messages in expansions.items():
            for assistant_msg in assistant_messages:
                # Create a new node representing the state AFTER the assistant's turn
                new_node_after_assistant = TreeNode(
                    messages=parent.messages + [assistant_msg],
                    parent=parent,
                    assistant_message=assistant_msg,
                    level=parent.level + 1
                )
                parent.children.append(new_node_after_assistant)

                temp_state = {"messages": new_node_after_assistant.messages}
                if should_continue(temp_state, self.config) == "end": 
                    new_node_after_assistant.is_terminal = True
                    all_new_nodes.append(new_node_after_assistant)
                else:
                    task = call_api(temp_state, self.config)
                    user_response_tasks[task] = new_node_after_assistant
                    
        if user_response_tasks:
            tasks = list(user_response_tasks.keys())
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for task, result in zip(tasks, results):
                node_to_complete = user_response_tasks[task]
                user_message = result["messages"][0]

                node_to_complete.messages.append(user_message)
                node_to_complete.verifier_response = user_message.additional_kwargs["verifier_response"] 
                all_new_nodes.append(node_to_complete)
        
        return all_new_nodes

    def _prune_and_update_active_leaves(self, evaluated_nodes: List[TreeNode]):
        next_active_leaves = []
        
        # Group new leaves by their parent
        parent_to_children = defaultdict(list)
        for node in evaluated_nodes:
            if node.parent:
                parent_to_children[node.parent].append(node)

        for parent, children in parent_to_children.items():
            # Pruning logic remains the same
            if not self.params["pruning_enabled"] or len(children) <= 1:
                survivors = children
            else:
                q_values = [child.q_value for child in children]
                variance = np.var(q_values)
                parent.q_value_variance = variance
                nor_variance = variance # U2
                if f'q_value_variance_l{parent.level}_mean' in self.statistic_metrics:
                    nor_variance = (variance - self.statistic_metrics[f'q_value_variance_l{parent.level}_mean']) / (
                                self.statistic_metrics[f'q_value_variance_l{parent.level}_std'] + 1e-7)
                elif 'q_value_variance_mean' in self.statistic_metrics:
                    nor_variance = (variance - self.statistic_metrics['q_value_variance_mean']) / (self.statistic_metrics['q_value_variance_std'] + 1e-7)
                diff_value = parent.critic_state_value - np.mean(q_values)   # U1
                if 0.3*nor_variance + 0.7*abs(diff_value) < self.params["variance_threshold"] and random.random() < 0.9:
                    select_child = random.sample(children, 1)[0]
                    survivors = [select_child]
                else:
                    survivors = children

            for child in survivors:
                if child.is_terminal:
                    self.completed_trajectories.append(child)
                else:
                    next_active_leaves.append(child)

        self.active_leaves = next_active_leaves

    async def _evaluate_nodes(self, nodes: List[TreeNode]):
        if self.params["call_critic_enabled"]:
            value_tasks = [call_critic_model(node.messages, node.is_terminal, self.config) for node in nodes]
            values = await asyncio.gather(*value_tasks)
            
        for i, node in enumerate(nodes):
            reward_info = self._calculate_turn_reward(node)

            if node.is_terminal:
                if reward_info['details']['response_type'] == 'final_answer':
                    node.action_reward = reward_info["outcome_reward"]     
                else:
                    node.action_reward = self.reward_coefficients["exceed_max_turn_penalty"]
                node.critic_state_value = 0.0 if self.params["call_critic_enabled"] else None
            else:
                # node.action_reward = reward_info["process_reward"]
                node.action_reward = 0.0
                node.critic_state_value = values[i] if self.params["call_critic_enabled"] else None

            gamma = self.params.get("gamma", 1.0)
            node.q_value = (node.action_reward + gamma * node.critic_state_value) if self.params["call_critic_enabled"] else None
            
    def _calculate_turn_reward(self, node: TreeNode) -> Dict:
        # ... (implementation from before)
        if not node.parent:
            print(node)
            raise ValueError("No parent node found for the current node.")

        model_response = node.assistant_message.content
        is_final_turn = node.is_terminal
        turn_id = sum(1 for msg in node.parent.messages if msg.type == 'ai') 
        human_response = node.messages[-1].content if len(node.messages) > len(node.parent.messages) + 1 and node.messages[-1].type == 'human' else None
        ground_truth = self.config["configurable"]["ground_truth"]

        reward_info = calculate_single_turn_reward(
            model_response=model_response,
            correct_answer=ground_truth.get('answer'),
            is_final_turn=is_final_turn,
            turn_id=turn_id,  
            verifier_response=node.verifier_response,
            human_response=human_response,
            **self.reward_coefficients
        )
        return reward_info

    async def _calculate_advantages_and_returns_for_trajectories(self, trajectories: List[TreeNode]):
        if not trajectories:
            return

        all_nodes_in_tree = set()
        nodes_to_visit = [self.root]
        while nodes_to_visit:
            current_node = nodes_to_visit.pop()
            if current_node in all_nodes_in_tree:
                continue
            all_nodes_in_tree.add(current_node)
            nodes_to_visit.extend(current_node.children)
        if self.params["call_critic_enabled"]:
            await self._batch_fill_critic_values(all_nodes_in_tree)

        for leaf_node in trajectories:
            path = []
            curr = leaf_node
            while curr:
                curr.explore_num += 1
                path.append(curr.explore_num)
                if curr.parent:
                    curr = curr.parent
                else:
                    break
        # print('last_node_path_explore num', path)

        self._perform_tree_backup(all_nodes_in_tree)
        
        self._calculate_advantage_and_returns_for_all_nodes(all_nodes_in_tree)

    async def _batch_fill_critic_values(self, nodes: set):
        """Fills the initial `state_value` for any node where it's missing, using a critic model."""
        nodes_to_evaluate = [node for node in nodes if node.critic_state_value is None]

        if not nodes_to_evaluate:
            return
        value_tasks = [call_critic_model(node.messages, node.is_terminal, self.config) for node in nodes_to_evaluate]
        values = await asyncio.gather(*value_tasks)
        for node, value in zip(nodes_to_evaluate, values):
            node.critic_state_value = value

    def _perform_tree_backup(self, all_nodes: set):
        node_depths = {self.root: 0}
        nodes_to_visit = [self.root]
        head = 0
        while head < len(nodes_to_visit):
            parent = nodes_to_visit[head]
            head += 1
            for child in parent.children:
                if child not in node_depths:
                     node_depths[child] = node_depths[parent] + 1
                     nodes_to_visit.append(child)

        sorted_nodes = sorted(list(all_nodes), key=lambda n: node_depths.get(n, -1), reverse=True)
        gamma = self.params.get("gamma", 1.0)

        for node in sorted_nodes:
            if not node.children:
                if node.mdp_state_value is None:
                    node.mdp_state_value = 0.0 if node.is_terminal else node.mdp_state_value
                if self.params["call_critic_enabled"]:
                    assert node.critic_state_value is not None, 'node.critic_state_value should not be None'
            else:
                child_q_values = []
                for child in node.children:
                    if child.mdp_state_value is None:
                        assert child.explore_num == 0, "Child mdp_state_value is None must be pruned by value variance."
                        continue
                    # assert child.mdp_state_value is not None, "Child's mdp_state_value should have been computed."
                    assert child.action_reward is not None, "Child's action_reward cannot be None."

                    q_value_for_backup = child.action_reward + gamma * child.mdp_state_value
                    child_q_values.append(q_value_for_backup)

                assert len(child_q_values) > 0, "Child must be bigger than 0, children num {}".format(len(node.children))
                if child_q_values:
                    node.mdp_state_value = np.mean(child_q_values)

    def _calculate_advantage_and_returns_for_all_nodes(self, all_nodes: set):
        gamma = self.params.get("gamma", 1.0)

        for node in all_nodes:
            node.returns = node.mdp_state_value
            if node.parent is None:
                continue

            parent = node.parent

            is_validate = self.params["M_trajectories"] == 1 and self.params["N_candidates"] == 1
            if is_validate:
                v_s = parent.mdp_state_value
                v_s_prime = node.mdp_state_value
            elif self.params['only_use_critic_value']:
                v_s = parent.critic_state_value
                v_s_prime = node.critic_state_value
            elif len(parent.children) == 1:
                v_s = parent.critic_state_value
                v_s_prime = node.critic_state_value
            else:
                v_s = parent.mdp_state_value
                v_s_prime = node.mdp_state_value
            advantage = node.action_reward + gamma * v_s_prime - v_s

            node.advantage = advantage / max(node.explore_num, 1)

class UserAssistantAgentLoop(AgentLoopBase):
    @classmethod
    def init_class(cls, config, tokenizer, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        cls.reward_coefficients = config.get("reward_coefficients", {
            'format_score': 0.1, 'verify_score': 0.4, 'effective_score': 0.5,
            'correctness_score': 1.0, 'exceed_max_turn_penalty': -1.0
        })
        print("Performing class-level UserAssistantAgentLoop initialization")

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> list[AgentLoopOutput]:
        initial_messages = list(kwargs['extra_info']["prompt"])
        atomic_facts = kwargs["extra_info"]["atomic_facts"]
        ground_truth = kwargs["reward_model"]["ground_truth"]
        critic_worker = kwargs["critic_worker"]

        rollout = self.config.actor_rollout_ref.rollout
        model_type = self.config.actor_rollout_ref.model_type

        tree_search_params = kwargs["tree_search_params"]
        search_params = {
            "M_trajectories": tree_search_params.get("M_trajectories", 32),
            "N_candidates": tree_search_params.get("N_candidates", 2),
            "variance_threshold": tree_search_params.get("variance_threshold", 0.1),
            "pruning_enabled": tree_search_params.get("pruning_enabled", False),
            "call_critic_enabled": tree_search_params.get("call_critic_enabled", False),
            "only_use_critic_value": tree_search_params.get("only_use_critic_value", False),
            "gamma": self.config.algorithm.gamma,
            "lam": self.config.algorithm.lam 
        }
        statistic_metrics = kwargs["statistic_metrics"]

        model = ChatModel(
            model=model_type,
            client=self.server_manager,
            tokenizer=self.tokenizer,
            max_tokens=rollout.response_length,
        )

        config = {
            "configurable": {
                "model": model,
                "critic_worker": critic_worker,
                "sampling_params": sampling_params,
                "max_assistant_turns": rollout.multi_turn.max_assistant_turns,
                "apikey": kwargs["api_key"],
                "atomic_facts": atomic_facts,
                "ground_truth": ground_truth
            }
        }
        
        initial_messages = convert_to_messages(initial_messages)
        
        runner = TreeSearchRunner(initial_messages, search_params, model, config, self.reward_coefficients, statistic_metrics)
        final_trajectories = await runner.run()

        print(f"\n--- Search Complete ---")
        print(f"Generated {len(final_trajectories)} final trajectories.")

        outputs = [
            convert_to_agent_output(node, rollout.response_length) for node in final_trajectories
        ]

        return outputs
