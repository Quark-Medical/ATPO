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
Ref: https://python.langchain.com/docs/how_to/custom_chat_model/
"""

import asyncio
import logging
import os
import uuid
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    convert_to_openai_messages,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from pydantic import Field

from recipe.atpo.agent_loop import AgentLoopOutput, AsyncLLMServerManager
from recipe.atpo.tree_search_manager import TreeNode
from copy import deepcopy


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

async def decode_assistant_response(responses_ids: list[int], tokenizer) -> str:
    loop = asyncio.get_running_loop()
    content = await loop.run_in_executor(None, tokenizer.decode, responses_ids)

    return content

class MaxTokenExceededError(Exception):
    """Indicate that history chat messages + human message exceeds LLM max_tokens."""

    pass


class ChatModel(BaseChatModel):
    model_name: str = Field(alias="model")
    """The name of the model"""

    client: AsyncLLMServerManager
    """AsyncLLM server manager"""

    tokenizer: Any
    """Tokenizer for the model"""

    max_tokens: int
    """Max tokens to generate"""

    temperature: float = 1.0
    """Temperature for sampling"""

    top_p: float = 1.0
    """Top p for sampling"""

    repetition_penalty: float = 1.0
    """Repetition penalty for sampling"""


    def with_structured_output(
        self,
        schema: dict | type,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, dict | BaseChatModel]:
        """Ref: https://langchain-ai.github.io/langgraph/how-tos/react-agent-structured-output/"""
        raise NotImplementedError

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise NotImplementedError

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Asynchronously generate chat completion message(s).

        Args:
            messages (list[BaseMessage]): List of messages.
            stop (Optional[list[str]], optional): Stop words. Defaults to None.
            n (int, optional): Number of candidate generations. Defaults to 1.

        Returns:
            ChatResult: Chat result containing one or more generations.
        """
        # Get the number of generations to produce
        n = kwargs.get("n", 1)

        request_id_base, prompt_ids, response_mask = await self.preprocess(messages, **kwargs)

        sampling_params = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
        }
        if "sampling_params" in kwargs:
            sampling_params.update(kwargs["sampling_params"])

        # Create N parallel generation requests
        tasks = []
        for i in range(n):
            request_id = f"{request_id_base}-{i}"
            tasks.append(
                self.client.generate(
                    request_id=request_id, prompt_ids=prompt_ids, sampling_params=sampling_params
                )
            )

        outputs = await asyncio.gather(*tasks)

        # Postprocess all N results
        postprocess_tasks = []
        for i, output in enumerate(outputs):
            request_id = f"{request_id_base}-{i}"
            postprocess_tasks.append(
                self._postprocess(request_id, prompt_ids, response_mask, output.token_ids, **kwargs)
            )

        processed_messages = await asyncio.gather(*postprocess_tasks) # [AIMessage, AIMessage, AIMessage, ...]

        generations = [ChatGeneration(message=msg) for msg in processed_messages]
        return ChatResult(generations=generations)

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return self.model_name

    async def preprocess(self, messages: list[BaseMessage], **kwargs: Any) -> tuple[str, list[int], list[int]]:

        # messages: [system], human, ai, human, ai, human, ai
        assert messages[-1].type == "human", (f"Last message must be human, but got {messages[-1].type}")
        loop = asyncio.get_running_loop()

        # Case 1: initial chat completion: [system], human
        if messages[-1].type == "human" and len(messages) == 2:
            prompt_ids = await loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.apply_chat_template(
                        convert_to_openai_messages(messages),
                        add_generation_prompt=True,
                        tokenize=True,
                    ),
                )
            return str(uuid.uuid4()), prompt_ids, []

        # Case 2: follow up chat completion with human response: [system], human, ai, human, ...
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].type == "ai":
                break
        if "prompt_ids" not in messages[i].response_metadata:
            print('messages[i].response_metadata', messages[i].response_metadata)
            print('messages types', [x.type for x in messages], i)
        assert "prompt_ids" in messages[i].response_metadata, "Last message must have prompt_ids in response_metadata"
        assert "response_mask" in messages[i].response_metadata, ("Last message must have response_mask in response_metadata")

        # encode human response
        human_responses = convert_to_openai_messages(messages[i + 1 :])
        human_response_ids = await loop.run_in_executor(
            None,
            lambda messages=human_responses: self.tokenizer.encode('\n') + self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True
            ),
        )

        # stop generation if response length exceeds max response length
        if len(messages[i].response_metadata["response_mask"]) + len(human_response_ids) >= self.max_tokens:
            raise MaxTokenExceededError(f"Max response length {self.max_tokens} exceeded")

        # append human response to prompt
        if kwargs.get('is_critic', False):
            request_id = messages[i].response_metadata["request_id"]
            prompt_ids = deepcopy(messages[i].response_metadata["prompt_ids"])
            response_mask = deepcopy(messages[i].response_metadata["response_mask"])
        else:
            request_id = messages[i].response_metadata.pop("request_id")
            prompt_ids = messages[i].response_metadata.pop("prompt_ids")
            response_mask = messages[i].response_metadata.pop("response_mask")
        prompt_ids += human_response_ids
        response_mask += [0] * len(human_response_ids)

        return request_id, prompt_ids, response_mask

    async def _postprocess(
        self, request_id: str, prompt_ids: list[int], response_mask: list[int], response_ids: list[int], **kwargs: Any
    ) -> AIMessage:
        new_prompt_ids = prompt_ids + response_ids
        new_response_mask = response_mask + [1] * len(response_ids)
        content = await decode_assistant_response(response_ids, self.tokenizer)

        message = AIMessage(
            content=content,
            response_metadata={
                "request_id": request_id,
                "prompt_ids": new_prompt_ids,
                "response_mask": new_response_mask,
            },
        )
        return message

def convert_to_agent_output(node: 'TreeNode', response_length: int) -> AgentLoopOutput:
    """Convert a completed TreeNode to AgentLoopOutput, including token-level advantages.

    Args:
        node (TreeNode): The final leaf node of a trajectory, containing calculated advantages.
        response_length (int): Max length of response.

    Returns:
        AgentLoopOutput: agent loop output trajectory used for training.
    """
    messages = node.messages
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].type != "human":
            break
    last_message = messages[i]
    assert last_message.type == "ai", f"Last message must be assistant, but got {last_message.type}"
    assert "prompt_ids" in last_message.response_metadata, "Last message must have prompt_ids in response_metadata"
    assert "response_mask" in last_message.response_metadata, (
        "Last message must have response_mask in response_metadata"
    )

    prompt_ids = last_message.response_metadata["prompt_ids"]
    response_mask = last_message.response_metadata["response_mask"]

    token_advantages = []

    path = []
    curr = node
    q_value_variance_list = []  # level, q_var
    mdp_value_list = []
    critic_value_list = []
    while curr:
        path.append(curr)
        if curr.q_value_variance is not None:
            q_value_variance_list.append([curr.uid, curr.level, curr.q_value_variance])
        if curr.mdp_state_value is not None:
            mdp_value_list.append([curr.uid, curr.level, curr.mdp_state_value])
        if curr.critic_state_value is not None:
            critic_value_list.append([curr.uid, curr.level, curr.critic_state_value])
        if curr.parent:
            curr = curr.parent
        else:
            break
    path.reverse()

    advantages_per_turn = [n.advantage for n in path[1:]]   
    returns_per_turn = [n.returns for n in path]            
    explore_num_per_turn = [n.explore_num for n in path]
    token_advantages = []
    token_returns = [0.0] * len(response_mask)
    value_response_mask = [0] * len(response_mask)
    turn_idx = 0


    i = 0
    while i < len(response_mask):
        if response_mask[i] == 1:
            advantage_for_this_turn = advantages_per_turn[turn_idx]
            ret_for_turn = returns_per_turn[turn_idx]
            explore_for_this_turn = explore_num_per_turn[turn_idx]
            turn_idx += 1

            j = i
            while j < len(response_mask) and response_mask[j] == 1:
                j += 1

            num_tokens_in_turn = j - i
            token_advantages.extend([advantage_for_this_turn] * num_tokens_in_turn)
            for t in range(i, i+5):
                if t >= len(token_returns):
                    continue
                token_returns[t] = ret_for_turn
                value_response_mask[t] = 1

            i = j
        else:
            token_advantages.append(0.0)
            i += 1
    assert len(token_advantages) == len(response_mask), "token_advantages and response_mask should have the same length"
    num_turns = 0
    verifier_responses = []
    for i in range(len(messages)):
        if messages[i].type == "ai":
            num_turns += 1
        elif messages[i].type == "human" and i > 2 and i != len(messages)-1:
            verifier_resp = messages[i].additional_kwargs.get("verifier_response", None)
            verifier_responses.append(verifier_resp)

    response_ids = prompt_ids[-len(response_mask) :]
    prompt_ids = prompt_ids[: len(prompt_ids) - len(response_mask)]

    statistics = {
        "q_value_variance_list" : q_value_variance_list,
        "mdp_value_list" : mdp_value_list,
        "critic_value_list" : critic_value_list
    }

    output = AgentLoopOutput(
        prompt_ids=prompt_ids,
        response_ids=response_ids[:response_length],
        response_mask=response_mask[:response_length],
        advantages=token_advantages[:response_length],
        returns=token_returns[:response_length],
        value_response_mask=value_response_mask[:response_length],
        num_turns=num_turns,
        metrics={},
        verifier_responses=verifier_responses,
        statistics=statistics
    )
    return output

