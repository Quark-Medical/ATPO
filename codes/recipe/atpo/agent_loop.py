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
import asyncio
import heapq
import logging
import math
import os
import random
from abc import ABC, abstractmethod
from typing import Any, Optional, List

import hydra
import numpy as np
import ray
import torch
import threading
from cachetools import LRUCache
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict
from sympy.abc import alpha
from tensordict import TensorDict
from transformers import AutoProcessor, AutoTokenizer

from verl.protocol import DataProto
from verl.single_controller.ray.base import RayWorkerGroup
from verl.trainer.ppo.reward import load_reward_manager
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask
from verl.utils.rollout_trace import RolloutTraceConfig, rollout_trace_attr, rollout_trace_op
from verl.workers.rollout.async_server import TokenOutput, async_server_class

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AsyncLLMServerManager:
    """
    A class to manage multiple OpenAI compatible LLM servers. This class provides
    - Load balance: least requests load balancing
    - Sticky session: send multi-turn chat completions to same server for automatic prefix caching
    """

    def __init__(self, config: DictConfig, server_handles: list[ray.actor.ActorHandle], max_cache_size: int = 10000):
        """Initialize the AsyncLLMServerManager.

        Args:
            config (DictConfig): YAML config.
            server_handles (List[ray.actor.ActorHandle]): OpenAI compatible LLM server actor handles.
            max_cache_size (int, optional): max cache size for request_id to server mapping. Defaults to 10000.
        """
        self.config = config
        self.server_handles = server_handles
        random.shuffle(self.server_handles)

        # Least requests load balancing
        self.weighted_serveres = [[0, (hash(server), server)] for server in server_handles]
        heapq.heapify(self.weighted_serveres)

        # LRU cache to map request_id to server
        self.request_id_to_server = LRUCache(maxsize=max_cache_size)

    def _choose_server(self, request_id: str) -> ray.actor.ActorHandle:
        # TODO: implement server pressure awareness load balancing
        if request_id in self.request_id_to_server:
            return self.request_id_to_server[request_id]

        server = self.weighted_serveres[0][1][1]
        self.weighted_serveres[0][0] += 1
        heapq.heapreplace(self.weighted_serveres, self.weighted_serveres[0])
        self.request_id_to_server[request_id] = server
        return server

    @rollout_trace_op
    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
    ) -> TokenOutput:
        """Generate tokens from prompt ids.

        Args:
            request_id (str): request id for sticky session.
            prompt_ids (List[int]): List of prompt token ids.
            sampling_params (Dict[str, Any]): Sampling parameters for the chat completion.

        Returns:
            TokenOutput: token output
        """
        server = self._choose_server(request_id)
        output = await server.generate.remote(
            request_id=request_id,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            image_data=image_data,
        )
        return output


class AgentLoopMetrics(BaseModel):
    """Agent loop performance metrics."""
    generate_sequences: float = 0.0
    tool_calls: float = 0.0


class AgentLoopOutput(BaseModel):
    """Agent loop output."""

    prompt_ids: list[int]
    """Prompt token ids."""
    response_ids: list[int]
    """Response token ids including LLM generated token, tool response token."""
    response_mask: list[int]
    """Response mask, 1 for LLM generated token, 0 for tool response token."""
    response_logprobs: Optional[list[float]] = None
    """Log probabilities for the response tokens."""
    multi_modal_data: Optional[dict[str, Any]] = None
    """Multi-modal data for multi-modal tools."""
    reward_score: Optional[float] = None
    """Reward score for the trajectory."""
    num_turns: int = 0
    """Number of chat turns, including user, assistant, tool."""
    metrics: AgentLoopMetrics
    """Auxiliary performance metrics"""
    verifier_responses: list[str]
    advantages: Optional[list[float]] = None
    returns: Optional[list[float]] = None
    value_response_mask: list[float] = None
    statistics: Optional[dict[str, Any]] = None


class _InternalAgentLoopOutput(AgentLoopOutput):
    """Internal agent loop output with padded sequences."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt_ids: torch.Tensor
    """Padded prompt token ids."""
    response_ids: torch.Tensor
    """Padded response token ids."""
    input_ids: torch.Tensor
    """Padded input ids(prompt_ids + response_ids)."""
    position_ids: torch.Tensor
    """Padded position ids."""
    response_mask: torch.Tensor
    """Padded response mask."""
    attention_mask: torch.Tensor
    """Padded attention mask."""
    response_logprobs: Optional[torch.Tensor] = None
    """Padded log probabilities for the response tokens."""
    multi_modal_inputs: Optional[dict[str, torch.Tensor]] = None
    """Multi-modal inputs for processors (e.g., pixel_values, image_grid_thw)."""
    verifier_responses: list[str]
    advantages: Optional[torch.Tensor] = None
    returns: Optional[torch.Tensor] = None
    value_response_mask: Optional[torch.Tensor] = None


# make hydra.utils.instantiate happy
class _DummyConfig:
    def __init__(self, config: DictConfig) -> None:
        self.config = config


class AgentLoopBase(ABC):
    """An agent loop takes a input message, chat with OpenAI compatible LLM server and interact with various
    environments."""

    _class_initialized = False

    def __init__(
        self,
        trainer_config: _DummyConfig,
        server_manager: AsyncLLMServerManager,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        **kwargs,
    ):
        """Initialize agent loop, each sample will have its own loop instance.

        Args:
            trainer_config (_DummyConfig): trainer config.
            server_manager (AsyncLLMServerManager): OpenAI compatible LLM server manager.
            tokenizer (AutoTokenizer): Tokenizer for tokenize messages.
            processor (AutoProcessor): Processor for process messages.
        """
        self.init_class(config=trainer_config.config, tokenizer=tokenizer, processor=processor, **kwargs)
        self.config = trainer_config.config
        self.server_manager = server_manager
        self.tokenizer = tokenizer
        self.processor = processor
        self.loop = asyncio.get_running_loop()

    @classmethod
    def init_class(cls, config: DictConfig, tokenizer: AutoTokenizer, processor: AutoProcessor, **kwargs):
        """This is used to do heavy initialization work that should shared across all instances. It's only called once.

        Args:
            config (DictConfig): trainer config.
            tokenizer (AutoTokenizer): Tokenizer for tokenize messages.
            processor (AutoProcessor): Processor for process multi_modal data.
            **kwargs: extra kwargs from config file passed in by `hydra.utils.instantiate`.
        """
        if cls._class_initialized:
            return
        cls._class_initialized = True

    @abstractmethod
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Run agent loop to interact with LLM server and environment.

        Args:
            sampling_params (Dict[str, Any]): LLM sampling params.
            **kwargs: dataset fields from `verl.utils.dataset.RLHFDataset`.

        Returns:
            AgentLoopOutput: Agent loop output.
        """
        raise NotImplementedError


"""Agent loop registry: key is agent_name, value is a dict of agent loop config
used by hydra.utils.instantiate to initialize agent loop instance.

https://hydra.cc/docs/advanced/instantiate_objects/overview/
"""
_agent_loop_registry: dict[str, dict] = {}


def register(agent_name: str):
    """Register agent loop class."""

    def decorator(subclass: type[AgentLoopBase]) -> type[AgentLoopBase]:
        fqdn = f"{subclass.__module__}.{subclass.__qualname__}"
        _agent_loop_registry[agent_name] = {"_target_": fqdn}
        return subclass

    return decorator


@ray.remote(num_cpus=1)
class RewardManagerWorker:
    """Reward manager worker to compute reward score asynchronously to overlap with agent loop."""

    def __init__(self, config: DictConfig, local_path: str) -> None:
        tokenizer = hf_tokenizer(local_path, trust_remote_code=True)
        self.reward_manager = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        self.loop = asyncio.get_event_loop()

    async def compute_score(self, output: AgentLoopOutput, kwargs: dict) -> float:
        """Compute reward score for agent loop output.

        NOTE: Since `reward_manager.__call__` is blocking function, we run it in thread pool to
        compute multiple samples in parallel.

        Args:
            output (AgentLoopOutput): Agent loop output.
            kwargs (dict): Dataset fields from `verl.utils.dataset.RLHFDataset`.

        Returns:
            float: Reward score.
        """
        prompts = torch.tensor(output.prompt_ids, dtype=torch.long).unsqueeze(0)
        responses = torch.tensor(output.response_ids, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.ones((1, prompts.shape[1] + responses.shape[1]), dtype=torch.long)
        batch = TensorDict(
            {
                "prompts": prompts,  # [1, prompt_length]
                "responses": responses,  # [1, response_length]
                "attention_mask": attention_mask,  # [1, prompt_length + response_length]
            },
            batch_size=1,
        )
        non_tensor_batch = {
            **{k: np.array([v]) for k, v in kwargs.items()},
            "__num_turns__": np.array([output.num_turns]),
        }
        data = DataProto(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
        )
        reward_tensor = await self.loop.run_in_executor(
            None,
            self.reward_manager,
            data,
        )
        return reward_tensor.sum(dim=-1).item()


class CriticManagerWorker:
    def __init__(self, config: DictConfig, critic_work_group, batch_wait_time: float = 0.01, chunk_size=8):
        self.tokenizer = hf_tokenizer(config.critic.model.path, trust_remote_code=True)
        self.critic_work_group = critic_work_group
        self.queue = asyncio.Queue()
        self.batch_wait_time = batch_wait_time
        self.chunk_size = chunk_size
        self.is_running = False
        self.processing_task = None

    async def wake_up(self):
        if not self.is_running:
            self.is_running = True
            self.processing_task = asyncio.create_task(self._process_queue())
            print("Critic prediction service started")

    async def sleep(self):
        self.is_running = False
        if self.processing_task:
            await self.processing_task
        print("Critic prediction service stopped")

    async def predict(self, input_data: Any):
        result_future = asyncio.Future()
        # print('get async critic predict', input_data)
        await self.queue.put((input_data, result_future))
        return {'result_future': result_future}

    async def _process_queue(self):
        while self.is_running:
            await asyncio.sleep(self.batch_wait_time)

            if self.queue.empty():
                continue

            batch_data = []
            batch_futures = []

            while not self.queue.empty():
                data, future = await self.queue.get()
                batch_data.append(data)
                batch_futures.append(future)
                self.queue.task_done()

            if batch_data:
                results = await self._batch_predict(batch_data)
                for future, result in zip(batch_futures, results):
                    future.set_result(result)

    async def _batch_predict(self, batch_data):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._predict,
            batch_data
        )

    def _predict(self, batch_data):
        # print('critic batch predict input: {}'.format(len(batch_data)))
        true_batch_size = len(batch_data)
        input_batch = self._preprocess_data(batch_data)
        outputs = self.critic_work_group.compute_values(input_batch)
        values = outputs.batch["values"]
        # print('critic batch predict results: {}'.format(values.shape))
        # print('critic values,', values[0])
        results = values.tolist()
        results = results[:true_batch_size]
        return results

    def _preprocess_data(self, batch_data) -> DataProto:
        # ["responses", "input_ids", "response_mask", "attention_mask", "position_ids"]
        max_prompt_len = 0
        max_response_len = 0
        for item in batch_data:
            max_prompt_len = max(max_prompt_len, len(item["prompt_ids"]))
            max_response_len = max(max_response_len, len(item["response_ids"]))
        input_ids = list()
        attention_mask = list()
        responses = list()
        response_mask = list()
        # print('max prompt len:', max_prompt_len, 'max response len:', max_response_len)
        for item in batch_data:
            left_pad_num = max_prompt_len - len(item['prompt_ids'])
            right_pad_num = max_response_len - len(item['response_ids'])
            resp_ids = item["response_ids"] + [self.tokenizer.pad_token_id] * right_pad_num
            resp_mks = [1] * len(item['response_ids']) + [0] * right_pad_num
            inp_ids = [self.tokenizer.pad_token_id] * left_pad_num + item["prompt_ids"] + resp_ids
            # append to batch
            input_ids.append(inp_ids)
            attention_mask.append([0] * left_pad_num + [1] * len(item['prompt_ids']) + resp_mks)
            responses.append(resp_ids)
            response_mask.append(resp_mks)
        # print('input_ids', input_ids[0])
        # print('attention_mask', attention_mask[0])
        # print('responses', responses[0])
        # print('response_mask', response_mask[0])
        while len(input_ids) % self.chunk_size != 0:
            input_ids = input_ids + [input_ids[-1]]
            attention_mask = attention_mask + [attention_mask[-1]]
            responses = responses + [responses[-1]]
            response_mask = response_mask + [response_mask[-1]]
        tensor_dic = {
            "input_ids" : torch.tensor(input_ids, dtype=torch.int64),
            "attention_mask" : torch.tensor(attention_mask, dtype=torch.int64),
            "responses" : torch.tensor(responses, dtype=torch.int64),
            "response_mask" : torch.tensor(response_mask, dtype=torch.int64),
        }
        tensor_dic["position_ids"] = torch.clip(torch.cumsum(tensor_dic["attention_mask"], dim=-1) - 1, min=0, max=None)
        output = DataProto.from_dict(tensors=tensor_dic)
        return output


@ray.remote
class AgentLoopWorker:
    """Agent loop worker takes a batch of messages and run each message in an agent loop."""

    def __init__(self, config: DictConfig, server_handles: list[ray.actor.ActorHandle], critic_worker: CriticManagerWorker):
        """Initialize agent loop manager.

        Args:
            config (DictConfig): YAML config.
            server_handles (List[ray.actor.ActorHandle]): OpenAI compatible LLM server actor handles.
            critic_worker: CriticManagerWorker.
        """
        self.config = config
        self.server_manager = AsyncLLMServerManager(config, server_handles)
        self.critic_worker = critic_worker

        model_path = config.actor_rollout_ref.model.path
        self.model_name = "/".join(model_path.split("/")[-2:])
        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=True)
        self.processor = hf_processor(local_path, trust_remote_code=True)

        agent_loop_config_path = config.actor_rollout_ref.rollout.agent.agent_loop_config_path
        if agent_loop_config_path:
            agent_loop_configs = OmegaConf.load(agent_loop_config_path)
            for agent_loop_config in agent_loop_configs:
                _agent_loop_registry[agent_loop_config.name] = agent_loop_config
        if self.config.actor_rollout_ref.model.get("custom_chat_template", None) is not None:
            if self.processor is not None:
                self.processor.chat_template = self.config.actor_rollout_ref.model.custom_chat_template
            self.tokenizer.chat_template = self.config.actor_rollout_ref.model.custom_chat_template

        if self.config.reward_model.loop_enable:
            self.reward_manager_worker = RewardManagerWorker.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=ray.get_runtime_context().get_node_id(),
                    soft=False,
                ),
            ).remote(self.config, local_path)

        trace_config = self.config.actor_rollout_ref.rollout.get("trace", {})
        RolloutTraceConfig.init(
            self.config.trainer.project_name,
            self.config.trainer.experiment_name,
            trace_config.get("backend"),
            trace_config.get("token2text", False),
        )

    async def generate_sequences(self, batch: DataProto, statistic_metrics: dict[str, Any]) -> tuple[DataProto, list[int]]:
        """Generate sequences from agent loop.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        if self.critic_worker is not None:
            await self.critic_worker.wake_up()  
        config = self.config.actor_rollout_ref.rollout
        tree_search = self.config.tree_search
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )
        tree_search_params = dict(
            M_trajectories=tree_search.M_trajectories,
            N_candidates=tree_search.N_candidates,
            variance_threshold=tree_search.variance_threshold,
            pruning_enabled=tree_search.pruning_enabled,
            call_critic_enabled=tree_search.call_critic_enabled,
            only_use_critic_value=tree_search.only_use_critic_value
        )

        # override sampling params for validation
        if batch.meta_info.get("validate", False):
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["temperature"] = config.val_kwargs.temperature
            tree_search_params["M_trajectories"] = tree_search.M_trajectories_eval
            tree_search_params["N_candidates"] = tree_search.N_candidates_eval
            tree_search_params["variance_threshold"] = tree_search.variance_threshold_eval
            tree_search_params["pruning_enabled"] = tree_search.pruning_enabled_eval
            tree_search_params["call_critic_enabled"] = tree_search.call_critic_enabled_eval

        # by default, we assume it's a single turn agent
        if "agent_name" not in batch.non_tensor_batch:
            batch.non_tensor_batch["agent_name"] = np.array(["single_turn_agent"] * len(batch), dtype=object)

        if "index" in batch.non_tensor_batch:
            index = batch.non_tensor_batch["index"]
        else:
            index = np.arange(len(batch))

        trajectory_info = await get_trajectory_info(
            batch.meta_info.get("global_steps", -1), index.tolist(), batch.meta_info.get("validate", False)
        )

        tasks = []
        for i in range(len(batch)):
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
            kwargs["tree_search_params"] = tree_search_params
            kwargs['critic_worker'] = self.critic_worker
            kwargs['statistic_metrics'] = statistic_metrics
            tasks.append(asyncio.create_task(self._run_agent_loop(sampling_params, trajectory_info[i], **kwargs)))
        outputs = await asyncio.gather(*tasks)

        output, repeat_time_list = self._postprocess(outputs)
        return output, repeat_time_list

    async def _run_agent_loop(
        self,
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
        *,
        agent_name: str,
        **kwargs,
    ) -> list[_InternalAgentLoopOutput]:
        with rollout_trace_attr(
            step=trajectory["step"],
            sample_index=trajectory["sample_index"],
            rollout_n=trajectory["rollout_n"],
            validate=trajectory["validate"],
            name="agent_loop",
        ):
            assert agent_name in _agent_loop_registry, (
                f"Agent loop {agent_name} not registered, registered agent loops: {_agent_loop_registry.keys()}"
            )

            agent_loop_config = _agent_loop_registry[agent_name]
            agent_loop = hydra.utils.instantiate(
                config=agent_loop_config,
                trainer_config=_DummyConfig(config=self.config),
                server_manager=self.server_manager,
                tokenizer=self.tokenizer,
                processor=self.processor,
            )
            outputs: list[AgentLoopOutput] = await agent_loop.run(sampling_params, **kwargs)

            # Some AgentLoop may have already computed the reward score, e.g SWE-agent.
            # if output.reward_score is None and not self.config.reward_model.enable and self.config.reward_model.loop_enable:
            #     output.reward_score = await self.reward_manager_worker.compute_score.remote(output, kwargs)

            # NOTE: consistent with batch version of generate_sequences in vllm_rollout_spmd.py
            # prompt_ids: left padded with zeros (e.g., [0,0,0,0,1,2,3,4])
            # response_ids: right padded with zeros (e.g., [5,6,7,8,0,0,0,0])
            # input_ids: concatenation of prompt + response
            # Mask:
            # For example, if the prompt is [1,2,3,4] and the response is [5,6,7,(tool start)8,9(tool end),10,11,12]
            # - prompt_attention_mask: 0s for padding, 1s for tokens
            #   e.g., [0,0,0,0,1,1,1,1]
            # - response_attention_mask: 0s for padding, 1s for tokens
            #   e.g., [1,1,1,1,1,1,1,1,1,1,1,0,0,0,0]
            # attention_mask: concatenation of prompt_attention_mask and response_attention_mask
            #   e.g., [0,0,0,0,1,1,1,1(prompt),1,1,1,1,1,1,1,1,1,1,1,0,0,0,0(response)]
            # - response_mask: 1s for LLM generated tokens, 0 for tool response/padding tokens
            #   e.g., [1,1,1,1,1,1,1,(tool start),0,0(tool end),1,1,0,0,0,0]
            # - position_ids: sequential positions for tokens, starting at 0
            #   e.g., [0,0,0,0,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,0,0,0,0]
            internal_outputs = []
            for output in outputs:
                self.tokenizer.padding_side = "left"
                prompt_output = self.tokenizer.pad(
                    {"input_ids": output.prompt_ids},
                    padding="max_length",
                    max_length=self.config.actor_rollout_ref.rollout.prompt_length,
                    return_tensors="pt",
                    return_attention_mask=True,
                )
                if prompt_output["input_ids"].dim() == 1:
                    prompt_output["input_ids"] = prompt_output["input_ids"].unsqueeze(0)
                    prompt_output["attention_mask"] = prompt_output["attention_mask"].unsqueeze(0)

                self.tokenizer.padding_side = "right"
                response_output = self.tokenizer.pad(
                    {"input_ids": output.response_ids},
                    padding="max_length",
                    max_length=self.config.actor_rollout_ref.rollout.response_length,
                    return_tensors="pt",
                    return_attention_mask=True,
                )
                if response_output["input_ids"].dim() == 1:
                    response_output["input_ids"] = response_output["input_ids"].unsqueeze(0)
                    response_output["attention_mask"] = response_output["attention_mask"].unsqueeze(0)

                response_mask_output = self.tokenizer.pad(
                    {"input_ids": output.response_mask},
                    padding="max_length",
                    max_length=self.config.actor_rollout_ref.rollout.response_length,
                    return_tensors="pt",
                    return_attention_mask=False,
                )
                if response_mask_output["input_ids"].dim() == 1:
                    response_mask_output["input_ids"] = response_mask_output["input_ids"].unsqueeze(0)

                response_logprobs = None
                if output.response_logprobs is not None:
                    pad_size = self.config.actor_rollout_ref.rollout.response_length - len(output.response_logprobs)
                    response_logprobs = torch.tensor(output.response_logprobs + [0.0] * pad_size).unsqueeze(0)
                advantages = None
                if output.advantages is not None:
                    pad_size = self.config.actor_rollout_ref.rollout.response_length - len(output.advantages)
                    advantages_padded = output.advantages + [0.0] * pad_size
                    advantages = torch.tensor(advantages_padded, dtype=torch.float32).unsqueeze(0)
                returns = None
                if output.returns is not None:
                    pad_size = self.config.actor_rollout_ref.rollout.response_length - len(output.returns) + 5
                    returns_padded = output.returns + [0.0] * pad_size
                    returns = torch.tensor(returns_padded, dtype=torch.float32).unsqueeze(0)
                value_response_mask = None
                if output.value_response_mask is not None:
                    pad_size = self.config.actor_rollout_ref.rollout.response_length - len(output.value_response_mask) + 5
                    value_response_mask_padded = output.value_response_mask + [0] * pad_size
                    value_response_mask = torch.tensor(value_response_mask_padded, dtype=torch.float32).unsqueeze(0)

                response_mask = response_mask_output["input_ids"] * response_output["attention_mask"]
                attention_mask = torch.cat([prompt_output["attention_mask"], response_output["attention_mask"]], dim=1)
                input_ids = torch.cat([prompt_output["input_ids"], response_output["input_ids"]], dim=1)

                # Handle multi-modal inputs and position_ids calculation
                # Only support Qwen2VLImageProcessor for multi-modal processing currently
                # TODO: support other multi-modal inputs
                multi_modal_inputs = None
                if (
                    self.processor is not None
                    and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__
                ):
                    from verl.models.transformers.qwen2_vl import get_rope_index

                    images = output.multi_modal_data.get("image", None)
                    current_text = self.tokenizer.decode(input_ids.squeeze(0), skip_special_tokens=True)
                    multi_modal_inputs = self.processor(text=[current_text], images=images, return_tensors="pt")
                    multi_modal_inputs.pop("input_ids", None)
                    multi_modal_inputs.pop("attention_mask", None)

                    # We must use dict(multi_modal_inputs) to convert BatchFeature values to a new dict
                    # because np.array() only keeps the keys for BatchFeature.
                    multi_modal_inputs = dict(multi_modal_inputs)

                    image_grid_thw = multi_modal_inputs.get("image_grid_thw")
                    video_grid_thw = multi_modal_inputs.get("video_grid_thw")
                    second_per_grid_ts = multi_modal_inputs.get("second_per_grid_ts")

                    position_ids = get_rope_index(
                        self.processor,
                        input_ids=input_ids.squeeze(0),
                        image_grid_thw=image_grid_thw,
                        video_grid_thw=video_grid_thw,
                        second_per_grid_ts=second_per_grid_ts,
                        attention_mask=attention_mask.squeeze(0),
                    ).unsqueeze(0)  # (1, 3, seq_len)
                else:
                    position_ids = compute_position_id_with_mask(attention_mask)  # (1, seq_len)

                internal_outputs.append(_InternalAgentLoopOutput(
                            prompt_ids=prompt_output["input_ids"],
                            response_ids=response_output["input_ids"],
                            input_ids=input_ids,
                            position_ids=position_ids,
                            response_mask=response_mask,
                            attention_mask=attention_mask,
                            response_logprobs=response_logprobs,
                            multi_modal_inputs=multi_modal_inputs,
                            multi_modal_data=output.multi_modal_data,
                            reward_score=output.reward_score,
                            num_turns=output.num_turns,
                            metrics=output.metrics,
                            verifier_responses=output.verifier_responses,
                            advantages=advantages,
                            returns=returns,
                            value_response_mask=value_response_mask,
                            statistics=output.statistics
                        ))

            return internal_outputs

    def _postprocess(self, inputs: list[list[_InternalAgentLoopOutput]]) -> tuple[DataProto, list[int]]:
        """Process the padded outputs from _run_agent_loop and combine them into a batch."""
        # Convert lists back to tensors and stack them to create a batch.
        repeat_time_list = [len(sublist) for sublist in inputs]
        inputs = [item for sublist in inputs for item in sublist]
        prompt_ids = torch.cat([input.prompt_ids for input in inputs], dim=0)
        response_ids = torch.cat([input.response_ids for input in inputs], dim=0)
        response_mask = torch.cat([input.response_mask for input in inputs], dim=0)
        attention_mask = torch.cat([input.attention_mask for input in inputs], dim=0)
        input_ids = torch.cat([input.input_ids for input in inputs], dim=0)
        position_ids = torch.cat([input.position_ids for input in inputs], dim=0)

        optional_outputs = {}
        if inputs[0].response_logprobs is not None:
            optional_outputs["rollout_log_probs"] = torch.cat([input.response_logprobs for input in inputs], dim=0)
        if inputs[0].advantages is not None:
            optional_outputs["advantages"] = torch.cat([input.advantages for input in inputs], dim=0)
        if inputs[0].returns is not None:
            optional_outputs["returns"] = torch.cat([input.returns for input in inputs], dim=0)
        if inputs[0].value_response_mask is not None:
            optional_outputs["value_response_mask"] = torch.cat([input.value_response_mask for input in inputs], dim=0)

        batch = TensorDict(
            {
                "prompts": prompt_ids,  # [bsz, prompt_length]
                "responses": response_ids,  # [bsz, response_length]
                "response_mask": response_mask,  # [bsz, response_length]
                "input_ids": input_ids,  # [bsz, prompt_length + response_length]
                "attention_mask": attention_mask,  # [bsz, prompt_length + response_length]
                # position_ids: [bsz, 3, prompt_length + response_length] or [bsz, prompt_length + response_length]
                "position_ids": position_ids,
                **optional_outputs,
            },
            batch_size=len(inputs),
        )

        scores = [input.reward_score for input in inputs]
        if all(score is not None for score in scores):
            prompt_length = prompt_ids.size(1)
            response_length = attention_mask[:, prompt_length:].sum(dim=1) - 1
            rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)
            rm_scores[torch.arange(response_mask.size(0)), response_length] = torch.tensor(scores, dtype=torch.float32)
            batch["rm_scores"] = rm_scores

        non_tensor_batch = {
            "__num_turns__": np.array([input.num_turns for input in inputs], dtype=np.int32),
        }
        verifier_responses_list = [input.verifier_responses for input in inputs]
        num_samples = len(verifier_responses_list)
        verifier_responses_array = np.empty(num_samples, dtype=object)
        verifier_responses_array[:] = verifier_responses_list
        non_tensor_batch['verifier_responses'] = verifier_responses_array
        statistics_list = [input.statistics for input in inputs]
        non_tensor_batch['statistics'] = np.array(statistics_list)

        # Add multi_modal_inputs to non_tensor_batch if any samples have them
        multi_modal_inputs_list = [input.multi_modal_inputs for input in inputs]
        if any(mmi is not None for mmi in multi_modal_inputs_list):
            non_tensor_batch["multi_modal_inputs"] = np.array(multi_modal_inputs_list, dtype=object)

        metrics = [input.metrics.model_dump() for input in inputs]
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info={"metrics": metrics}), repeat_time_list


async def get_trajectory_info(step, index, validate):
    """Get trajectory info.

    Args:
        step (int): global steps in the trainer.
        index (list): form datastore extra_info.index column.
        validate (bool): whether is a validate step.

    Returns:
        list: trajectory.
    """
    trajectory_info = []
    rollout_n = 0
    for i in range(len(index)):
        if i > 0 and index[i - 1] == index[i]:
            rollout_n += 1
        else:
            rollout_n = 0
        trajectory_info.append({"step": step, "sample_index": index[i], "rollout_n": rollout_n, "validate": validate})
    return trajectory_info


class AgentLoopManager:
    """Agent loop manager that manages a group of agent loop workers."""

    def __init__(self, config: DictConfig, worker_group: RayWorkerGroup, critic_work_group=None):
        """Initialize agent loop manager.

        Args:
            config (DictConfig): trainer config.
            worker_group (RayWorkerGroup): ActorRolloutRef worker group.
            critic_work_group: (RayWorkerGroup): Critic worker group.
        """
        self.config = config
        self.worker_group = worker_group
        self.critic_work_group = critic_work_group
        if critic_work_group is not None:
            self.critic_manager_worker = CriticManagerWorker(config, critic_work_group, 0.02)
        else:
            self.critic_manager_worker = None

        self.statistic_metrics = {}  

        self._initialize_llm_servers()
        self._init_agent_loop_workers()

        # Initially we're in sleep mode.
        self.sleep()

    def _initialize_llm_servers(self):
        self.rollout_tp_size = self.config.actor_rollout_ref.rollout.tensor_model_parallel_size
        self.rollout_dp_size = self.worker_group.world_size // self.rollout_tp_size

        workers_info = ray.get(
            [
                worker.__ray_call__.remote(lambda self: ray.get_runtime_context().get_node_id())
                for worker in self.worker_group.workers
            ]
        )
        assert len(workers_info) == self.worker_group.world_size

        self.async_llm_servers = [None] * self.rollout_dp_size
        self.server_addresses = [None] * self.rollout_dp_size

        if self.config.actor_rollout_ref.rollout.agent.custom_async_server:
            server_class = async_server_class(
                rollout_backend=self.config.actor_rollout_ref.rollout.name,
                rollout_backend_module=self.config.actor_rollout_ref.rollout.agent.custom_async_server.path,
                rollout_backend_class=self.config.actor_rollout_ref.rollout.agent.custom_async_server.name,
            )
        else:
            server_class = async_server_class(rollout_backend=self.config.actor_rollout_ref.rollout.name)

        # Start all server instances, restart if address already in use.
        unready_dp_ranks = set(range(self.rollout_dp_size))
        while len(unready_dp_ranks) > 0:
            servers = {
                rollout_dp_rank: server_class.options(
                    # make sure AsyncvLLMServer colocates with its corresponding workers
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=workers_info[rollout_dp_rank * self.rollout_tp_size],
                        soft=False,
                    ),
                    name=f"async_llm_server_{rollout_dp_rank}",
                ).remote(self.config, self.rollout_dp_size, rollout_dp_rank, self.worker_group.name_prefix)
                for rollout_dp_rank in unready_dp_ranks
            }

            for rollout_dp_rank, server in servers.items():
                try:
                    address = ray.get(server.get_server_address.remote())
                    self.server_addresses[rollout_dp_rank] = address
                    self.async_llm_servers[rollout_dp_rank] = server
                    unready_dp_ranks.remove(rollout_dp_rank)
                except Exception:
                    ray.kill(server)
                    print(f"rollout server {rollout_dp_rank} failed, maybe address already in use, restarting...")

        # All server instances are ready, init AsyncLLM engine.
        ray.get([server.init_engine.remote() for server in self.async_llm_servers])

    def _init_agent_loop_workers(self):
        self.agent_loop_workers = []
        num_workers = self.config.actor_rollout_ref.rollout.agent.num_workers

        node_ids = [node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]
        for i in range(num_workers):
            # Round-robin scheduling over the all nodes
            node_id = node_ids[i % len(node_ids)]
            self.agent_loop_workers.append(
                AgentLoopWorker.options(
                    name=f"agent_loop_worker_{i}",
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_id, soft=True
                    ),
                ).remote(self.config, self.async_llm_servers, self.critic_manager_worker)
            )

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Split input batch and dispatch to agent loop workers.

        Args:
            prompts (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
        """
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.wake_up()
        chunkes = prompts.chunk(len(self.agent_loop_workers))
        outputs_and_repeat_time_lists = ray.get(
            [
                worker.generate_sequences.remote(chunk, self.statistic_metrics)
                for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=True)
            ]
        )
        outputs = []
        repeat_time_lists_flatten = []
        for output, repeat_time_list in outputs_and_repeat_time_lists:
            outputs.append(output)
            repeat_time_lists_flatten.extend(repeat_time_list)
        output = DataProto.concat(outputs)
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.sleep()

        # udpate statistics
        statistics = output.non_tensor_batch.pop('statistics')
        statistics_metric = self._update_statistics(statistics)

        # calculate performance metrics
        metrics = [output.meta_info["metrics"] for output in outputs]  # List[List[Dict[str, str]]]
        timing = self._performance_metrics(metrics, output)

        output.meta_info = {"timing": timing, "statistics_metric": statistics_metric}
        return output, repeat_time_lists_flatten

    def _performance_metrics(self, metrics: list[list[dict[str, str]]], output: DataProto) -> dict[str, float]:
        timing = {}
        t_generate_sequences = np.array([metric["generate_sequences"] for chunk in metrics for metric in chunk])
        t_tool_calls = np.array([metric["tool_calls"] for chunk in metrics for metric in chunk])
        timing["agent_loop/generate_sequences/min"] = t_generate_sequences.min()
        timing["agent_loop/generate_sequences/max"] = t_generate_sequences.max()
        timing["agent_loop/generate_sequences/mean"] = t_generate_sequences.mean()
        timing["agent_loop/tool_calls/min"] = t_tool_calls.min()
        timing["agent_loop/tool_calls/max"] = t_tool_calls.max()
        timing["agent_loop/tool_calls/mean"] = t_tool_calls.mean()

        # batch sequence generation is bounded by the slowest sample
        slowest = np.argmax(t_generate_sequences + t_tool_calls)
        attention_mask = output.batch["attention_mask"][slowest]
        prompt_length = output.batch["prompts"].shape[1]
        timing["agent_loop/slowest/generate_sequences"] = t_generate_sequences[slowest]
        timing["agent_loop/slowest/tool_calls"] = t_tool_calls[slowest]
        timing["agent_loop/slowest/prompt_length"] = attention_mask[:prompt_length].sum().item()
        timing["agent_loop/slowest/response_length"] = attention_mask[prompt_length:].sum().item()

        return timing

    def _update_statistics(self, statistics: list[dict[str, Any]]) :
        # EWMA alpha: New_Avg = α * Current_Value + (1 - α) * Old_Avg
        alpha = 0.4
        max_assistant_turns = self.config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns
        single_metric = {}
        def calc_single_values(input_info, mode_type):
            seen_uids = set()
            level2values = dict()
            values = []
            true_input_infos = []
            for v in input_info:
                if len(v) == 0: continue
                if type(v[0]) is list:
                    true_input_infos.extend(v)
                else:
                    true_input_infos.append(v)
            for v in true_input_infos:
                if v[0] in seen_uids: continue
                seen_uids.add(v[0])
                if str(v[1]) not in level2values:
                    level2values[str(v[1])] = []
                level2values[str(v[1])].append(v[2])
                values.append(v[2])
            single_metric[f'{mode_type}_mean'] = np.mean(values)
            single_metric[f'{mode_type}_std'] = np.std(values)
            for level in range(max_assistant_turns+1):
                if str(level) not in level2values: continue
                single_metric[f'{mode_type}_l{level}_mean'] = np.mean(level2values[str(level)])
                single_metric[f'{mode_type}_l{level}_std'] = np.std(level2values[str(level)])
        calc_single_values([x['q_value_variance_list'] for x in statistics], 'q_value_variance')
        calc_single_values([x['mdp_value_list'] for x in statistics], 'mdp_value')
        calc_single_values([x['critic_value_list'] for x in statistics], 'critic_value')
        for k, v in single_metric.items():
            if math.isnan(v): continue
            if k not in self.statistic_metrics:
                self.statistic_metrics[k] = v
            else:
                self.statistic_metrics[k] = alpha * v + (1 - alpha) * self.statistic_metrics[k]
        # print('single_metric:', single_metric)
        # print('statistic_metrics:', self.statistic_metrics)
        new_single_metric = {}
        for k, v in single_metric.items():
            new_single_metric['static/' + k] = v
        return new_single_metric


    def wake_up(self):
        """Wake up all rollout server instances."""
        ray.get([server.wake_up.remote() for server in self.async_llm_servers])

    def sleep(self):
        """Sleep all rollout server instances."""
        ray.get([server.sleep.remote() for server in self.async_llm_servers])
