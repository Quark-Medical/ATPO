# Set processed files for training
DATASET_FILES="['./data/train.parquet']"
TEST_DATASET_FILES="['./data/test.parquet']"

# Setup tensorboard logging
export TENSORBOARD_DIR="./logs/tensorboard"
export SAVE_CHECKPOINT_DIR='./save_checkpoints'
export OUTPUT_DIR='./outputs'


# Setup actor checkpoint
export ACTOR_LOAD="/path/to/actor_model"


# Resource info
export NNODES=1
export NODE_RANK=0

# ================= algorithm =================
adv_estimator=gae

use_kl_in_reward=False
kl_coef=0.001
use_kl_loss=True
kl_loss_coef=0.01

clip_ratio_low=0.2
clip_ratio_high=0.28

max_turns=10
max_prompt_length=1024
max_response_length=4096
actor_lr=1e-6

train_batch_size=128
ppo_mini_batch_size=128  
ppo_micro_batch_size_per_gpu=1
log_prob_micro_batch_size_per_gpu=1  
n_resp_per_prompt=1
n_resp_per_prompt_val=1

# ================= perfomance =================
infer_tp=2 
train_sp=1 
offload=True

actor_max_token_len_per_gpu=$(( (max_prompt_length + max_response_length) * 4 ))
log_prob_max_token_len_per_gpu=$(( actor_max_token_len_per_gpu * 2 ))

python3 -m recipe.atpo.mt_main \
        algorithm.adv_estimator=${adv_estimator} \
        algorithm.use_kl_in_reward=$use_kl_in_reward \
        algorithm.kl_ctrl.kl_coef=$kl_coef \
        algorithm.gamma=1.0 \
        algorithm.lam=1.0 \
        data.train_files="$DATASET_FILES" \
        data.val_files="$TEST_DATASET_FILES" \
        data.return_raw_chat=True \
        data.train_batch_size=$train_batch_size \
        data.max_prompt_length=$max_prompt_length \
        data.max_response_length=$max_response_length \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        +data.seed=1 \
        actor_rollout_ref.model.path=${ACTOR_LOAD} \
        actor_rollout_ref.model_type=qwen3 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
        actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low \
        actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high \
        actor_rollout_ref.actor.clip_ratio_c=10.0 \
        actor_rollout_ref.actor.optim.lr=$actor_lr \
        actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 \
        actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$actor_max_token_len_per_gpu \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=$train_sp \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.fsdp_config.param_offload=$offload \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=$offload \
        actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$log_prob_max_token_len_per_gpu \
        actor_rollout_ref.rollout.name=sglang \
        actor_rollout_ref.rollout.mode=async \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu} \
        actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp \
        actor_rollout_ref.rollout.multi_turn.enable=True \
        actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$max_turns \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
        actor_rollout_ref.rollout.n=$n_resp_per_prompt \
        actor_rollout_ref.rollout.top_p=0.8 \
        actor_rollout_ref.rollout.temperature=1.0 \
        actor_rollout_ref.rollout.calculate_log_probs=False \
        actor_rollout_ref.rollout.val_kwargs.top_p=0.6 \
        actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
        actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu} \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        actor_rollout_ref.rollout.agent.agent_loop_config_path='recipe/atpo/agent.yaml' \
        actor_rollout_ref.rollout.agent.num_workers=8 \
        custom_reward_function.path='recipe/atpo/mt_reward_fn.py' \
        custom_reward_function.name='mt_reward_fn' \
        reward_model.reward_manager='mt_reward_manager' \
        reward_model.loop_enable=False \
        critic.model.path=${ACTOR_LOAD} \
        critic.model.use_remove_padding=True \
        critic.model.fsdp_config.param_offload=False \
        critic.model.fsdp_config.optimizer_offload=False \
        critic.optim.lr=1e-5 \
        critic.ppo_epochs=1 \
        critic.loss_agg_mode=token-mean \
        tree_search.M_trajectories=128 \
        tree_search.N_candidates=4 \
        tree_search.variance_threshold=1.2 \
        tree_search.pruning_enabled=True \
        tree_search.only_use_critic_value=True \
        tree_search.call_critic_enabled=True \
        extra_params.use_critic_in_loop=True \
        reward_coefficients.correctness_score=3 \
        trainer.logger=['console','tensorboard'] \
        trainer.project_name='multi-turn-rl' \
        trainer.experiment_name='rl' \
        trainer.n_gpus_per_node=8 \
        trainer.critic_warmup=4 \
        trainer.val_before_train=True \
        trainer.log_val_generations=0 \
        trainer.nnodes=${NNODES} \
        trainer.save_freq=5 \
        trainer.default_local_dir=${SAVE_CHECKPOINT_DIR} \
        trainer.rollout_data_dir=${OUTPUT_DIR}/rollout_data \
        trainer.validation_data_dir=${OUTPUT_DIR}/validation_data \
        trainer.test_freq=5 \
        trainer.total_epochs=2 $@
