# Copyright 2022 The HuggingFace Team
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
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import TYPE_CHECKING, Tuple

import numpy as np
import torch

from ..utils import torch_functional as VF


if TYPE_CHECKING:
    from .config import AlgorithmConfig


class KLController(ABC):
    @abstractmethod
    def update(self, current_kl: float, n_steps: int) -> None: ...


class AdaptiveKLController(KLController):
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef: float, target_kl: float, horizon: float):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl: float, n_steps: int) -> None:
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController(KLController):
    """Fixed KL controller."""

    def __init__(self, init_kl_coef: float):
        self.value = init_kl_coef

    def update(self, current_kl: float, n_steps: int) -> None:
        pass


def get_kl_controller(algorithm_config: "AlgorithmConfig") -> KLController:
    if algorithm_config.kl_type == "fixed":
        kl_ctrl = FixedKLController(init_kl_coef=algorithm_config.kl_coef)
    elif algorithm_config.kl_type == "adaptive":
        assert algorithm_config.kl_horizon > 0, f"horizon must be larger than 0. Got {algorithm_config.kl_horizon}."
        kl_ctrl = AdaptiveKLController(
            init_kl_coef=algorithm_config.kl_coef,
            target_kl=algorithm_config.kl_target,
            horizon=algorithm_config.kl_horizon,
        )
    else:
        raise ValueError(f"Unknown kl type: {algorithm_config.kl_type}.")

    return kl_ctrl


@torch.no_grad()
def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    eos_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    lastgaelam = 0
    advantages_reversed = []
    gen_len = token_level_rewards.shape[-1]
    for t in reversed(range(gen_len)):
        nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
        delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
        lastgaelam = delta + gamma * lam * lastgaelam
        advantages_reversed.append(lastgaelam)

    advantages = torch.stack(advantages_reversed[::-1], dim=1)
    returns = advantages + values
    advantages = VF.masked_whiten(advantages, eos_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
@torch.no_grad()
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor, eos_mask: torch.Tensor, index: torch.Tensor, epsilon: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    index:标识不同prompt组的索引
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
       Args:  
        token_level_rewards: `(torch.Tensor)`  
            shape: (bs, response_length)  
            # 包含了每个 token 可能的奖励。在 outcome supervision 的情况下，  
            # 通常只有一个非零值，位于响应序列的末尾，代表整个序列的标量奖励。  
        eos_mask: `(torch.Tensor)`  
            shape: (bs, response_length)  
            # 结束符 (End-Of-Sequence) 掩码。值为 1 的位置表示有效的响应 token，  
            # 通常在实际的 EOS token 处为 1，之后为 0。  
            # GRPO 论文中提到，优势被放置在 EOS token 的位置。 
            # 这里是responsemask，掩码出来的是响应部分的 token。 
        index: `(torch.Tensor)`  
            shape: (bs,)  
            # 每个样本的提示 (prompt) 索引。用于对具有相同提示的响应进行分组，  
            # 以便在同一提示下对它们的得分进行归一化。  
        epsilon: `(float)`  
            # 一个小的常数，用于防止在归一化时除以零（如果标准差为零）。  

    Returns:  
        advantages: `(torch.Tensor)`  
            shape: (bs, response_length)  
            # 计算得到的优势值。在 outcome supervision 的情况下，这通常是归一化后的标量奖励，  
            # 扩展到响应序列的长度，并由 eos_mask 掩码。  
        Returns: `(torch.Tensor)`  
            shape: (bs, response_length)  
            # 在这个特定的 GRPO outcome 实现中，返回值 (Returns) 与优势值 (advantages) 相同。  
            # 这是因为 GRPO 的 outcome 奖励直接作为优势，没有使用值函数进行基线扣除或 GAE 计算。  
    
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)
    id2mean, id2std = {}, {}

    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append(scores[i])

    for idx in id2score:
        if len(id2score[idx]) == 1:
            id2mean[idx] = torch.tensor(0.0)
            id2std[idx] = torch.tensor(1.0)
        elif len(id2score[idx]) > 1:
            id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
        else:
            raise ValueError(f"no score in prompt index: {idx}")

    for i in range(bsz):
        scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)

    scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask
    return scores, scores

@torch.no_grad()
def compute_grpo_outcome_advantage_hallucinations(
    token_level_rewards: torch.Tensor, eos_mask: torch.Tensor, index: torch.Tensor,hallucinations: torch.Tensor, epsilon: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    index:标识不同prompt组的索引
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)
    id2mean, id2std = {}, {}

    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append(scores[i])

    for idx in id2score:
        if len(id2score[idx]) == 1:
            id2mean[idx] = torch.tensor(0.0)
            id2std[idx] = torch.tensor(1.0)
        elif len(id2score[idx]) > 1:
            id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
        else:
            raise ValueError(f"no score in prompt index: {idx}")

    for i in range(bsz):
        scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)

    scores = scores.unsqueeze(-1).tile([1, response_length])*hallucinations * eos_mask

    return scores, scores
@torch.no_grad()
def compute_grpo_outcome_advantage_aug(
    status,token_level_rewards: torch.Tensor, eos_mask: torch.Tensor, index: torch.Tensor, epsilon: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    index:标识不同prompt组的索引
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)
    raw_scores = scores.clone()  # 保存原始得分用于后续计算
    
    # 原始分组逻辑（基于index）
    id2score = defaultdict(list)
    id2mean, id2std = {}, {}
    
    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append(scores[i])
    
    for idx in id2score:
        if len(id2score[idx]) == 1:
            id2mean[idx] = torch.tensor(0.0)
            id2std[idx] = torch.tensor(1.0)
        elif len(id2score[idx]) > 1:
            id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
        else:
            raise ValueError(f"no score in prompt index: {idx}")
    
    for i in range(bsz):
        scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
    
    group_scores = scores.unsqueeze(-1).expand(-1, response_length) * eos_mask
    
    # 2. 新增逻辑：计算基于(index, status)复合分组的归一化得分
    comp2score = defaultdict(list)
    comp2mean, comp2std = {}, {}
    
    # 构建复合分组字典
    for i in range(bsz):
        comp_key = (index[i], status[i])  # 使用(index, status)作为复合键
        comp2score[comp_key].append(raw_scores[i])  # 使用原始得分
    
    # 计算每个复合分组的统计量
    for comp_key in comp2score:
        group_scores_list = comp2score[comp_key]
        if len(group_scores_list) == 1:
            comp2mean[comp_key] = torch.tensor(0.0)
            comp2std[comp_key] = torch.tensor(1.0)
        else:
            scores_tensor = torch.stack(group_scores_list)
            comp2mean[comp_key] = torch.mean(scores_tensor)
            comp2std[comp_key] = torch.std(scores_tensor)
    
    # 计算复合分组归一化得分
    comp_scores = torch.zeros_like(raw_scores)
    for i in range(bsz):
        comp_key = (index[i], status[i])
        comp_scores[i] = (raw_scores[i] - comp2mean[comp_key]) / (comp2std[comp_key] + epsilon)
    
    comp_scores = comp_scores.unsqueeze(-1).expand(-1, response_length) * eos_mask
    
    # 返回原始分组得分（两份）和复合分组得分（两份）
    return group_scores, group_scores, comp_scores, comp_scores



@torch.no_grad()
def compute_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor, eos_mask: torch.Tensor, index: torch.Tensor, epsilon: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append(scores[i])

    for idx in id2score:
        if len(id2score[idx]) == 1:
            id2mean[idx] = torch.tensor(0.0)
        elif len(id2score[idx]) > 1:
            id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
        else:
            raise ValueError(f"no score in prompt index: {idx}.")

    for i in range(bsz):
        response_num = len(id2score[index[i]])
        if response_num > 1:
            scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[index[i]] * response_num / (
                response_num - 1
            )

    scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask
    return scores, scores


@torch.no_grad()
def compute_reinforce_plus_plus_outcome_advantage(
    token_level_rewards: torch.Tensor, eos_mask: torch.Tensor, gamma: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    returns = torch.zeros_like(token_level_rewards)
    running_return = 0
    for t in reversed(range(token_level_rewards.shape[1])):
        running_return = token_level_rewards[:, t] + gamma * running_return
        returns[:, t] = running_return
        # Reset after EOS
        running_return = running_return * eos_mask[:, t]

    advantages = VF.masked_whiten(returns, eos_mask)
    advantages = advantages * eos_mask
    return advantages, returns


@torch.no_grad()
def compute_remax_outcome_advantage(
    token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor, eos_mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505

    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    # scores = token_level_rewards.sum(dim=-1)
    returns = (token_level_rewards * eos_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
    advantages = returns - reward_baselines.unsqueeze(-1).tile([1, response_length]) * eos_mask
    return advantages, returns


def compute_rewards(
    token_level_scores: torch.Tensor,
    old_log_prob: torch.Tensor,
    ref_log_prob: torch.Tensor,
    kl_ratio: float,
) -> torch.Tensor:
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def compute_policy_loss1(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    eos_mask: torch.Tensor,
    cliprange: float,
    # aug_status = "normal",
) -> Tuple[torch.Tensor, float, float]:
    """Compute the policy loss.

    Adapted from https://github.com/huggingface/trl/blob/v0.15.0/trl/trainer/ppo_trainer.py#L568

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            a float number indicating the fraction of policy gradient loss being clipped
    """

    # clamp the ratio before exp to avoid nan
    # see: https://github.com/pytorch/pytorch/issues/10729
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    # clipped_ratio = torch.exp(torch.clamp(negative_approx_kl, torch.log(1.0 - cliprange), torch.log(1.0 + cliprange)))
    clipped_ratio = torch.exp(torch.clamp(negative_approx_kl, np.log(1.0 - cliprange), np.log(1.0 + cliprange)))
    ppo_kl = VF.masked_mean(-negative_approx_kl, eos_mask)

    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * clipped_ratio
    pg_loss = VF.masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)

    pg_clipfrac = VF.masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)
    return pg_loss, pg_clipfrac, ppo_kl

def compute_policy_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    eos_mask: torch.Tensor,
    cliprange: float,
    # aug_status = "normal",
) -> Tuple[torch.Tensor, float, float]:
      

    # clamp the ratio before exp to avoid nan
    # see: https://github.com/pytorch/pytorch/issues/10729
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    # clipped_ratio = torch.exp(torch.clamp(negative_approx_kl, torch.log(1.0 - cliprange), torch.log(1.0 + cliprange)))
    clipped_ratio = torch.exp(torch.clamp(negative_approx_kl, np.log(1.0 - cliprange), np.log(1.0 + cliprange)))
    ppo_kl = VF.masked_mean(-negative_approx_kl, eos_mask)

    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * clipped_ratio
    pg_loss = torch.max(pg_losses, pg_losses2)

    #统计被裁剪样本比例，用于调试更新幅度
    pg_clipfrac = VF.masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)
    return pg_loss, pg_clipfrac, ppo_kl


def compute_entropy_loss(logits: torch.Tensor, eos_mask: torch.Tensor) -> torch.Tensor:
    """Compute categorical entropy loss.

    Adapted from https://github.com/huggingface/trl/blob/v0.15.0/trl/trainer/ppo_trainer.py#L582

    Args:
        logits: `(torch.Tensor)`
            shape: (bs, response_length, vocab_size)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    entropy = VF.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = VF.masked_mean(entropy, mask=eos_mask)
    return entropy_loss


def compute_value_loss(
    vpreds: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    eos_mask: torch.Tensor,
    cliprange_value: float,
) -> Tuple[torch.Tensor, float]:
    """Compute the value loss.

    Copied from https://github.com/huggingface/trl/blob/v0.15.0/trl/trainer/ppo_trainer.py#L556

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange_value: (float)
            The clip range for value net used in PPO. See https://arxiv.org/abs/1707.06347

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped
    """
    vpredclipped = VF.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = torch.square(vpreds - returns)
    vf_losses2 = torch.square(vpredclipped - returns)
    vf_loss = 0.5 * VF.masked_mean(torch.max(vf_losses1, vf_losses2), eos_mask)
    vf_clipfrac = VF.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), eos_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty: str) -> torch.Tensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104

    Args:
        logprob: torch.Tensor
        ref_logprob: torch.Tensor

    Returns:
        kl_div: torch.Tensor
    """
    if kl_penalty == "kl":
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty == "low_var_kl":
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError
