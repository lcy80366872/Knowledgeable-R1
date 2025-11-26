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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from typing import Any, Callable, Dict, List, Optional, Type

import numpy as np
import ray
import torch
import torchvision.transforms as T
import torch.nn.functional as F
from codetiming import Timer
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..single_controller.base import Worker
from ..single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from ..single_controller.ray.base import create_colocated_worker_cls
from ..utils import torch_functional as VF
from ..utils.dataset import RLHFDataset, collate_fn
from ..utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from ..utils.tracking import Tracking, ValGenerationsLogger
from ..utils.image_aug import augment_batch,augment_batch_noisy
from ..workers.fsdp_workers import FSDPWorker
from . import core_algos
from .config import PPOConfig
from .metrics import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics, reduce_metrics
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

WorkerType = Type[Worker]
system_prompt="""You are a helpful assistant. After the user asks a question, you first think carefully and then give the answer.
When responding, please keep the following points in mind:
- The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively.
- Output your final answer directly between the tag <answer> </answer> without any intermediate steps.
Here is an exmaple:
user's question: what is the capital of China? 
<think> reasoning process here </think>
<answer> BeiJing </answer>"""
def get_per_sample_advantages(advantages, mask):
    """
    Extract the advantage value of the first valid token from each sample.

    Args:
    
    advantages: Advantages tensor, shape [batch_size, seq_len]
    mask: Mask of valid tokens, shape [batch_size, seq_len]

    Returns:

    per_sample_advantages: Advantages representation of each sample, shape [batch_size]
    valid_samples: Mask of valid samples (samples with at least one valid token)
    """
   
    row_has_valid = mask.any(dim=1)
    first_valid_indices = mask.int().argmax(dim=1) 
    per_sample_advantages = advantages[torch.arange(advantages.size(0)), first_valid_indices]
    return per_sample_advantages, row_has_valid

def visualize_advantage_distribution(advantages, mask, group_mask=None, name="Advantages", bins=50,per_sample=True):
    """
    Visualize the distribution of advantages

    Args:
    advantages: Advantages tensor
    mask: Mask of valid tokens (same shape as advantages)
    group_mask: Grouping mask (optional, same shape as advantages)
    name: Statistic name, used for plot title
    bins: Number of bins for the histogram
    """
    if per_sample:
        advantages_per_sample, valid_mask = get_per_sample_advantages(advantages, mask)
        if group_mask is not None:
            group_mask_per_sample, _ = get_per_sample_advantages(group_mask.float(), mask)
            group_mask_per_sample = group_mask_per_sample.bool()
        else:
            group_mask_per_sample = None
        advantages = advantages_per_sample
        mask = valid_mask 
        group_mask = group_mask_per_sample
    plt.switch_backend('Agg')
    
    # 确保mask是布尔型
    mask_bool = mask.bool()
    
    # 提取有效优势值
    valid_advantages = advantages[mask_bool].cpu().numpy()
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Distribution of {name}', fontsize=16)
    
    # 总体分布
    axes[0, 0].hist(valid_advantages, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Overall Distribution')
    axes[0, 0].set_xlabel('Advantage Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(x=np.mean(valid_advantages), color='red', linestyle='--', label=f'Mean: {np.mean(valid_advantages):.4f}')
    axes[0, 0].legend()
    
    axes[0, 1].boxplot(valid_advantages)
    axes[0, 1].set_title('Box Plot')
    axes[0, 1].set_ylabel('Advantage Value')
    

    sns.kdeplot(valid_advantages, ax=axes[1, 0], fill=True, color='orange')
    axes[1, 0].set_title('Kernel Density Estimation')
    axes[1, 0].set_xlabel('Advantage Value')
    axes[1, 0].set_ylabel('Density')

    stats.probplot(valid_advantages, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot (Normality Check)')
    
    plt.tight_layout()

    if group_mask is not None:
        
        valid_group_mask = group_mask[mask_bool].cpu().numpy()
        if np.any(valid_group_mask):
            group1_advantages = valid_advantages[valid_group_mask]
        else:
            group1_advantages = np.array([])
        if np.any(~valid_group_mask):
            group2_advantages = valid_advantages[~valid_group_mask]
        else:
            group2_advantages = np.array([])
        
        fig2, axes2 = plt.subplots(1, 2, figsize=(15, 6))
        fig2.suptitle(f'Group Comparison of {name}', fontsize=16)
        
        # 分组直方图
        if len(group1_advantages) > 0 and len(group2_advantages) > 0:
            axes2[0].hist(group1_advantages, bins=bins, alpha=0.5, label='Group 1', color='blue')
            axes2[0].hist(group2_advantages, bins=bins, alpha=0.5, label='Group 2', color='red')
            axes2[0].set_title('Group Histograms')
            axes2[0].set_xlabel('Advantage Value')
            axes2[0].set_ylabel('Frequency')
            axes2[0].legend()
        elif len(group1_advantages) > 0:
            axes2[0].hist(group1_advantages, bins=bins, alpha=0.7, color='blue', label='Group 1')
            axes2[0].set_title('Group 1 Histogram')
            axes2[0].set_xlabel('Advantage Value')
            axes2[0].set_ylabel('Frequency')
            axes2[0].legend()
        elif len(group2_advantages) > 0:
            axes2[0].hist(group2_advantages, bins=bins, alpha=0.7, color='red', label='Group 2')
            axes2[0].set_title('Group 2 Histogram')
            axes2[0].set_xlabel('Advantage Value')
            axes2[0].set_ylabel('Frequency')
            axes2[0].legend()
        else:
            axes2[0].text(0.5, 0.5, 'No data available', ha='center', va='center', transform=axes2[0].transAxes)
            axes2[0].set_title('No Data')
        
        # 分组箱线图
        if len(group1_advantages) > 0 or len(group2_advantages) > 0:
            data_to_plot = []
            labels = []
            
            if len(group1_advantages) > 0:
                data_to_plot.append(group1_advantages)
                labels.append('Group 1')
            
            if len(group2_advantages) > 0:
                data_to_plot.append(group2_advantages)
                labels.append('Group 2')
            
            axes2[1].boxplot(data_to_plot, labels=labels)
            axes2[1].set_title('Group Box Plots')
            axes2[1].set_ylabel('Advantage Value')
        else:
            axes2[1].text(0.5, 0.5, 'No data available', ha='center', va='center', transform=axes2[1].transAxes)
            axes2[1].set_title('No Data')
        
        plt.tight_layout()
        return fig, fig2
    else:
        return fig, None
class Role(IntEnum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = auto()
    Rollout = auto()
    ActorRollout = auto()
    Critic = auto()
    RefPolicy = auto()
    RewardModel = auto()
    ActorRolloutRef = auto()


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REMAX = "remax"
    RLOO = "rloo"


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker."""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get("GPU", 0) for node, node_info in node_available_resources.items()}

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}."
            )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.KLController, kl_penalty="kl"):
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch["attention_mask"] #这个attentoonmask包括prompt+respone
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if "ref_log_prob" in data.batch.keys():
        kld = core_algos.kl_penalty(
            data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
        )  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = VF.masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # According to https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L880
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"critic/kl": current_kl, "critic/kl_coeff": beta}
    return data, metrics

def calculate_advantage_stats(advantages, mask, group_mask=None, name="Advantages",per_sample=True):
    """
    Computes advantage statistics

    Args:
    advantages: Advantages tensor
    mask: Mask of valid tokens (same shape as advantages)
    group_mask: Group mask (optional, same shape as advantages)
    name: Output name of the statistic

    Returns:
    stats_dict: Dictionary containing all statistics
    """
    if per_sample:
        # Extract the representative advantage value of each sample
        advantages_per_sample, valid_mask = get_per_sample_advantages(advantages, mask)
        if group_mask is not None:
            group_mask_per_sample, _ = get_per_sample_advantages(group_mask.float(), mask)
            group_mask_per_sample = group_mask_per_sample.bool()
        else:
            group_mask_per_sample = None
        advantages = advantages_per_sample
        mask = valid_mask 
        group_mask = group_mask_per_sample
    mask_bool = mask.bool()
    # 提取有效优势值
    valid_advantages = advantages[mask_bool]
    
    # 初始化统计字典
    stats_dict = {
        'group1': {},
        'group2': {},
        'overall': {}
    }
    
    if group_mask is not None:
        
        valid_group_mask = group_mask[mask_bool]
        if valid_group_mask.any():
            group1_advantages = valid_advantages[valid_group_mask]
            pos_group1 = group1_advantages > 0
            pos_group1_count = pos_group1.sum()
            neg_group1 = group1_advantages <= 0
            neg_group1_count = neg_group1.sum()
            
            group1_pos_rate = pos_group1_count / len(group1_advantages) if len(group1_advantages) > 0 else 0.0
            group1_neg_rate = neg_group1_count / len(group1_advantages) if len(group1_advantages) > 0 else 0.0
            
            group1_mean = group1_advantages.mean().item() if len(group1_advantages) > 0 else 0.0
            group1_pos_mean = group1_advantages[pos_group1].mean().item() if pos_group1_count > 0 else 0.0
            group1_neg_mean = group1_advantages[neg_group1].mean().item() if neg_group1_count > 0 else 0.0
            
            # 存储组1统计信息
            stats_dict['group1'] = {
                'positive_rate': group1_pos_rate,
                'negative_rate': group1_neg_rate,
                'mean': group1_mean,
                'positive_mean': group1_pos_mean,
                'negative_mean': group1_neg_mean,
                'count': len(group1_advantages)
            }
        else:
            stats_dict['group1'] = {
                'positive_rate': 0.0,
                'negative_rate': 0.0,
                'mean': 0.0,
                'positive_mean': 0.0,
                'negative_mean': 0.0,
                'count': 0
            }

        if (~valid_group_mask).any():
            group2_advantages = valid_advantages[~valid_group_mask]
            pos_group2 = group2_advantages > 0
            pos_group2_count = pos_group2.sum()
            neg_group2 = group2_advantages <= 0
            neg_group2_count = neg_group2.sum()
            
            group2_pos_rate = pos_group2_count / len(group2_advantages) if len(group2_advantages) > 0 else 0.0
            group2_neg_rate = neg_group2_count / len(group2_advantages) if len(group2_advantages) > 0 else 0.0
            
            group2_mean = group2_advantages.mean().item() if len(group2_advantages) > 0 else 0.0
            group2_pos_mean = group2_advantages[pos_group2].mean().item() if pos_group2_count > 0 else 0.0
            group2_neg_mean = group2_advantages[neg_group2].mean().item() if neg_group2_count > 0 else 0.0

            stats_dict['group2'] = {
                'positive_rate': group2_pos_rate,
                'negative_rate': group2_neg_rate,
                'mean': group2_mean,
                'positive_mean': group2_pos_mean,
                'negative_mean': group2_neg_mean,
                'count': len(group2_advantages)
            }
        else:
            stats_dict['group2'] = {
                'positive_rate': 0.0,
                'negative_rate': 0.0,
                'mean': 0.0,
                'positive_mean': 0.0,
                'negative_mean': 0.0,
                'count': 0
            }
        print(f"{name}:")
        print(f"  Group CK (True): positive rate {stats_dict['group1']['positive_rate']*100:.2f}%, negative rate {stats_dict['group1']['negative_rate']*100:.2f}%")
        print(f"    Mean: {stats_dict['group1']['mean']:.4f}, Positive mean: {stats_dict['group1']['positive_mean']:.4f}, Negative mean: {stats_dict['group1']['negative_mean']:.4f}")
        print(f"  Group PK (False): positive rate {stats_dict['group2']['positive_rate']*100:.2f}%, negative rate {stats_dict['group2']['negative_rate']*100:.2f}%")
        print(f"    Mean: {stats_dict['group2']['mean']:.4f}, Positive mean: {stats_dict['group2']['positive_mean']:.4f}, Negative mean: {stats_dict['group2']['negative_mean']:.4f}")

    if len(valid_advantages) > 0:
        overall_pos = valid_advantages > 0
        overall_pos_count = overall_pos.sum()
        overall_neg = valid_advantages <= 0
        overall_neg_count = overall_neg.sum()
        
        overall_mean = valid_advantages.mean().item()
        overall_pos_rate = overall_pos_count / len(valid_advantages)
        overall_neg_rate = overall_neg_count / len(valid_advantages)
        overall_pos_mean = valid_advantages[overall_pos].mean().item() if overall_pos_count > 0 else 0.0
        overall_neg_mean = valid_advantages[overall_neg].mean().item() if overall_neg_count > 0 else 0.0

        stats_dict['overall'] = {
            'positive_rate': overall_pos_rate,
            'negative_rate': overall_neg_rate,
            'mean': overall_mean,
            'positive_mean': overall_pos_mean,
            'negative_mean': overall_neg_mean,
            'count': len(valid_advantages)
        }
        
        if group_mask is None:
            print(f"{name}:")
        print(f"  Overall: positive rate {stats_dict['overall']['positive_rate'].item()*100:.2f}%, negative rate {stats_dict['overall']['negative_rate'].item()*100:.2f}%")
        print(f"    Mean: {stats_dict['overall']['mean']:.4f}, Positive mean: {stats_dict['overall']['positive_mean']:.4f}, Negative mean: {stats_dict['overall']['negative_mean']:.4f}")
    else:
        stats_dict['overall'] = {
            'positive_rate': 0.0,
            'negative_rate': 0.0,
            'mean': 0.0,
            'positive_mean': 0.0,
            'negative_mean': 0.0,
            'count': 0
        }
        print(f"{name}: No valid data available")
    
    return stats_dict
def compute_advantage(data: DataProto, adv_estimator: AdvantageEstimator, gamma=1.0, lam=1.0,is_aug=False):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.GAE:
        values = data.batch["values"]
        responses = data.batch["responses"]
        response_length = responses.size(-1)
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch["token_level_rewards"]
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=token_level_rewards, values=values, eos_mask=response_mask, gamma=gamma, lam=lam
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GRPO:
        token_level_rewards = data.batch["token_level_rewards"]
        index = data.non_tensor_batch["uid"]
        responses = data.batch["responses"]
        response_length = responses.size(-1)
        attention_mask = data.batch["attention_mask"]
        
        response_mask = attention_mask[:, -response_length:]
        status = data.non_tensor_batch["aug_status"]
        if is_aug:
            #Global advantages are obtained by normalizing overall parametric and contextual knowledge through rewards, while local advantages are normalized within their respective types of knowledge groups  
            global_advantages, returns,local_advantages, local_returns = core_algos.compute_grpo_outcome_advantage_aug(
                status, token_level_rewards=token_level_rewards, eos_mask=response_mask, index=index
            )
            is_ck = torch.tensor([s == "ck" for s in status], 
                            device=global_advantages.device, 
                            dtype=torch.bool).unsqueeze(1)
                            
            rewards = token_level_rewards.sum(dim=-1)
            is_ck_reward = is_ck.squeeze(1)  

            # compute mean CK rewad 
            ck_rewards = rewards[is_ck_reward]
            ck_avg = ck_rewards.mean() if len(ck_rewards) > 0 else torch.tensor(0.0, device=rewards.device)

            # compute mean RPK rewad 
            rpk_rewards = rewards[~is_ck_reward]
            rpk_avg = rpk_rewards.mean() if len(rpk_rewards) > 0 else torch.tensor(0.0, device=rewards.device)
            print(f"ck group average reward: {ck_avg.item():.4f}")
            print(f"rpk group average reward: {rpk_avg.item():.4f}")
            ck_rewards_avg = torch.full((global_advantages.size(0),), ck_avg.item(), 
                 device=global_advantages.device, dtype=global_advantages.dtype)
            pk_rewards_avg = torch.full((global_advantages.size(0),), rpk_avg.item(), 
                 device=global_advantages.device, dtype=global_advantages.dtype)
    
            is_ck_expanded = is_ck.expand(-1, global_advantages.size(1))
            stats_global = calculate_advantage_stats(
                global_advantages, 
                response_mask, 
                is_ck_expanded, 
                "global_advantages"
            )
            response_len = torch.sum(response_mask, dim=1)

            stats_local = calculate_advantage_stats(
                local_advantages, 
                response_mask, 
                is_ck_expanded, 
                "local_advantages"
            )
            # A'= A_global + A_local
            advantages_CK = global_advantages+local_advantages
            stats_CK = calculate_advantage_stats(
                advantages_CK, 
                response_mask, 
                is_ck_expanded, 
                "CK_advantages"
            )
            stats_RPK= stats_global 
            E_ck_adv = stats_CK['group1']['mean']
            
            E_rpk_adv_pos = stats_RPK['group2']['positive_rate']*stats_RPK['group2']['positive_mean']
            E_rpk_adv_neg = stats_RPK['group2']['negative_rate']*stats_RPK['group2']['negative_mean']
            beta= (E_ck_adv-  E_rpk_adv_pos)/E_rpk_adv_neg
            beta= max(beta,0.01)
            if E_ck_adv<0:
                beta=1
            # Apply an asymmetric transformation for Advantages_RPK（A_hat） 
            advantages_RPK=F.leaky_relu(global_advantages, negative_slope=beta)
            beta = torch.full((global_advantages.size(0),), beta, 
                 device=global_advantages.device, dtype=global_advantages.dtype)
            print("beta:",torch.mean(beta))

            RPK_trans_status= calculate_advantage_stats(
                advantages_RPK, 
                response_mask, 
                is_ck_expanded, 
                "Advantages_inner (after LeakyReLU)"
            )
            #"Choose different advantages based on various sources of knowledge which is using for computing J_CK and J_RPK
            advantages = torch.where(is_ck, advantages_CK, advantages_RPK)  
            data.batch["advantages_pk"] = global_advantages  #this is using for compute J_PK
            data.batch["beta"] = torch.tensor(beta)
            data.batch["ck_reward"] = torch.tensor(ck_rewards_avg)
            data.batch["pk_reward"] = torch.tensor(pk_rewards_avg)
            data.batch["beta"] = torch.tensor(beta)
            
        else:
            advantages, returns = core_algos.compute_grpo_outcome_advantage(
                token_level_rewards=token_level_rewards, eos_mask=response_mask, index=index
            )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        token_level_rewards = data.batch["token_level_rewards"]
        responses = data.batch["responses"]
        response_length = responses.size(-1)
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=token_level_rewards, eos_mask=response_mask, gamma=gamma
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        token_level_rewards = data.batch["token_level_rewards"]
        index = data.non_tensor_batch["uid"]
        responses = data.batch["responses"]
        response_length = responses.size(-1)
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]
        reward_baselines = data.batch["reward_baselines"]
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=token_level_rewards, reward_baselines=reward_baselines, eos_mask=response_mask
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        token_level_rewards = data.batch["token_level_rewards"]
        index = data.non_tensor_batch["uid"]
        responses = data.batch["responses"]
        response_length = responses.size(-1)
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=token_level_rewards, eos_mask=response_mask, index=index
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        raise NotImplementedError

    return data


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield

    timing_raw[name] = timer.last


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config: PPOConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        reward_fn: Callable = None,
        val_reward_fn: Callable = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.worker.hybrid_engine
        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, (
                f"ActorRollout should be included in {role_worker_mapping.keys()}."
            )
        else:
            raise NotImplementedError

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_reward_model = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.val_generations_logger = ValGenerationsLogger()

        # define KL control
        if self.use_reference_policy:
            self.kl_ctrl = core_algos.get_kl_controller(config.algorithm)
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.0)

        if config.algorithm.adv_estimator not in list(AdvantageEstimator):
            raise NotImplementedError(f"Unknown advantage estimator: {config.algorithm.adv_estimator}.")

        if config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        else:
            self.use_critic = False

        if self.config.data.rollout_batch_size % self.config.worker.actor.global_batch_size != 0:
            raise ValueError("Rollout batch size must be divisible by global batch size.")

        if self.use_critic and self.config.data.rollout_batch_size % self.config.worker.critic.global_batch_size != 0:
            raise ValueError("Rollout batch size must be divisible by global batch size.")

        self._create_dataloader()

    def _create_dataloader(self) -> None:
        if self.config.data.dataset_name=="math":
            prompt = self.config.data.system_prompt
        else:
            prompt =system_prompt
        self.train_dataset = RLHFDataset(
            data_path=self.config.data.train_files,
            tokenizer=self.tokenizer,
            processor=self.processor,
            prompt_key=self.config.data.prompt_key,
            answer_key=self.config.data.answer_key,
            image_key=self.config.data.image_key,
            max_prompt_length=self.config.data.max_prompt_length,
            truncation="right",
            system_prompt=prompt,
            min_pixels=self.config.data.min_pixels,
            max_pixels=self.config.data.max_pixels,
            if_augment=self.config.data.if_augment,
            dataset_name=self.config.data.dataset_name,
            use_rag=self.config.data.use_rag
        )
        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.seed)
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)
       
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.rollout_batch_size,
            sampler=sampler,
            num_workers=8,
            collate_fn=collate_fn,
            pin_memory=False,
            drop_last=True,
            
        )
        # print("system prompt: \n",self.config.data.system_prompt)
        self.val_dataset = RLHFDataset(
            data_path=self.config.data.val_files,
            tokenizer=self.tokenizer,
            processor=self.processor,
            prompt_key=self.config.data.prompt_key,
            answer_key=self.config.data.answer_key,
            image_key=self.config.data.image_key,
            max_prompt_length=self.config.data.max_prompt_length,
            truncation="right",
            system_prompt=prompt,
            min_pixels=self.config.data.min_pixels,
            max_pixels=self.config.data.max_pixels,
            if_augment=False,
            dataset_name=self.config.data.dataset_name,
            is_val=True,
            use_rag=self.config.data.use_rag
        )
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=len(self.val_dataset),
            shuffle=False,
            num_workers=8,
            collate_fn=collate_fn,
            pin_memory=False,
            drop_last=False,
        )
        print(f"Size of train dataloader: {len(self.train_dataloader)}")

        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) == 1

        if self.config.trainer.max_steps is not None:
            training_steps = self.config.trainer.max_steps
        else:
            training_steps = len(self.train_dataloader) * self.config.trainer.total_episodes

        self.training_steps = training_steps
        self.config.worker.actor.optim.training_steps = training_steps
        self.config.worker.critic.optim.training_steps = training_steps
        print(f"Total training steps: {self.training_steps}")

    def _maybe_log_val_generations(self, inputs: List[str], outputs: List[str], scores: List[float]) -> None:
        """Log a table of validation samples"""
        if self.config.trainer.val_generations_to_log == 0:
            return

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        samples = samples[: self.config.trainer.val_generations_to_log]
        self.val_generations_logger.log(self.config.trainer.logger, samples, self.global_step)

    def _validate(self) -> Dict[str, Any]:
        reward_tensor_lst = []
        # Lists to collect samples for the table
        sample_inputs, sample_outputs, sample_scores = [], [], []
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            if "multi_modal_inputs" in test_batch.non_tensor_batch.keys():
                test_gen_batch = test_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data", "multi_modal_inputs"],
                )
            else:
                test_gen_batch = test_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids"],
                )

            test_gen_batch.meta_info = {"do_sample": False}
            test_gen_batch, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            test_output_gen_batch = self.actor_rollout_wg.generate_sequences(test_gen_batch)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch, pad_size=pad_size)
            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            reward_tensor = self.val_reward_fn(test_batch)

            # Store scores
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)
        reward_score = torch.cat(reward_tensor_lst, dim=0).sum(-1).mean().item()
        return {"val/test_score": reward_score}

    def init_workers(self) -> None:
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout], config=self.config.worker, role="actor_rollout"
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=self.config.worker, role="critic"
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy], config=self.config.worker, role="ref"
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_reward_model:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RewardModel], config=self.config.worker, role="reward"
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg: FSDPWorker = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg: FSDPWorker = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_reward_model:
            self.rm_wg: FSDPWorker = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg: FSDPWorker = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self) -> None:
        # path: {save_checkpoint_path}/global_step_{global_step}/actor
        folder_path = os.path.join(self.config.trainer.save_checkpoint_path, f"global_step_{self.global_step}")
        actor_path = os.path.join(folder_path, "actor")

        self.actor_rollout_wg.save_checkpoint(
            actor_path,
            self.global_step,
            remove_previous_ckpt=self.config.trainer.remove_previous_ckpt,
        )

        if self.use_critic:
            critic_path = os.path.join(folder_path, "critic")
            self.critic_wg.save_checkpoint(
                critic_path,
                self.global_step,
                remove_previous_ckpt=self.config.trainer.remove_previous_ckpt,
            )

        dataloader_path = os.path.join(folder_path, "dataloader.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_path)

        last_global_step_path = os.path.join(self.config.trainer.save_checkpoint_path, "latest_global_step.txt")
        with open(last_global_step_path, "w") as f:
            f.write(str(self.global_step))

    def _load_checkpoint(self) -> None:
        if self.config.trainer.load_checkpoint_path is None:
            return

        if "global_step_" not in self.config.trainer.load_checkpoint_path.strip(os.path.sep).split(os.path.sep)[-1]:
            raise ValueError("`load_checkpoint_path` should end with `global_step_*`.")

        print(f"Load from checkpoint: {self.config.trainer.load_checkpoint_path}.")
        self.global_step = int(self.config.trainer.load_checkpoint_path.strip(os.path.sep).split("global_step_")[-1])
        actor_path = os.path.join(self.config.trainer.load_checkpoint_path, "actor")
        self.actor_rollout_wg.load_checkpoint(
            actor_path, remove_ckpt_after_load=self.config.trainer.remove_ckpt_after_load
        )
        if self.use_critic:
            critic_path = os.path.join(self.config.trainer.load_checkpoint_path, "critic")
            self.critic_wg.load_checkpoint(
                critic_path, remove_ckpt_after_load=self.config.trainer.remove_ckpt_after_load
            )

        dataloader_path = os.path.join(self.config.trainer.load_checkpoint_path, "dataloader.pt")
        if os.path.exists(dataloader_path):
            dataloader_state_dict = torch.load(dataloader_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"No dataloader state found at {dataloader_path}, will start from scratch.")

    def _balance_batch(self, batch: DataProto, metrics: Dict[str, Any], logging_prefix: str = "global_seqlen") -> None:
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=self.config.to_dict(),
        )
        val_metrics: Optional[Dict[str, Any]] = None
        self.global_step = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.val_before_train:
            val_metrics = self._validate()
            print(f"Initial validation metrics: {val_metrics}.")
            logger.log(data=val_metrics, step=self.global_step)
            if self.config.trainer.val_only:
                return
        is_augment =self.config.data.if_augment
        is_noisy = self.config.worker.actor.is_noisy
        for ep_id in range(self.config.trainer.total_episodes):
            for batch_dict in self.train_dataloader:
                self.global_step += 1
                if self.global_step > self.training_steps:
                    break

                metrics, timing_raw = {}, {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                # pop those keys for generation
                if "multi_modal_inputs" in batch.non_tensor_batch.keys():
                    gen_batch = batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data", "multi_modal_inputs"],
                    )
                    if is_augment:
                        aug_batch: DataProto = DataProto.from_single_dict_aug(batch_dict)
                        aug_gen_batch = aug_batch.pop(
                            batch_keys=["input_ids", "attention_mask", "position_ids"],
                            non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data", "multi_modal_inputs"],
                        )
                        if "input_ids_pk" in batch.batch:
                            if "is_cf" in batch.non_tensor_batch:
                                aug_gen_batch_pk = batch.pop(
                                    batch_keys=["input_ids_pk", "attention_mask_pk", "position_ids_pk"],
                                    non_tensor_batch_keys=["is_cf"],
                                )
                            else:
                                aug_gen_batch_pk = batch.pop(
                                    batch_keys=["input_ids_pk", "attention_mask_pk", "position_ids_pk"],
                                )
                        
                            aug_gen_batch.union(aug_gen_batch_pk)
                    
                            gen_batch.union(aug_gen_batch_pk)

                else:
                    gen_batch = batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                    )
                    
                    if is_augment:
                        aug_batch: DataProto = DataProto.from_single_dict_aug(batch_dict)
                        aug_gen_batch = aug_batch.pop(
                            batch_keys=["input_ids", "attention_mask", "position_ids"],
                            non_tensor_batch_keys=["raw_prompt_ids"],
                        )
                        if "input_ids_pk" in batch.batch:
                            if "is_cf" in batch.non_tensor_batch:
                                aug_gen_batch_pk = batch.pop(
                                    batch_keys=["input_ids_pk", "attention_mask_pk", "position_ids_pk"],
                                    non_tensor_batch_keys=["is_cf"]
                                )
                            else:
                                aug_gen_batch_pk = batch.pop(
                                batch_keys=["input_ids_pk", "attention_mask_pk", "position_ids_pk"], 
                                )
                            
                            aug_gen_batch.union(aug_gen_batch_pk)
                            gen_batch.union(aug_gen_batch_pk)
                            
                        
                with _timer("step", timing_raw):
                    # generate a batch
                    with _timer("gen", timing_raw):  # wg: worker group
                        if is_augment:
                            
                            gen_batch = DataProto.concat([gen_batch, aug_gen_batch])
                            
                            gen_batch = gen_batch.interleave_batches(2)
                        
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)


                        gen_batch_output.non_tensor_batch["aug_status"] = \
                            np.array(["ck" if (i // self.config.worker.rollout.n) % 2 == 0 
                                      else "pk" for i in range(len(gen_batch_output.batch))], dtype=object)
                    if self.config.algorithm.adv_estimator == "remax":
                        with _timer("gen_max", timing_raw): 
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                            batch.batch["reward_baselines"] = reward_baseline_tensor
                            del gen_baseline_batch, gen_baseline_output
                    batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                    )

                    repeat = 2*self.config.worker.rollout.n if is_augment else self.config.worker.rollout.n
                    batch = batch.repeat(repeat_times=repeat, interleave=True)
                    batch = batch.union(gen_batch_output)

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        batch = batch.union(old_log_prob)
                    if "input_ids_pk" in batch.batch:
                        with _timer("old_log_prob_pk", timing_raw):
                            batch_pk =  deepcopy(batch)
                            batch_pk.batch["input_ids"]=batch.batch["input_ids_pk"]
                            batch_pk.batch["attention_mask"]=batch.batch["attention_mask_pk"]
                            batch_pk.batch["position_ids"]=batch.batch["position_ids_pk"]
                            old_log_prob_pk = self.actor_rollout_wg.compute_log_prob(batch_pk)
                            old_log_prob_pk.batch["old_log_probs_pk"] = old_log_prob_pk.batch["old_log_probs"]
                            old_log_prob_pk.batch.pop("old_log_probs")
                            if hasattr(batch.batch, 'lock_'):
                                batch.batch.unlock_()
                            batch = batch.union(old_log_prob_pk)
                            
                            

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # reward_fn should combine the results from reward model and rule-based results
                        if self.use_reward_model:
                            raise NotImplementedError("RM is not supported for PPO yet.")

                        # we combine with rule-based rm
                        reward_tensor = self.reward_fn(batch)
                        
                        batch.batch["token_level_scores"] = reward_tensor

                        # compute rewards. apply_kl_penalty if available
                        if not self.config.worker.actor.use_kl_loss:  # not grpo's kl loss
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]


                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            is_aug =is_augment
                        )

                    # update critic
                    #GRPO do not use critic, so we skip the critic update
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)

                        critic_metrics = reduce_metrics(critic_output.non_tensor_batch)
                        metrics.update(critic_metrics)

                    # update actor
                    if self.config.trainer.critic_warmup <= self.global_step:
                        with _timer("update_actor", timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)

                        actor_metrics = reduce_metrics(actor_output.non_tensor_batch)
                        metrics.update(actor_metrics)

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.val_freq > 0
                        and self.global_step % self.config.trainer.val_freq == 0
                    ):
                        with _timer("testing", timing_raw):
                            val_metrics = self._validate()

                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # collect metrics
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_step)

        # perform validation after training
        if self.val_reward_fn is not None:
            if val_metrics is None or self.global_step % self.config.trainer.val_freq != 0:
                val_metrics = self._validate()
                logger.log(data=val_metrics, step=self.global_step)

            print(f"Final validation metrics: {val_metrics}.")
