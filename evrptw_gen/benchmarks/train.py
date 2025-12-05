import argparse
import os
import random
import shutil
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler

from evrptw_gen.benchmarks.models.attention_model_wrapper import Agent

from evrptw_gen.benchmarks.wrappers.recordWrapper import RecordEpisodeStatistics
from evrptw_gen.benchmarks.wrappers.syncVectorEnvPomo import SyncVectorEnv
from evrptw_gen.configs.load_config import Config

def print_env_info(env, keys=[]):
    env_num = len(env.envs) if hasattr(env, "envs") else 1
    print(f"Number of Environments: {env_num}")

    for i in range(env_num):
        print(f"\nEnvironment {i+1} Information:")
        single_env = env.envs[i] if hasattr(env, "envs") else env
        if keys:
            for key in keys:
                if hasattr(single_env, key):
                    print(f"  {key}: {getattr(single_env, key)}")
                else:
                    print(f"  {key}: Not found in environment.")
        else:
            for attr in dir(single_env):
                if not attr.startswith("_"):
                    print(f"  {attr}: {getattr(single_env, attr)}")

def print_env_info_shape(env, keys=[]):
    env_num = len(env.envs) if hasattr(env, "envs") else 1
    print(f"Number of Environments: {env_num}")

    for i in range(env_num):
        print(f"\nEnvironment {i+1} Information:")
        single_env = env.envs[i] if hasattr(env, "envs") else env
        if keys:
            for key in keys:
                if hasattr(single_env, key):
                    value = getattr(single_env, key)
                    if isinstance(value, np.ndarray):
                        print(f"  {key}: {value.shape}")
                else:
                    print(f"  {key}: Not found in environment.")
        else:
            for attr in dir(single_env):
                if not attr.startswith("_"):
                    value = getattr(single_env, attr)
                    if isinstance(value, np.ndarray):
                        print(f"  {key}: {value.shape}")



def make_env(env_id, seed, cfg={}):
    def thunk():
        env = gym.make(env_id, **cfg)
        env = RecordEpisodeStatistics(env)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

def train(args):
    #########################
    #### Env Definition #####
    #########################
    # Register the environment.
    # Note: entry_point must be a fully-qualified import path 
    # (details explained in the discussion above).
    gym.envs.register(
        id=args.env_id,
        entry_point=args.env_entry_point,
    )

    # Uncomment for debugging
    # breakpoint()

    #########################
    ### Model Definition ####
    #########################
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config_iter_number = 10
    agent = Agent(device=device, 
                  name=args.problem, 
                  tanh_clipping = args.tanh_clipping, 
                  n_encode_layers = args.n_encode_layers,
                  use_graph_token = True).to(device)

    optimizer = optim.AdamW(
        agent.parameters(), lr=args.learning_rate, eps=1e-5, weight_decay=args.weight_decay
    )

    #######################
    # Algorithm defintion #
    #######################

    # 1: for config (决定instance difficulty)
    # 2: for cus / cs size (选择好此时batch的cus / cs size)
    # 3: for iter: (训练次数 + PPO)
    
    config_iter_number = 1  # 或更多
    num_updates = 10
    for config_iter in range(config_iter_number):
        # 1. 选本轮使用的 config_path 和 n_traj
        # config_path = args.config_path[config_iter]
        # n_traj_num = args.n_traj[config_iter]
        config_path = args.config_path
        n_traj_num = args.n_traj

        # 2. 只在这里创建一套 envs（本 config 共用这一套）

        customer_numbers = 5
        charging_stations_numbers = 3
        for batch_iter in range(num_updates):
            envs = SyncVectorEnv(
                [
                    make_env(
                        args.env_id,
                        args.seed + i,
                        cfg={"config_path": config_path, "n_traj": n_traj_num, "num_customers": customer_numbers, "num_charging_stations": charging_stations_numbers},
                    )
                    for i in range(args.num_envs)
                ]
            )

            # For Debug
            # print_env_info_shape(envs, keys=["nodes"])

            # arg steps 应该根据customer_numbers 和 charging_stations_numbers 调整

            obs = [None] * args.num_steps
            actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
            logprobs = torch.zeros((args.num_steps, args.num_envs, args.n_traj)).to(device)
            rewards = torch.zeros((args.num_steps, args.num_envs, args.n_traj)).to(device)
            dones = torch.zeros((args.num_steps, args.num_envs, args.n_traj)).to(device)
            values = torch.zeros((args.num_steps, args.num_envs, args.n_traj)).to(device)

            # TRY NOT TO MODIFY: start the game
            global_step = 0
            next_done = torch.zeros(args.num_envs, args.n_traj).to(device)
            num_updates = args.total_timesteps // args.batch_size

            agent.train()
            next_obs = envs.reset()

            encoder_state = agent.backbone.encode(next_obs)
            next_done = torch.zeros(args.num_envs, args.n_traj).to(device)
            r = []

            ## Main Logic ##
            for step in range(args.num_steps):
                global_step += 1 * args.num_envs
                obs[step] = next_obs
                dones[step] = next_done
                
                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value, _ = agent.get_action_and_value_cached(
                        next_obs, state=encoder_state
                    )
                    action = action.view(args.num_envs, args.n_traj)
                    values[step] = value.view(args.num_envs, args.n_traj)
                actions[step] = action
                logprobs[step] = logprob.view(args.num_envs, args.n_traj)

                if step == args.num_steps - 1:
                    envs.update_attr("terminate", True)
                    
                next_obs, reward, done, info = envs.step(action.cpu().numpy())
                rewards[step] = torch.tensor(reward).to(device)
                next_obs, next_done = next_obs, torch.Tensor(done).to(device)

            ## PPO Logic ##
            # bootstrap value if not done
            with torch.no_grad():
                next_value = agent.get_value_cached(next_obs, encoder_state).squeeze(-1)  # B x T
                advantages = torch.zeros_like(rewards).to(device)  # steps x B x T
                lastgaelam = torch.zeros(args.num_envs, args.n_traj).to(device)  # B x T
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done  # next_done: B
                        nextvalues = next_value  # B x T
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]  # B x T
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = (
                        delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    )
                returns = advantages + values
            breakpoint()
            # flatten the batch
            b_obs = {
                k: np.concatenate([obs_[k] for obs_ in obs]) for k in envs.single_observation_space
            }

            # Edited
            b_logprobs = logprobs.reshape(-1, args.n_traj)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_actions_mask = actions_mask.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1, args.n_traj)
            b_returns = returns.reshape(-1, args.n_traj)
            b_values = values.reshape(-1, args.n_traj)

            # Optimizing the policy and value network
            assert args.num_envs % args.num_minibatches == 0
            envsperbatch = args.num_envs // args.num_minibatches
            envinds = np.arange(args.num_envs)
            flatinds = np.arange(args.batch_size).reshape(args.num_steps, args.num_envs)

            clipfracs = []

            for epoch in range(args.update_epochs):
                np.random.shuffle(envinds)
                for start in range(0, args.num_envs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]  # mini batch env id
                    mb_inds = flatinds[:, mbenvinds].ravel()  # be really careful about the index
                    valid_mask = b_actions_mask[mb_inds]
                    r_inds = np.tile(np.arange(envsperbatch), args.num_steps)

                    cur_obs = {k: v[mbenvinds] for k, v in obs[0].items()}
                    # Get mini batch of each trajactory
                    
                    # embedding, avg(encoder), glimpse_k, glimpse_v, logit_k
                    encoder_state = agent.backbone.encode(cur_obs)

                    # x =  {k: v[mb_inds] for k, v in b_obs.items()}, action = b_actions, state= (embedding[r_inds, :] for embedding in encoder_state)
                    _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value_cached(
                        {k: v[mb_inds] for k, v in b_obs.items()},
                        b_actions.long()[mb_inds],
                        (embedding[r_inds, :] for embedding in encoder_state),
                    )
                    # _, newlogprob, entropy, newvalue = agent.get_action_and_value({k:v[mb_inds] for k,v in b_obs.items()}, b_actions.long()[mb_inds])
                    logratio = (newlogprob - b_logprobs[mb_inds])[valid_mask]
                    ratio = (logratio.exp())

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds][valid_mask]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std() + 1e-8
                        )

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - args.clip_coef, 1 + args.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    # Value loss
                    newvalue = newvalue.view(-1, args.n_traj)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        # [valid_mask]
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max[valid_mask].mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2)[valid_mask].mean()

                    entropy_loss = entropy[valid_mask].mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                if args.target_kl is not None:
                    if approx_kl > args.target_kl:
                        break


            ## Update Next Environment ##

            # A policy to update the customer_numbers and charging_stations_numbers and other env parameters (Curriculum Learning)
            customer_numbers += 1
            charging_stations_numbers += 1





