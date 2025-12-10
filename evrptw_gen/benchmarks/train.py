import os
import time
from tqdm import tqdm
from distutils.util import strtobool

import warnings

warnings.filterwarnings(
    "ignore",
    message="WARN: A Box observation space has an unconventional shape*",
    category=UserWarning,
)

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
    device = f"cuda:{args.cuda_id}" if torch.cuda.is_available() else "cpu"
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
    num_updates = 1000
    customer_numbers = 100
    charging_stations_numbers = 20
    # num_updates = args.total_timesteps // args.batch_size
    
    for config_iter in range(config_iter_number):
        # 1. 选本轮使用的 config_path 和 n_traj
        # config_path = args.config_path[config_iter]
        # n_traj_num = args.n_traj[config_iter]
        config_path = args.config_path
        n_traj_num = args.n_traj
        test_traj_num = args.test_agent
        # 2. 只在这里创建一套 envs（本 config 共用这一套）
        for update_step in tqdm(range(num_updates)):
            DEBUG_TEST = True
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

            obs = [None] * args.num_steps
            actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
            logprobs = torch.zeros((args.num_steps, args.num_envs, args.n_traj)).to(device)
            rewards = torch.zeros((args.num_steps, args.num_envs, args.n_traj)).to(device)
            dones = torch.zeros((args.num_steps, args.num_envs, args.n_traj)).to(device)
            values = torch.zeros((args.num_steps, args.num_envs, args.n_traj)).to(device)
            valid_masks = torch.zeros((args.num_steps, args.num_envs, args.n_traj)).to(device)

            # TRY NOT TO MODIFY: start the game
            global_step = 0
            agent.train()
            next_obs = envs.reset()

            encoder_state = agent.backbone.encode(next_obs)
            next_done = torch.zeros(args.num_envs, args.n_traj).to(device)
            alive = torch.ones(args.num_envs, args.n_traj, dtype=torch.bool, device=device)

            r = []

            ## Main Logic ##
            cur_a = []
            for step in range(args.num_steps):
                global_step += 1 * args.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                valid_masks[step] = alive

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value, _ = agent.get_action_and_value_cached(
                        next_obs, state=encoder_state
                    )
                    action = action.view(args.num_envs, args.n_traj)
                    values[step] = value.view(args.num_envs, args.n_traj)
                actions[step] = action
                logprobs[step] = logprob.view(args.num_envs, args.n_traj)

                cur_a.append(action[0][0].item())
                if step == args.num_steps - 1:
                    envs.update_attr("terminate", True) 
                
                next_obs, reward, done, info = envs.step(action.cpu().numpy())

                rewards[step] = torch.tensor(reward, device=device)

                done_tensor = torch.tensor(done, device=device, dtype=torch.bool)
                next_done = done_tensor.float()
                alive = alive & (~done_tensor)
            print("Current Actions:", cur_a)
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

            # flatten the batch
            b_obs = {
                k: np.concatenate([obs_[k] for obs_ in obs]) for k in envs.single_observation_space
            }

            # Edited
            b_logprobs = logprobs.reshape(-1, args.n_traj)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1, args.n_traj).detach()
            b_returns = returns.reshape(-1, args.n_traj)
            b_values = values.reshape(-1, args.n_traj)
            b_valid_masks = valid_masks.reshape(-1, args.n_traj).bool()

            # Optimizing the policy and value network
            # args.num_minibatches -> decide the mini-batch size: smaller num_minibatches -> larger mini-batch size (large GPU RAM)
            assert args.num_envs % args.num_minibatches == 0
            envsperbatch = args.num_envs // args.num_minibatches
            envinds = np.arange(args.num_envs)
            flatinds = np.arange(args.batch_size).reshape(args.num_steps, args.num_envs)

            clipfracs = []

            # b_advantages 存在bias
            # if args.norm_adv:
            #     b_advantages = (b_advantages - b_advantages.mean()) / (
            #         b_advantages.std() + 1e-8
            #     )
            if args.norm_adv:
                valid_adv = b_advantages[b_valid_masks]          # 只取有效样本
                adv_mean = valid_adv.mean()
                adv_std = valid_adv.std() + 1e-8

                b_advantages = (b_advantages - adv_mean) / adv_std
                # padding 的位置不参与优化，直接置 0
                b_advantages = b_advantages * b_valid_masks

            for epoch in range(args.update_epochs):
                np.random.shuffle(envinds)
                for start in range(0, args.num_envs, envsperbatch):
                    end = start + envsperbatch

                    mbenvinds = envinds[start:end]  # mini batch env id
                    mb_inds = flatinds[:, mbenvinds].ravel()  # be really careful about the index
                    r_inds = np.tile(np.arange(envsperbatch), args.num_steps)

                    cur_obs = {k: v[mbenvinds] for k, v in obs[0].items()}

                    # Get mini batch of each trajactory
                    encoder_state = agent.backbone.encode(cur_obs)

                    _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value_cached(
                        {k: v[mb_inds] for k, v in b_obs.items()},
                        b_actions.long()[mb_inds],
                        (embedding[r_inds, :] for embedding in encoder_state),
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]          # [Mb, n_traj]
                    mb_returns    = b_returns[mb_inds]
                    mb_values     = b_values[mb_inds]
                    mb_valid      = b_valid_masks[mb_inds]         # [Mb, n_traj] bool

                    # 如果极端情况：这个 mini-batch 里全是 padding，直接跳过（可选）
                    valid_count = mb_valid.sum()
                    if valid_count == 0:
                        continue
                    valid_count = valid_count.float()

                    # ---------- Policy loss（只在 valid 上平均） ----------
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - args.clip_coef, 1 + args.clip_coef
                    )
                    pg_loss_all = torch.max(pg_loss1, pg_loss2)

                    # mask 掉 padding
                    pg_loss = (pg_loss_all * mb_valid).sum() / valid_count

                    # ---------- Value loss（同样用 mask） ----------
                    newvalue = newvalue.view(-1, args.n_traj)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - mb_returns) ** 2
                        v_clipped = mb_values + torch.clamp(
                            newvalue - mb_values,
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - mb_returns) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * (v_loss_max * mb_valid).sum() / valid_count
                    else:
                        v_loss = 0.5 * (((newvalue - mb_returns) ** 2) * mb_valid).sum() / valid_count

                    # entropy 你可以直接用 mean（它是当前 policy 的探索程度，本身和 padding 没太大关系）
                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()


            ## Update Next Environment ##

            # A policy to update the customer_numbers and charging_stations_numbers and other env parameters (Curriculum Learning)
            test_num_cus = 100
            test_num_cs = 20

            if DEBUG_TEST and (update_step + 1) % 30 == 0:
                # Evaluation Process
                test_envs = SyncVectorEnv(
                    [
                        make_env(
                            args.env_id,
                            args.seed + i,
                            cfg={"env_mode": "eval", 
                                 "eval_mode": "fixed", 
                                 "config_path": config_path, 
                                 "n_traj": test_traj_num, 
                                 "num_customers": test_num_cus, 
                                 "num_charging_stations": test_num_cs},
                        )
                        for i in range(args.num_envs)
                    ]
                )
                
                # TRY NOT TO MODIFY: start the game
                agent.eval()
                test_obs = test_envs.reset()
                r = []
                record_done = np.zeros((args.num_envs, test_traj_num))
                record_cs = np.zeros((args.num_envs, test_traj_num))
                record_action = ['D']
                for step in range(0, 500):
                    # ALGO LOGIC: action logic
                    with torch.no_grad():
                        action, logits = agent(test_obs)
                    # TRY NOT TO MODIFY: execute the game and log data.
                    test_obs, _, test_done, test_info = test_envs.step(action.cpu().numpy())
                    finish_idx = (record_done == 0) & (test_done == True)
                    record_done[finish_idx] = step + 1
                    record_cs[action.cpu().numpy()> test_num_cus] += 1  # action > 100 means go to CS
                    if DEBUG_TEST:
                        if action[0][0] == 0:
                            if record_action[-1] == "D":
                                DEBUG_TEST = False
                            else:
                                record_action.append("D")
                        elif action[0][0] > test_num_cus:
                            record_action.append("R")
                        else:
                            record_action.append("C" + str(action[0][0].item()))

                    for item in test_info:
                        if "episode" in item.keys():
                            r.append(item)

                    if test_done.all():
                        break

                avg_reward = np.mean([item["episode"]["r"] for item in r])
                print("----- Evaluation Result -----")
                print("Number of Customers:", test_num_cus, "Number of Charging Stations:", test_num_cs)
                print(f"Evaluation over {len(r)} episodes: {avg_reward:.3f}, Step: {step}, Avg Done Step: {record_done.mean().item():.2f}, #CS visited: {record_cs.mean().item():.2f}")
                print('->'.join(record_action))
                print("-----------------------------")
                test_envs.close()
                return None
            envs.close()




