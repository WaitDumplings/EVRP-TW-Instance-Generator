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


def make_env(env_id, seed, cfg={}):
    def thunk():
        env = gym.make(env_id, **cfg)
        env = RecordEpisodeStatistics(env)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

def evaluate_policy(agent, eval_envs, max_iter = 1000):
    agent.eval()
    all_objs = []
    with torch.no_grad():
        obs = eval_envs.reset()
        done = False
        # 如果是 VecEnv，就要等所有 env 都 done
        cur_iter = 0
        while not done_all and cur_iter < max_iter:
            action, _, _, _, _ = agent.get_action_and_value_cached(obs)
            obs, reward, done, info = eval_envs.step(action.cpu().numpy())
            cur_iter += 1
            # 当某个 env done 时，从 info 里拿该 episode 的 objective
            for item in info:
                if "episode" in item:
                    # item["episode"]["r"]: 回报
                    # 如果你想要objective，比如 total_distance，就在 env 里加到 info["episode"]["obj"]
                    all_objs.append(item["episode"]["obj"])  

    # record finished part, finish rate
    finish_rate = 1.0
    return np.mean(all_objs), finish_rate

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
                  use_graph_token = True)

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
    for config_iter in range(config_iter_number):
        # 1. 选本轮使用的 config_path 和 n_traj
        # config_path = args.config_path[config_iter]
        # n_traj_num = args.n_traj[config_iter]
        config_path = args.config_path
        n_traj_num = args.n_traj

        # 2. 只在这里创建一套 envs（本 config 共用这一套）
        envs = SyncVectorEnv(
            [
                make_env(
                    args.env_id,
                    args.seed + i,
                    cfg={"config_path": config_path, "n_traj": n_traj_num},
                )
                for i in range(args.num_envs)
            ]
        )

        # 3. 一些全局状态（这个 config 内持续累积）
        global_step = 0
        min_obj = 1e10 * torch.ones(args.n_test, 1)
        num_updates = args.total_timesteps // args.batch_size  # 或者你想先固定成 100 也行

        # 4. 只在最开始 reset 一次环境
        next_obs = envs.reset()
        next_done = torch.zeros(args.num_envs, n_traj_num).to(device)

        # 5. 进入训练循环：每次 update = 一个 rollout（num_steps 步）
        for update in range(num_updates):
            agent.train()

            # ========== 每个 rollout 的 buffer 初始化放在这里 ==========
            obs = [None] * args.num_steps

            actions = torch.zeros(
                (args.num_steps, args.num_envs) + envs.single_action_space.shape,
                device=device,
            )
            actions_mask = torch.zeros_like(actions, dtype=torch.bool, device=device)
            actions_mask[0, :, :] = True

            logprobs = torch.zeros(
                (args.num_steps, args.num_envs, n_traj_num), device=device
            )
            rewards = torch.zeros(
                (args.num_steps, args.num_envs, n_traj_num), device=device
            )

            # 这几个只做 logging，放 CPU 就行
            obj_rewards = np.zeros((args.num_steps, args.num_envs, n_traj_num))
            serve_rewards = np.zeros((args.num_steps, args.num_envs, n_traj_num))
            rs_cur_rewards = np.zeros((args.num_steps, args.num_envs, n_traj_num))
            go_to_cus_rewards = np.zeros((args.num_steps, args.num_envs, n_traj_num))

            dones = torch.zeros(
                (args.num_steps, args.num_envs, n_traj_num), device=device
            )
            values = torch.zeros(
                (args.num_steps, args.num_envs, n_traj_num), device=device
            )
            # ==================================================

            # 这里千万不要再 reset 环境了！
            breakpoint()
            encoder_state = agent.backbone.encode(next_obs)
            r = []

            # 6. rollout：在当前 next_obs 的基础上往前跑 num_steps 步
            # for step in range(args.num_steps):
            for step in range(20):
                global_step += args.num_envs

                obs[step] = next_obs
                dones[step] = next_done

                with torch.no_grad():
                    action, logprob, _, value, _ = agent.get_action_and_value_cached(
                        next_obs, state=encoder_state
                    )
                    action = action.view(args.num_envs, n_traj_num)
                    values[step] = value.view(args.num_envs, n_traj_num)
                
                actions[step] = action
                logprobs[step] = logprob.view(args.num_envs, n_traj_num)

                if step > 0:
                    actions_mask[step] = (actions[step - 1] != 0) | (action != 0)

                # 与环境交互
                next_obs, reward, done, info = envs.step(action.cpu().numpy())
                rewards[step] = torch.tensor(reward, device=device)
                next_done = torch.tensor(done, device=device)

                for item in info:
                    if "episode" in item:
                        r.append(item)

            print("Success")
            return 

            # 7. 这里再算 avg_return / GAE / PPO update 等（跟你原来那段一样）
            # ...

        test_env_objective_value = evaluate_policy(agent, test_envs,)
        envs.close()

    # for config_iter in range(config_iter_number):
    #     envs = SyncVectorEnv(
    #         [
    #             make_env(
    #                 args.env_id,
    #                 args.seed + i,
    #                 cfg={"config_path": args.config_path, "n_traj": args.n_traj},
    #             )
    #             for i in range(args.num_envs)
    #         ]
    #     )

    #     # other initializations
    #     obs = [None] * args.num_steps
    #     actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    #     actions_mask = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device).to(torch.bool)
    #     actions_mask[0, :, :] = True

    #     logprobs = torch.zeros((args.num_steps, args.num_envs, args.n_traj)).to(device)
    #     rewards = torch.zeros((args.num_steps, args.num_envs, args.n_traj)).to(device)
    #     obj_rewards = np.zeros((args.num_steps, args.num_envs, args.n_traj))
    #     serve_rewards = np.zeros((args.num_steps, args.num_envs, args.n_traj))
    #     rs_cur_rewards = np.zeros((args.num_steps, args.num_envs, args.n_traj))
    #     go_to_cus_rewards = np.zeros((args.num_steps, args.num_envs, args.n_traj))

    #     dones = torch.zeros((args.num_steps, args.num_envs, args.n_traj)).to(device)
    #     values = torch.zeros((args.num_steps, args.num_envs, args.n_traj)).to(device)

    #     # TRY NOT TO MODIFY: start the game
    #     global_step = 0
    #     min_obj = 1e10 * torch.ones(args.n_test, 1)
    #     start_time = time.time()
    #     next_obs = envs.reset()
    #     next_done = torch.zeros(args.num_envs, args.n_traj).to(device)
    #     num_updates = args.total_timesteps // args.batch_size
    #     best_return = 1e10 # Miminize Objective Function

    #     # Train like PPO
    #     # (PPO training loop, logging, checkpoint saving, etc. will go here)

    #     next_obs = envs.reset()
    #     for update in range(1, num_updates + 1):
    #         agent.train()

    #         encoder_state = agent.backbone.encode(next_obs)
    #         next_done = torch.zeros(args.num_envs, args.n_traj).to(device)
    #         r = []

    #         for step in range(0, args.num_steps):
    #             global_step += 1 * args.num_envs
    #             obs[step] = next_obs
    #             dones[step] = next_done

    #             # ALGO LOGIC: action logic
    #             with torch.no_grad():
    #                 action, logprob, _, value, _ = agent.get_action_and_value_cached(
    #                     next_obs, state=encoder_state
    #                 )
    #                 action = action.view(args.num_envs, args.n_traj)
    #                 values[step] = value.view(args.num_envs, args.n_traj)
    #             actions[step] = action
    #             logprobs[step] = logprob.view(args.num_envs, args.n_traj)

    #             if step > 0:
    #                 actions_mask[step] = (actions[step - 1]!=0) | (action != 0)
    #             # TRY NOT TO MODIFY: execute the game and log data.
    #             # check 0:
    #             next_obs, reward, done, info = envs.step(action.cpu().numpy())
    #             rewards[step] = torch.tensor(reward).to(device)
    #             next_obs, next_done = next_obs, torch.Tensor(done).to(device)
    #             # Count rewards
    #             for k in range(args.num_envs):
    #                 obj_rewards[step,k] = envs.envs[k].env.env.env.obj_reward
    #                 serve_rewards[step,k] = envs.envs[k].env.env.env.serve_reward
    #                 rs_cur_rewards[step,k] = envs.envs[k].env.env.env.rs_reward
    #                 go_to_cus_rewards[step,k] = envs.envs[k].env.env.env.go_to_cus_reward

    #             for item in info:
    #                 if "episode" in item.keys():
    #                     r.append(item)

    #         avg_episodic_return = np.mean([rollout["episode"]["r"].mean() for rollout in r])
    #         max_episodic_return = np.mean([rollout["episode"]["r"].max() for rollout in r])
    #         avg_episodic_length = np.mean([rollout["episode"]["l"].mean() for rollout in r])
    #         print(
    #             f"[Train] global_step={global_step}\n \
    #             avg_episodic_return={avg_episodic_return}\n \
    #             max_episodic_return={max_episodic_return}\n \
    #             avg_episodic_length={avg_episodic_length}\n \
    #             avg_objective_value={np.mean(obj_rewards[actions_mask.cpu().numpy()])}\n \
    #             avg_serve_value={np.mean(serve_rewards[actions_mask.cpu().numpy()])}  \n \
    #             avg_rs_cus_value={np.mean(rs_cur_rewards[actions_mask.cpu().numpy()])}\n \
    #             avg_go_to_cus_value={np.mean(go_to_cus_rewards[actions_mask.cpu().numpy()])}\n \
    #             "
    #         )

    #         # bootstrap value if not done
    #         with torch.no_grad():
    #             next_value = agent.get_value_cached(next_obs, encoder_state).squeeze(-1)  # B x T
    #             advantages = torch.zeros_like(rewards).to(device)  # steps x B x T
    #             lastgaelam = torch.zeros(args.num_envs, args.n_traj).to(device)  # B x T
    #             for t in reversed(range(args.num_steps)):
    #                 if t == args.num_steps - 1:
    #                     nextnonterminal = 1.0 - next_done  # next_done: B
    #                     nextvalues = next_value  # B x T
    #                 else:
    #                     nextnonterminal = 1.0 - dones[t + 1]
    #                     nextvalues = values[t + 1]  # B x T
    #                 delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
    #                 advantages[t] = lastgaelam = (
    #                     delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
    #                 )
    #             returns = advantages + values

    #         # flatten the batch
    #         b_obs = {
    #             k: np.concatenate([obs_[k] for obs_ in obs]) for k in envs.single_observation_space
    #         }

    #         # Edited
    #         b_logprobs = logprobs.reshape(-1, args.n_traj)
    #         b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
    #         b_actions_mask = actions_mask.reshape((-1,) + envs.single_action_space.shape)
    #         b_advantages = advantages.reshape(-1, args.n_traj)
    #         b_returns = returns.reshape(-1, args.n_traj)
    #         b_values = values.reshape(-1, args.n_traj)

    #         # Optimizing the policy and value network
    #         assert args.num_envs % args.num_minibatches == 0
    #         envsperbatch = args.num_envs // args.num_minibatches
    #         envinds = np.arange(args.num_envs)
    #         flatinds = np.arange(args.batch_size).reshape(args.num_steps, args.num_envs)

    #         clipfracs = []

    #         for epoch in range(args.update_epochs):
    #             np.random.shuffle(envinds)
    #             for start in range(0, args.num_envs, envsperbatch):
    #                 end = start + envsperbatch
    #                 mbenvinds = envinds[start:end]  # mini batch env id
    #                 mb_inds = flatinds[:, mbenvinds].ravel()  # be really careful about the index
    #                 valid_mask = b_actions_mask[mb_inds]
    #                 r_inds = np.tile(np.arange(envsperbatch), args.num_steps)

    #                 cur_obs = {k: v[mbenvinds] for k, v in obs[0].items()}
    #                 # Get mini batch of each trajactory
                    
    #                 # embedding, avg(encoder), glimpse_k, glimpse_v, logit_k
    #                 encoder_state = agent.backbone.encode(cur_obs)

    #                 # x =  {k: v[mb_inds] for k, v in b_obs.items()}, action = b_actions, state= (embedding[r_inds, :] for embedding in encoder_state)
    #                 _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value_cached(
    #                     {k: v[mb_inds] for k, v in b_obs.items()},
    #                     b_actions.long()[mb_inds],
    #                     (embedding[r_inds, :] for embedding in encoder_state),
    #                 )
    #                 # _, newlogprob, entropy, newvalue = agent.get_action_and_value({k:v[mb_inds] for k,v in b_obs.items()}, b_actions.long()[mb_inds])
    #                 logratio = (newlogprob - b_logprobs[mb_inds])[valid_mask]
    #                 ratio = (logratio.exp())

    #                 with torch.no_grad():
    #                     # calculate approx_kl http://joschu.net/blog/kl-approx.html
    #                     old_approx_kl = (-logratio).mean()
    #                     approx_kl = ((ratio - 1) - logratio).mean()
    #                     clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

    #                 mb_advantages = b_advantages[mb_inds][valid_mask]
    #                 if args.norm_adv:
    #                     mb_advantages = (mb_advantages - mb_advantages.mean()) / (
    #                         mb_advantages.std() + 1e-8
    #                     )

    #                 # Policy loss
    #                 pg_loss1 = -mb_advantages * ratio
    #                 pg_loss2 = -mb_advantages * torch.clamp(
    #                     ratio, 1 - args.clip_coef, 1 + args.clip_coef
    #                 )
    #                 pg_loss = torch.max(pg_loss1, pg_loss2).mean()
    #                 # Value loss
    #                 newvalue = newvalue.view(-1, args.n_traj)
    #                 if args.clip_vloss:
    #                     v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
    #                     v_clipped = b_values[mb_inds] + torch.clamp(
    #                         newvalue - b_values[mb_inds],
    #                         -args.clip_coef,
    #                         args.clip_coef,
    #                     )
    #                     v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
    #                     # [valid_mask]
    #                     v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
    #                     v_loss = 0.5 * v_loss_max[valid_mask].mean()
    #                 else:
    #                     v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2)[valid_mask].mean()

    #                 entropy_loss = entropy[valid_mask].mean()
    #                 loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

    #                 optimizer.zero_grad()
    #                 loss.backward()
    #                 nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
    #                 optimizer.step()

    #             if args.target_kl is not None:
    #                 if approx_kl > args.target_kl:
    #                     break

    #         y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
    #         var_y = np.var(y_true)
    #         explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        
    #     # Close environments when training finishes
    #     envs.close()



