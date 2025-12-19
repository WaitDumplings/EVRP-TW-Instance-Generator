import os
import time
from tqdm import tqdm
from distutils.util import strtobool
import pickle

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
from evrptw_gen.benchmarks.utils.utils import update_lambda_fail

def make_env(env_id, seed, cfg=None):
    if cfg is None:
        cfg = {}

    def thunk():
        env = gym.make(env_id, **cfg)
        env = RecordEpisodeStatistics(env)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

def curriculum_learning_setting(update_step):
    if update_step < 50:
        customer_numbers = 5
        charging_stations_numbers = 2
        num_steps = 30
        num_envs = 512
    elif update_step < 200:
        customer_numbers = 20
        charging_stations_numbers = 5
        num_steps = 60
        num_envs = 256
        # num-steps
    elif update_step < 500:
        customer_numbers = 50
        charging_stations_numbers = 10
        num_steps = 100
        num_envs = 128
    else:
        customer_numbers = 100
        charging_stations_numbers = 20
        num_steps = 160
        num_envs = 96
    return customer_numbers, charging_stations_numbers, num_steps, num_envs


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

    # ===== 新增：Lagrangian penalty 的参数 =====
    lambda_fail = args.lambda_fail_init      # 比如 5.0

    #########################
    ### Model Definition ####
    #########################
    device = f"cuda:{args.cuda_id}" if torch.cuda.is_available() else "cpu"
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
    eval_method = args.eval_method
    test_envs = None
    config_iter_number = 1  # 或更多
    num_updates = 10000
    # num_updates = args.total_timesteps // args.batch_size
    num_steps = args.num_steps
    num_envs = args.num_envs

    for config_iter in range(config_iter_number):
        # 1. 选本轮使用的 config_path 和 n_traj
        # config_path = args.config_path[config_iter]
        # n_traj_num = args.n_traj[config_iter]
        config_path = args.config_path
        n_traj_num = args.n_traj
        test_traj_num = args.test_agent


        customer_numbers = 100
        charging_stations_numbers = 20
        num_steps = 170
        num_envs = 96
        # args.batch_size = int(num_envs * num_steps)
        # args.minibatch_size = int(args.batch_size // args.num_minibatches)
        # 2. 只在这里创建一套 envs（本 config 共用这一套）
        for update_step in tqdm(range(num_updates)):
            # customer_numbers, charging_stations_numbers, num_steps, num_envs = curriculum_learning_setting(update_step)
            DEBUG_TEST = True
            envs = SyncVectorEnv(
                [
                    make_env(
                        args.env_id,
                        args.seed + i,
                        cfg={"config_path": config_path, 
                             "n_traj": n_traj_num, 
                             "num_customers": customer_numbers, 
                             "num_charging_stations": charging_stations_numbers,
                             "gamma": args.gamma,
                             "lambda_fail": lambda_fail},
                    )
                    for i in range(num_envs)
                ]
            )
            obs = [None] * num_steps
            actions = torch.zeros((num_steps, num_envs) + envs.single_action_space.shape).to(device)
            logprobs = torch.zeros((num_steps, num_envs, args.n_traj)).to(device)
            rewards = torch.zeros((num_steps, num_envs, args.n_traj)).to(device)
            dones = torch.zeros((num_steps, num_envs, args.n_traj)).to(device)
            values = torch.zeros((num_steps, num_envs, args.n_traj)).to(device)
            valid_masks = torch.zeros((num_steps, num_envs, args.n_traj)).to(device)

            # TRY NOT TO MODIFY: start the game
            global_step = 0
            agent.train()
            next_obs = envs.reset()
            encoder_state = agent.backbone.encode(next_obs)
            next_done = torch.zeros(num_envs, args.n_traj).to(device)
            alive = torch.ones(num_envs, args.n_traj, dtype=torch.bool, device=device)

            r = []

            ## Main Logic ##
            import time
            t0 = time.time()
            for step in range(num_steps):
                global_step += 1 * num_envs
                obs[step] = next_obs
                dones[step] = next_done

                valid_masks[step] = alive

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    # if step == 0:
                    #     print_label = True
                    # else:
                    #     print_label = False
                    print_label = False
                    action, logprob, _, value, _ = agent.get_action_and_value_cached(
                        next_obs, state=encoder_state, print_probs=print_label
                    )

                    action = action.view(num_envs, args.n_traj)
                    values[step] = value.view(num_envs, args.n_traj)
                actions[step] = action
                logprobs[step] = logprob.view(num_envs, args.n_traj)

                if step == num_steps - 1:
                    envs.update_attr("terminate", True) 

                next_obs, reward, done, info = envs.step(action.cpu().numpy())
                rewards[step] = torch.tensor(reward, device=device)

                done_tensor = torch.tensor(done, device=device, dtype=torch.bool)
                next_done = done_tensor.float()
                alive = alive & (~done_tensor)
                valid_step = step + 1
                if done.all():
                    break
            
            # ReFormat
            obs = obs[:valid_step]
            actions = actions[:valid_step]
            logprobs = logprobs[:valid_step]
            rewards = rewards[:valid_step]
            dones = dones[:valid_step]
            values = values[:valid_step]
            valid_masks = valid_masks[:valid_step]
            t1 = time.time()
            print("{} steps cost {:.4f} s".format(step, t1 - t0))

            args.batch_size = int(num_envs * valid_step)
            args.minibatch_size = int(args.batch_size // args.num_minibatches)

            visu_actions = actions.reshape((valid_step, -1)).cpu().numpy().copy()
            visu_actions[visu_actions == 0] = customer_numbers + 1      # depot 标成 > customer_numbers
            visu_actions[visu_actions < 1 + customer_numbers] = 1       # 所有 customers -> 1
            visu_actions[visu_actions >= 1 + customer_numbers] = 0      # depot + RS -> 0

            cus_count_per_traj = visu_actions.sum(axis=0)    # [num_envs * n_traj]
            finish_flags = (cus_count_per_traj == customer_numbers)
            success_rate = float(finish_flags.mean())

            print("------------------ Training Record ------------------")
            print(f"Epoch: {update_step}/{num_updates}") 
            print(f"Avg Customer Visits: {cus_count_per_traj.mean():.2f}") 
            print(f"Finish Rate: {finish_flags.sum()}/{finish_flags.size} = {success_rate:.3f}")
            print(f"Current lambda_fail (before): {lambda_fail:.3f}")

            lambda_fail = update_lambda_fail(
                lambda_fail=lambda_fail,
                success_rate=success_rate,
                target_success=args.target_success,
                lambda_max=args.lambda_max,
                lr_up=args.lambda_lr_up,
                lr_down=args.lambda_lr_down,
                tolerance=args.lambda_tolerance,
            )

            print(f"Updated lambda_fail: {lambda_fail:.3f}")
            print("----------------------------------------------------")
            


            ## PPO Logic ##
            # bootstrap value if not done
            with torch.no_grad():
                next_value = agent.get_value_cached(next_obs, encoder_state).squeeze(-1)  # B x T
                advantages = torch.zeros_like(rewards).to(device)  # steps x B x T
                lastgaelam = torch.zeros(num_envs, args.n_traj).to(device)  # B x T
                for t in reversed(range(valid_step)):
                    if t == valid_step - 1:
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
            assert num_envs % args.num_minibatches == 0
            envsperbatch = num_envs // args.num_minibatches
            envinds = np.arange(num_envs)
            flatinds = np.arange(args.batch_size).reshape(valid_step, num_envs)

            clipfracs = []

            if args.norm_adv:
                valid_adv = b_advantages[b_valid_masks]          
                adv_mean = valid_adv.mean()
                adv_std = valid_adv.std() + 1e-8

                b_advantages = (b_advantages - adv_mean) / adv_std
                b_advantages = b_advantages * b_valid_masks

            for epoch in range(args.update_epochs):
                np.random.shuffle(envinds)
                for start in range(0, num_envs, envsperbatch):
                    end = start + envsperbatch

                    mbenvinds = envinds[start:end]  # mini batch env id
                    mb_inds = flatinds[:, mbenvinds].ravel()  # be really careful about the index
                    r_inds = np.tile(np.arange(envsperbatch), valid_step)

                    cur_obs = {k: v[mbenvinds] for k, v in obs[0].items()}
                    # next_obs
                    # Get mini batch of each trajactory
  
                    # (node_embed, graph_context, glimpse_key, glimpse_val, logit_key)
                    encoder_state= agent.backbone.encode(cur_obs)

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

                    valid_count = mb_valid.sum()
                    if valid_count == 0:
                        continue
                    valid_count = valid_count.float()

                    # ---------- Policy loss w/ mask ----------
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - args.clip_coef, 1 + args.clip_coef
                    )
                    pg_loss_all = torch.max(pg_loss1, pg_loss2)

                    pg_loss = (pg_loss_all * mb_valid).sum() / valid_count

                    # ---------- Value loss w/ mask ----------
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

            if (update_step + 1) % 30 == 0:
                # Evaluation Process
                if eval_method == "generator":
                    num_test_envs = num_envs
                    test_envs = SyncVectorEnv(
                        [
                            make_env(
                                args.env_id,
                                args.seed + i,
                                cfg={"env_mode": "eval", 
                                    "eval_mode": eval_method,   # generator / solomon_txt
                                    "config_path": config_path, 
                                    "n_traj": test_traj_num, 
                                    "num_customers": test_num_cus, 
                                    "num_charging_stations": test_num_cs},
                            )
                            for i in range(num_test_envs)
                        ]
                    )
                else:
                    if test_envs is None:
                        num_test_envs = len(pickle.load(open(args.eval_data_path, "rb")))
                        test_envs = SyncVectorEnv(
                            [
                                make_env(
                                    args.env_id,
                                    args.seed + i,
                                    cfg={"env_mode": "eval", 
                                        "eval_mode": eval_method,   # fixed / solomon_txt
                                        "config_path": config_path, 
                                        "n_traj": test_traj_num,
                                        "ins_index": i,
                                        "eval_data_path": args.eval_data_path},   # New Arg
                                )
                                for i in range(num_test_envs)
                            ]
                        )
                
                # TRY NOT TO MODIFY: start the game
                agent.eval()
                test_obs = test_envs.reset()
                r = []
                record_done = np.zeros((num_test_envs, test_traj_num))
                record_cs = np.zeros((num_test_envs, test_traj_num))
                record_action = ['D']
                for step in range(0, 300):
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
                if eval_method == "generator":
                    test_envs.close()
            if eval_method == "solomon_txt":
                test_envs.close()
            envs.close()




