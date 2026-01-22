import os
import time
import numpy as np
from tqdm import tqdm
from distutils.util import strtobool
import pickle
from evrptw_gen.utils.nodes_generator_scheduler import NodesGeneratorScheduler
from evrptw_gen.configs.load_config import Config
import torch.nn.functional as F

def _mean(x): return float(np.mean(x)) if len(x) else 0.0
def _max(x):  return float(np.max(x)) if len(x) else 0.0
def _min(x):  return float(np.min(x)) if len(x) else 0.0
def _p90(x):  return float(np.percentile(x, 90)) if len(x) else 0.0
def _p10(x):  return float(np.percentile(x, 10)) if len(x) else 0.0


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

from evrptw_gen.benchmarks.DRL_Solver.models.attention_model_wrapper import Agent

from evrptw_gen.benchmarks.DRL_Solver.wrappers.recordWrapper import RecordEpisodeStatistics
from evrptw_gen.benchmarks.DRL_Solver.wrappers.syncVectorEnvPomo import SyncVectorEnv
from evrptw_gen.benchmarks.DRL_Solver.utils.utils import update_lambda_fail

def make_env(env_id, seed, cfg=None):
    if cfg is None:
        cfg = {}

    def thunk():
        env = gym.make(env_id, **cfg)
        env = RecordEpisodeStatistics(env)
        env.seed(int(seed))
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

    # Print Args
    print("---------------- Training Info ---------------------")
    for key, value in vars(args).items():
        print(key, value)
    print("----------------------------------------------------")

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
                  use_graph_token = args.use_graph_token).to(device)


    backbone_params = list(agent.backbone.parameters())
    critic_params   = list(agent.critic.parameters())

    optim_backbone = optim.AdamW(
        backbone_params,
        lr=args.learning_rate,
        eps=1e-5,
        weight_decay=args.weight_decay,
    )

    optim_critic = optim.AdamW(
        critic_params,
        lr=args.critic_lr,
        eps=1e-5,
        weight_decay=args.weight_decay,
    )

    #######################
    # Algorithm defintion #
    #######################

    # 1: for config (决定instance difficulty)
    # 2: for cus / cs size (选择好此时batch的cus / cs size)
    # 3: for iter: (训练次数 + PPO)
    test_envs = None
    config_iter_number = 1  # 或更多
    num_updates = 5000
    timestamp = time.strftime("%Y%m%d-%H%M%s")
    save_dir = os.path.join(args.save_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    best_reward = float("-inf")

    # num_updates = args.total_timesteps // args.batch_size
    num_steps = args.num_steps
    num_envs = args.num_envs

    node_generater_scheduler = NodesGeneratorScheduler(min_customer_num=5, max_customer_num=5, cus_per_cs=2)
    node_generate_policy = "linear" # "linear" / "random"   
    perturb_dict = Config("./evrptw_gen/configs/perturb_config.yaml").setup_env_parameters()
    customer_numbers, charging_stations_numbers = node_generater_scheduler(policy_name=node_generate_policy)

    test_num_cus = 5
    test_num_cs = 2
    num_steps = args.num_steps
    test_max_step = num_steps

    start = time.time()
    config = Config(args.config_path)
    eval_data = pickle.load(open(args.eval_data_path, "rb"))
    num_test_envs = len(eval_data)
    eval_batch_size = args.eval_batch_size

    for config_iter in range(config_iter_number):
        # 1. 选本轮使用的 config_path 和 n_traj
        # config_path = args.config_path[config_iter]
        # n_traj_num = args.n_traj[config_iter]
        scale = 1.0 / (customer_numbers ** 0.5)
        config_path = args.config_path
        n_traj_num = args.n_traj
        test_traj_num = args.test_agent

        batch_test_env_id = np.random.choice(
            num_test_envs, size=eval_batch_size, replace=False
        )
        batch_size = len(batch_test_env_id)

        test_envs = SyncVectorEnv(
            [
                make_env(
                    args.env_id,
                    int(args.seed + i),
                    cfg={"env_mode": "eval", 
                        "config": config, 
                        "n_traj": args.test_agent,
                        "eval_data": eval_data[i]},   # New Arg
                )
                for i in batch_test_env_id
            ]
        )

        # args.batch_size = int(num_envs * num_steps)
        # args.minibatch_size = int(args.batch_size // args.num_minibatches)
        # 2. 只在这里创建一套 envs（本 config 共用这一套）
        for update_step in tqdm(range(num_updates)):
            # customer_numbers, charging_stations_numbers, num_steps, num_envs = curriculum_learning_setting(update_step)
            DEBUG_TEST = True
            t0 = time.time()
            envs = SyncVectorEnv(
                [
                    make_env(
                        args.env_id,
                        args.seed + i,
                        cfg={"config": config, 
                             "n_traj": n_traj_num, 
                             "num_customers": customer_numbers, 
                             "num_charging_stations": charging_stations_numbers,
                             "gamma": args.gamma,
                             "lambda_fail": lambda_fail,
                             "perturb_dict": perturb_dict['perturb'],},
                    )
                    for i in range(num_envs)
                ]
            )

            obs = [None] * num_steps
            actions = torch.zeros((num_steps, num_envs) + envs.single_action_space.shape, dtype=torch.int16).to(device)
            logprobs = torch.zeros((num_steps, num_envs, args.n_traj)).to(device)
            rewards = torch.zeros((num_steps, num_envs, args.n_traj)).to(device)
            dones = torch.zeros((num_steps, num_envs, args.n_traj)).to(device)
            values = torch.zeros((num_steps, num_envs, args.n_traj)).to(device)
            valid_masks = torch.zeros((num_steps, num_envs, args.n_traj), dtype=torch.bool, device=device)

            # TRY NOT TO MODIFY: start the game
            global_step = 0
            agent.train()
            next_obs = envs.reset()
            encoder_state = agent.backbone.encode(next_obs)
            next_done = torch.zeros(num_envs, args.n_traj).to(device)
            alive = torch.ones(num_envs, args.n_traj, dtype=torch.bool, device=device)

            ## Main Logic ##
            t1 = time.time()
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
            t2 = time.time()

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
            print("Customer Numbers:", customer_numbers, "Charging Stations Numbers:", charging_stations_numbers)
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

            # ===== Rollout-level diagnostics =====
            with torch.no_grad():
                rollout_valid_ratio = valid_masks.float().mean().item()  # [T,B,n_traj] -> scalar

                # rewards: [T,B,n_traj], valid_masks same shape
                valid_r = rewards[valid_masks]
                r_mean = valid_r.mean().item() if valid_r.numel() > 0 else 0.0
                r_std  = valid_r.std().item()  if valid_r.numel() > 1 else 0.0

            print(f"[RolloutDiag] valid_step={valid_step}, valid_ratio={rollout_valid_ratio:.3f}, "
                f"reward_valid_mean={r_mean:.4f}, reward_valid_std={r_std:.4f}")


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

                    delta = (rewards[t]) + args.gamma * nextvalues * nextnonterminal - values[t]
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


            # ===== Batch-level diagnostics (before PPO updates) =====
            with torch.no_grad():
                valid_adv = b_advantages[b_valid_masks]
                valid_ret = b_returns[b_valid_masks]
                valid_val = b_values[b_valid_masks]

                adv_mean = valid_adv.mean().item() if valid_adv.numel() > 0 else 0.0
                adv_std  = valid_adv.std().item()  if valid_adv.numel() > 1 else 0.0
                adv_abs  = valid_adv.abs().mean().item() if valid_adv.numel() > 0 else 0.0

                ret_mean = valid_ret.mean().item() if valid_ret.numel() > 0 else 0.0
                ret_std  = valid_ret.std().item()  if valid_ret.numel() > 1 else 0.0

                val_mean = valid_val.mean().item() if valid_val.numel() > 0 else 0.0
                val_std  = valid_val.std().item()  if valid_val.numel() > 1 else 0.0

                # Explained variance: 1 - Var(ret - val)/Var(ret)
                ev = 0.0
                if valid_ret.numel() > 1:
                    var_y = torch.var(valid_ret)
                    if var_y > 1e-12:
                        ev = (1.0 - torch.var(valid_ret - valid_val) / var_y).item()

            print(f"[BatchDiag] adv_mean={adv_mean:.4f}, adv_std={adv_std:.4f}, adv_abs_mean={adv_abs:.4f} | "
                f"ret_mean={ret_mean:.4f}, ret_std={ret_std:.4f} | "
                f"val_mean={val_mean:.4f}, val_std={val_std:.4f} | "
                f"explained_var={ev:.6f}")

            # Optimizing the policy and value network
            # args.num_minibatches -> decide the mini-batch size: smaller num_minibatches -> larger mini-batch size (large GPU RAM)
            assert num_envs % args.num_minibatches == 0
            envsperbatch = num_envs // args.num_minibatches
            envinds = np.arange(num_envs)
            flatinds = np.arange(args.batch_size).reshape(valid_step, num_envs)

            clipfracs = []

            # ===== PPO minibatch diagnostics buffers =====
            mb_kls = []
            mb_clipfracs = []
            mb_logratio_stds = []
            mb_ratio_p95s = []
            mb_valid_ratios = []
            mb_pg_losses = []
            mb_v_losses = []
            mb_entropies = []
            mb_total_losses = []
            mb_grad_norms_backbone = []
            mb_grad_norms_critic = []


            if args.norm_adv:
                valid_adv = b_advantages[b_valid_masks]          
                adv_mean = valid_adv.mean()
                adv_std = valid_adv.std() + 1e-8

                b_advantages = (b_advantages - adv_mean) / adv_std
                b_advantages = b_advantages * b_valid_masks
                
            stop_early = False

            # ---------- Gradient Accumulation Config ----------
            accum_steps = int(getattr(args, "accum_steps", 4))
            accum_steps = max(1, accum_steps)

            for epoch in range(args.update_epochs):
                if stop_early:
                    break

                np.random.shuffle(envinds)
                epoch_kls = []  # 用于 early-stop

                # ====== 初始化：每个 epoch 开始先清 grad ======
                optim_backbone.zero_grad(set_to_none=True)
                optim_critic.zero_grad(set_to_none=True)

                # 计数：累计了多少个 minibatch 的梯度
                accum_counter = 0

                # 你原来 mb_grad_norms_* 是按 minibatch append；累积后更合理的是按 “真实 step” append
                step_grad_norms_backbone = []
                step_grad_norms_critic = []

                for start in range(0, num_envs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]

                    mb_inds = flatinds[:, mbenvinds].ravel()
                    r_inds = np.tile(np.arange(envsperbatch), valid_step)

                    # 重新 encode 该 env 的初始 embedding（你原逻辑）
                    cur_obs = {k: v[mbenvinds] for k, v in obs[0].items()}
                    encoder_state = agent.backbone.encode(cur_obs)

                    mb_valid = b_valid_masks[mb_inds]  # [Mb, n_traj] bool

                    # forward: 取新 logprob / entropy / value
                    _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value_cached(
                        {k: v[mb_inds] for k, v in b_obs.items()},
                        b_actions.long()[mb_inds],
                        (embedding[r_inds, :] for embedding in encoder_state),
                    )

                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    # ===== 统计 KL / clipfrac（只在 valid 上）=====
                    with torch.no_grad():
                        v = mb_valid
                        if v.any():
                            lr = logratio[v]
                            rr = ratio[v]
                            approx_kl = ((rr - 1) - lr).mean().item()
                            epoch_kls.append(approx_kl)

                            mb_kls.append(approx_kl)
                            mb_clipfracs.append(((rr - 1.0).abs() > args.clip_coef).float().mean().item())
                            mb_logratio_stds.append(lr.std().item() if lr.numel() > 1 else 0.0)
                            mb_ratio_p95s.append(torch.quantile(rr, 0.95).item())
                        else:
                            epoch_kls.append(0.0)
                            mb_kls.append(0.0)
                            mb_clipfracs.append(0.0)
                            mb_logratio_stds.append(0.0)
                            mb_ratio_p95s.append(0.0)

                    # ===== Loss（mask valid）=====
                    mb_advantages = b_advantages[mb_inds]
                    mb_returns    = b_returns[mb_inds]
                    mb_values     = b_values[mb_inds]

                    valid_count = mb_valid.sum()
                    mb_valid_ratios.append((valid_count.float() / mb_valid.numel()).item())

                    if valid_count == 0:
                        # 注意：这里不能 step，但也不应破坏累积逻辑
                        # 如果你希望“空 minibatch 也计入 accum_counter”，通常不推荐（会稀释有效梯度）
                        continue

                    valid_count = valid_count.float()

                    # policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss  = (torch.max(pg_loss1, pg_loss2) * mb_valid).sum() / valid_count

                    # value loss (Huber + optional clip)
                    newvalue = newvalue.view(-1, args.n_traj)

                    huber_unclipped = F.smooth_l1_loss(newvalue, mb_returns, reduction="none", beta=1.0)
                    if args.clip_vloss:
                        v_clipped = mb_values + torch.clamp(newvalue - mb_values, -args.clip_coef, args.clip_coef)
                        huber_clipped = F.smooth_l1_loss(v_clipped, mb_returns, reduction="none", beta=1.0)
                        v_loss = 0.5 * (torch.max(huber_unclipped, huber_clipped) * mb_valid).sum() / valid_count
                    else:
                        v_loss = 0.5 * (huber_unclipped * mb_valid).sum() / valid_count

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    mb_pg_losses.append(pg_loss.item())
                    mb_v_losses.append(v_loss.item())
                    mb_entropies.append(entropy_loss.item())
                    mb_total_losses.append(loss.item())

                    # ---------- Gradient Accumulation Backward ----------
                    # 保持等效学习率：把 loss 按 accum_steps 缩放
                    (loss / accum_steps).backward()
                    accum_counter += 1

                    # 只在每个 update 的第一个“有效 minibatch”打一次 GradSplit
                    if epoch == 0 and accum_counter == 1:
                        def grad_norm(params):
                            tot = 0.0
                            for p in params:
                                if p.grad is None:
                                    continue
                                g = p.grad.detach()
                                tot += g.pow(2).sum().item()
                            return tot ** 0.5

                        gn_backbone = grad_norm(agent.backbone.parameters())
                        gn_critic   = grad_norm(agent.critic.parameters())
                        print(f"[GradSplit pre-clip] backbone={gn_backbone:.6f}, critic={gn_critic:.6f}")
                        print(f"[Loss] pg_loss={pg_loss:.6f}, entropy_loss={args.ent_coef * entropy_loss:.6f}, value_loss={v_loss * scale:.6f}")

                    # ---------- Decide whether to step ----------
                    is_last_minibatch = (start + envsperbatch) >= num_envs
                    do_step = (accum_counter % accum_steps == 0) or is_last_minibatch

                    if do_step:
                        # clip on accumulated grads
                        pre_backbone = nn.utils.clip_grad_norm_(agent.backbone.parameters(), args.max_grad_norm_backbone)
                        pre_critic   = nn.utils.clip_grad_norm_(agent.critic.parameters(),   args.max_grad_norm_critic)

                        step_grad_norms_backbone.append(float(pre_backbone))
                        step_grad_norms_critic.append(float(pre_critic))

                        optim_backbone.step()
                        optim_critic.step()

                        optim_backbone.zero_grad(set_to_none=True)
                        optim_critic.zero_grad(set_to_none=True)

                # 把“真实 step”的 grad norm 统计并入你原来的数组（这样 PPODiag 仍然能打印）
                # 注意：这里 append 的数量会比原先少（因为一个 step 对应 accum_steps 个 minibatch）
                mb_grad_norms_backbone.extend(step_grad_norms_backbone)
                mb_grad_norms_critic.extend(step_grad_norms_critic)

                # ===== epoch 级 early-stop（target_kl）=====
                if args.target_kl is not None and len(epoch_kls) > 0:
                    mean_kl = float(np.mean(epoch_kls))
                    if mean_kl > args.target_kl:
                        stop_early = True
                        print(f"[EarlyStop] epoch={epoch} mean_kl={mean_kl:.5f} > target_kl={args.target_kl:.5f}")

                if len(epoch_kls) > 0:
                    print(f"[EpochKL] epoch={epoch} mean_kl={float(np.mean(epoch_kls)):.5f} max_kl={float(np.max(epoch_kls)):.5f}")
            t3 = time.time()
            print("[Time] "
                  f"Env Create Time: {t1 - t0:.4f}s |"
                  f"Rollout Collect Time: {t2 - t1:.4f}s |"
                  f"Loss & Update :{t3 - t2:.4f}")
            print(
                "[PPODiag] "
                f"kl_mean={_mean(mb_kls):.4f}, kl_p90={_p90(mb_kls):.4f}, kl_max={_max(mb_kls):.4f} | "
                f"clipfrac_mean={_mean(mb_clipfracs):.3f}, logratio_std_mean={_mean(mb_logratio_stds):.3f}, ratio_p95_mean={_mean(mb_ratio_p95s):.3f} | "
                f"valid_ratio_mb_mean={_mean(mb_valid_ratios):.3f}, p10={_p10(mb_valid_ratios):.3f}, min={_min(mb_valid_ratios):.3f} | "
                f"pg_loss_mean={_mean(mb_pg_losses):.4f}, v_loss_mean={_mean(mb_v_losses):.4f}, ent_mean={_mean(mb_entropies):.4f} | "
                f"gn_bb_mean={_mean(mb_grad_norms_backbone):.3f}, gn_bb_max={_max(mb_grad_norms_backbone):.3f} | "
                f"gn_v_mean={_mean(mb_grad_norms_critic):.3f}, gn_v_max={_max(mb_grad_norms_critic):.3f}"
            )

            
            # 建议：在训练主循环里，每隔比如 10 个 update 打一次
            if update_step % 5 == 0:
                print(f"[AttnScore] scale={agent.backbone.decoder.pointer.scale.item():.4f}, C={agent.backbone.decoder.pointer.C.item():.1f}")
                # Update curriculm learning setting
            print()
            # Update Next Environment ##

            # A policy to update the customer_numbers and charging_stations_numbers and other env parameters (Curriculum Learning)
            if (update_step + 1) % 10 == 0:
                t_eval_start = time.time()
                # Evaluation Process
                # TRY NOT TO MODIFY: start the game

                # del obs, actions, logprobs, rewards, dones, values, advantages, returns  # 举例
                # torch.cuda.empty_cache()

                agent.eval()

                # batch_test_env_id = np.random.choice(
                #     num_test_envs, size=eval_batch_size, replace=False
                # )
                # batch_size = len(batch_test_env_id)

                # test_envs = SyncVectorEnv(
                #     [
                #         make_env(
                #             args.env_id,
                #             int(args.seed + i),
                #             cfg={"env_mode": "eval", 
                #                 "config": config, 
                #                 "n_traj": args.test_agent,
                #                 "eval_data": eval_data[i]},   # New Arg
                #         )
                #         for i in batch_test_env_id
                #     ]
                # )

                record_info = []
                record_action = ['D']
                record_done = np.zeros((batch_size, test_traj_num))
                record_cs = np.zeros((batch_size, test_traj_num))
                record_cus = np.zeros((batch_size, test_traj_num))
                test_obs = test_envs.reset()
                for step in range(0, test_max_step):
                    # ALGO LOGIC: action logic
                    with torch.no_grad():
                        action, logits = agent(test_obs)
                    action = action.to("cpu").numpy()
                    # TRY NOT TO MODIFY: execute the game and log data.
                    test_obs, _, test_done, test_info = test_envs.step(action)
                    finish_idx = (record_done == 0) & (test_done == True)
                    record_done[finish_idx] = step + 1
                    record_cs[action> test_num_cus] += 1  # action > 100 means go to CS
                    record_cus[(action <= test_num_cus) & (action > 0)] += 1

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
                            record_info.append(item)

                    if test_done.all():
                        break
                    test_envs.close()

                avg_reward = np.mean([item["episode"]["r"] for item in record_info])

                print("----- Evaluation Result -----")
                print("Number of Customers:", test_num_cus, "Number of Charging Stations:", test_num_cs)
                print(f"Evaluation over {len(record_info)} episodes: {avg_reward:.3f}, Step: {step}, Avg Done Step: {record_done.mean().item():.2f}, #CS visited: {record_cs.mean().item():.2f}")
                print('->'.join(record_action))
                print("Eval cost : {:.4f}s".format(time.time() - t_eval_start))
                breakpoint()
                if avg_reward > best_reward and len(record_info) == len(batch_test_env_id):
                    best_reward = avg_reward
                    torch.save(agent.state_dict(), os.path.join(save_dir, "best_model.pth"))
                torch.save(agent.state_dict(), os.path.join(save_dir, "cur_model.pth"))
                print("-----------------------------")
                customer_numbers, charging_stations_numbers = node_generater_scheduler(policy_name=node_generate_policy)
                # del record_info, record_done_total, record_cs_total  # 这些是 numpy，主要是 CPU 内存
                # torch.cuda.empty_cache()
            # envs.close()




