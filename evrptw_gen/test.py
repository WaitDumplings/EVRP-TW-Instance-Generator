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
