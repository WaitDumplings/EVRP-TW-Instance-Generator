import numpy as np

def update_lambda_fail(
    lambda_fail: float,
    success_rate: float,
    target_success: float,
    lambda_max: float,
    lr_up: float,
    lr_down: float,
    tolerance: float,
) -> float:
    """
    Lagrangian update for failure penalty lambda_fail.

    逻辑：
    - success_rate < target_success - tolerance  → 失败太多 → 提高 lambda_fail
    - success_rate > target_success + tolerance  → 成功太多 → 降低 lambda_fail
    - 其余情况 → 不动（保持在安全带内）
    """
    lambda_fail = float(lambda_fail)
    success_rate = float(success_rate)
    target_success = float(target_success)

    # 失败率视角（更接近你原来的变量命名）
    fail_rate = 1.0 - success_rate
    target_fail = 1.0 - target_success

    gap = fail_rate - target_fail  # >0: fail too much; <0: fail too little

    # 在 [target_fail - tolerance, target_fail + tolerance] 内不更新
    if -tolerance <= gap <= tolerance:
        return float(np.clip(lambda_fail, 0.0, lambda_max))

    if gap > 0.0:
        # 失败多于目标（成功率太低）→ 增大 lambda
        # gap 越大，涨得越多
        lambda_fail = lambda_fail + lr_up * gap
    else:
        # gap < 0.0：失败少于目标（成功率高于 target）→ 减小 lambda
        lambda_fail = lambda_fail - lr_down * (-gap)

    # 投影到 [0, lambda_max]
    lambda_fail = float(np.clip(lambda_fail, 0.0, lambda_max))
    return lambda_fail
