"""
测试 WarmupScheduler 是否按预期工作：
1. Warmup 阶段：lr 从 0 线性上升到 base_lr
2. Warmup 结束后：CosineAnnealingWarmRestarts 从 base_lr 开始衰减
"""

import torch
import torch.optim as optim
import wandb
import matplotlib.pyplot as plt

from utils.schedulers.warmup_scheduler import WarmupScheduler

# 初始化 wandb（offline 模式）
wandb.init(mode="offline")


def test_warmup_scheduler():
    # 测试参数（对应 AVE Res 的实际配置）
    base_lr = 0.001           # optimizer.learning_rate
    n_warmup_steps = 780      # warm_up_steps
    total_steps = 2000        # 模拟总步数

    # 初始化模型和优化器
    model = torch.nn.Linear(10, 2)
    optimizer = optim.Adam(model.parameters(), lr=base_lr)

    # 构建 after_scheduler（与 Loader.py 保持一致）
    after_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer=optimizer,
        T_0=n_warmup_steps,   # 第一个周期长度 = warmup 步数（按 step 计）
        T_mult=1,
    )

    # 包装 WarmupScheduler
    scheduler = WarmupScheduler(
        optimizer=optimizer,
        base_lr=base_lr,
        n_warmup_steps=n_warmup_steps,
        after_scheduler=after_scheduler,
    )

    lr_history = []
    for step in range(total_steps):
        scheduler.step()   # 不传参数，与实际训练代码一致
        current_lr = optimizer.param_groups[0]["lr"]
        lr_history.append(current_lr)

    # ---- 验证 warmup 阶段 ----
    print("=" * 50)
    print("Warmup 阶段验证（前 10 步）")
    print("=" * 50)
    for step in range(10):
        expected_lr = base_lr * ((step + 1) / n_warmup_steps)
        actual_lr = lr_history[step]
        match = "✓" if abs(actual_lr - expected_lr) < 1e-10 else "✗"
        print(f"  Step {step+1:4d}: actual={actual_lr:.10f}  expected={expected_lr:.10f}  {match}")

    print()
    print("=" * 50)
    print("Warmup 结束点验证")
    print("=" * 50)
    print(f"  Step {n_warmup_steps}: lr = {lr_history[n_warmup_steps-1]:.10f}")
    print(f"  Expected at end of warmup: {base_lr:.10f}")
    assert abs(lr_history[n_warmup_steps-1] - base_lr) < 1e-10, "Warmup 结束点 lr 不等于 base_lr！"
    print("  ✓ Warmup 结束点 lr 等于 base_lr")

    # ---- 验证 warmup 结束后余弦衰减 ----
    print()
    print("=" * 50)
    print("余弦衰减阶段验证（warmup 后前 5 步）")
    print("=" * 50)
    after_scheduler_state = after_scheduler.state_dict()
    print(f"  after_scheduler T_cur (at step {n_warmup_steps}): {after_scheduler_state.get('T_cur', 'N/A')}")
    print(f"  after_scheduler T_i: {after_scheduler_state.get('T_i', 'N/A')}")

    # CosineAnnealingWarmRestarts 的衰减公式：
    # lr = eta_min + (base_lr - eta_min) * (1 + cos(pi * T_cur / T_i)) / 2
    # 其中 eta_min=0，所以 lr = base_lr * (1 + cos(pi * T_cur / T_i)) / 2
    # warmup 结束后，T_cur 从 0 开始，lr 从 base_lr 逐渐减小
    for step_offset in range(5):
        step = n_warmup_steps + step_offset
        actual_lr = lr_history[step]
        t_cur = step_offset
        T_i = n_warmup_steps   # 与 Loader.py 中的 T_0 一致
        expected_lr = base_lr * (1 + torch.cos(torch.tensor(torch.pi * t_cur / T_i)).item()) / 2
        match = "✓" if abs(actual_lr - expected_lr) < 1e-6 else "✗"
        print(f"  Step {step}: actual={actual_lr:.10f}  expected={expected_lr:.10f}  {match}")

    print()
    print("=" * 50)
    print("整体 LR 曲线概览")
    print("=" * 50)
    # 打印关键节点
    checkpoints = [1, 100, 500, 779, 780, 781, 800, 1000, 1500, 1999]
    for c in checkpoints:
        if c < total_steps:
            print(f"  Step {c:4d}: lr = {lr_history[c]:.10f}")

    # ---- 绘图 ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # 左图：完整曲线
    axes[0].plot(lr_history, linewidth=0.8)
    axes[0].axvline(x=n_warmup_steps, color="red", linestyle="--", label=f"Warmup ends at step {n_warmup_steps}")
    axes[0].axhline(y=base_lr, color="green", linestyle=":", label=f"base_lr={base_lr}")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Learning Rate")
    axes[0].set_title("Full LR Schedule")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 右图：Warmup 阶段放大
    warmup_lrs = lr_history[:n_warmup_steps]
    axes[1].plot(warmup_lrs, linewidth=1.0, color="tab:blue")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Learning Rate")
    axes[1].set_title(f"Warmup Phase (steps 1-{n_warmup_steps})")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    print("测试完成！")


if __name__ == "__main__":
    test_warmup_scheduler()
