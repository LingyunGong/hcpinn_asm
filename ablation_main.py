# -*- coding: utf-8 -*-
"""JIMS 修稿实验（main.py PINN 管线版）：硬约束 vs 软约束消融 + 训练计时

基座 = 本仓库根的生产代码（SpaceTimeSIREN + EtchingRateModel('integral')
+ LevelSetLoss），架构零改动：
  * hard: 原样 SpaceTimeSIREN（forward 内解析硬约束 φ_H=λ(t)I+η(t)N·B）
  * soft: 同一网络去掉输出变换（子类仅重写 forward 两行，层/初始化完全相同）
          + IC 罚（t=0 处 |φ-I|²）+ BC 罚（掩膜冻结区 |φ-I|²，即硬约束 B=0
          区域的软对应物）；λ_ic=λ_bc 可调（默认 1.0，扫 10/100 做鲁棒性）
两变体同一初始化种子、同一速率场、同一 PDE 配点采样器、同一优化器/调度器。

用法：
  python ablation_main.py --variant hard --epochs 1500 --points 4000 --outdir out_hard
  python ablation_main.py --variant soft --soft-lam 10  --epochs 1500 --points 4000 --outdir out_soft10
  python ablation_main.py --variant hard,soft --outdir out_ablation   # 顺序跑两个
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

import platform

REPO = os.environ.get('LEVELSET_REPO', os.path.dirname(os.path.abspath(__file__)))
if os.path.isdir(REPO):
    sys.path.insert(0, REPO)

# 无 h5py 的环境（如部分集群）下 trainer.py 仍可导入——它仅在时序数据模式才真正使用 h5py，
# 本实验不用时序数据 → 提供 stub 通过模块级 import。
try:
    import h5py  # noqa: F401
except ImportError:
    import types
    _stub = types.ModuleType('h5py')
    _stub.File = None
    sys.modules['h5py'] = _stub

from models.neural_siren import SpaceTimeSIREN            # noqa: E402
from models.etching_models import EtchingRateModel        # noqa: E402
from training.loss_functions import LevelSetLoss          # noqa: E402
from training.trainer import EtchingTrainer               # noqa: E402
from config.default_config import TrainingConfig          # noqa: E402


# ---------------------------------------------------------------------------
# 软约束网络：仅重写 forward（去掉输出变换），网络层与初始化完全继承
# ---------------------------------------------------------------------------
class SoftSIREN(SpaceTimeSIREN):
    def forward(self, x):
        features = self.network(x)
        return self.final_layer(features)


# ---------------------------------------------------------------------------
# 软约束损失：在原 LevelSetLoss 之上加 IC 罚 + 掩膜冻结区 BC 罚
# ---------------------------------------------------------------------------
class SoftLevelSetLoss(LevelSetLoss):
    def __init__(self, *a, lam_ic=1.0, lam_bc=1.0, **kw):
        super().__init__(*a, **kw)
        self.lam_ic = lam_ic
        self.lam_bc = lam_bc
        self.extra = {}

    def forward(self, model, samples, etching_rate_model):
        total = super().forward(model, samples, etching_rate_model)
        # IC 罚：t=0 处 φ 应等于初始场 I(x)=y-y0
        if 'ic_points' in self.extra:
            p = self.extra['ic_points']              # (N,3) 已 requires_grad
            t0 = torch.zeros(p.shape[0], 1, device=p.device)
            inp = torch.cat([p, t0], dim=1)
            phi0 = model(inp)
            target = p[:, 1:2] - p.new_tensor(-0.7)
            ic_loss = torch.mean((phi0 - target) ** 2)
            total = total + self.lam_ic * ic_loss
            self.last_ic = ic_loss.item()
        # BC 罚：掩膜冻结区（B≈0，即 f<=-0.1）任意时刻 φ 应回到初始场
        if 'bc_points' in self.extra:
            p = self.extra['bc_points']
            tb = self.extra['bc_t']                  # (N,1) 随机时刻
            inp = torch.cat([p, tb], dim=1)
            phi_b = model(inp)
            target = p[:, 1:2] - p.new_tensor(-0.7)
            bc_loss = torch.mean((phi_b - target) ** 2)
            total = total + self.lam_bc * bc_loss
            self.last_bc = bc_loss.item()
        return total


def sample_domain_points(n, device, y_range=(-1.0, 1.7), x_half=0.5):
    """域内均匀采样 (x,y,z)。"""
    xyz = torch.rand(n, 3, device=device)
    xyz[:, 0] = xyz[:, 0] * 2 * x_half - x_half
    xyz[:, 1] = xyz[:, 1] * (y_range[1] - y_range[0]) + y_range[0]
    xyz[:, 2] = xyz[:, 2] * 2 * x_half - x_half
    return xyz.requires_grad_(True)


def sample_mask_frozen_points(n, device, alpha=1.0, r=0.25, y0=-0.7,
                              y_range=(-1.0, 1.7), x_half=0.5, f_margin=-0.1):
    """掩膜冻结区采样：f = y-y0-alpha(x^2+z^2-R^2) <= f_margin（B≈0 区），拒绝采样。"""
    out = []
    while sum(o.shape[0] for o in out) < n:
        m = max(n // 2, 256)
        p = sample_domain_points(m, device, y_range, x_half).detach()
        f = p[:, 1] - y0 - alpha * (p[:, 0] ** 2 + p[:, 2] ** 2 - r ** 2)
        p = p[f <= f_margin]
        out.append(p)
    return torch.cat(out, 0)[:n].requires_grad_(True)


def run_variant(variant, args, etching_model):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = TrainingConfig()
    cfg.lambda_data = 0.0
    cfg.lambda_pde = args.lambda_pde
    cfg.lambda_eikonal = args.lambda_reg
    if getattr(args, 'hidden_dim', None):
        cfg.hidden_dim = args.hidden_dim
    if getattr(args, 'hidden_layers', None):
        cfg.hidden_layers = args.hidden_layers
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(args.outdir, exist_ok=True)
    log_path = os.path.join(args.outdir, 'train.log')
    logf = open(log_path, 'w')

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line)
        logf.write(line + '\n')
        logf.flush()

    log(f"variant={variant} soft_lam={args.soft_lam} device={device} "
        f"epochs={args.epochs} points={args.points}")

    if variant == 'hard':
        model = SpaceTimeSIREN(hidden_layers=cfg.hidden_layers,
                               hidden_dim=cfg.hidden_dim,
                               r=cfg.radius, alpha=cfg.alpha).to(device)
        loss_fn = LevelSetLoss(lambda_data=0.0, lambda_pde=cfg.lambda_pde,
                               lambda_eikonal=cfg.lambda_eikonal)
    else:
        model = SoftSIREN(hidden_layers=cfg.hidden_layers,
                          hidden_dim=cfg.hidden_dim,
                          r=cfg.radius, alpha=cfg.alpha).to(device)
        loss_fn = SoftLevelSetLoss(lambda_data=0.0, lambda_pde=cfg.lambda_pde,
                                   lambda_eikonal=cfg.lambda_eikonal,
                                   lam_ic=args.soft_lam, lam_bc=args.soft_lam)

    # PDE 配点采样器（复用生产代码的拒绝采样 + 截断指数时间分布）
    sampler = EtchingTrainer(model, loss_fn, etching_model,
                             r=cfg.radius, alpha=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=500, factor=0.5)

    n_pde = args.points
    n_aux = max(1, args.points // 4)
    losses, pde_hist, ic_hist, bc_hist = [], [], [], []
    epoch_times = []
    t0 = time.perf_counter()
    for epoch in range(args.epochs):
        te = time.perf_counter()
        st_points, st_t = sampler.sample_pde_points(n_pde, time_interval=(0, 1.8))
        st_points, st_t = st_points.to(device), st_t.to(device)  # 采样器产 CPU 张量
        batch = {'pde_points': st_points, 'pde_t': st_t}
        if variant == 'soft':
            ic_p = sample_domain_points(n_aux, device)
            bc_p = sample_mask_frozen_points(n_aux, device)
            bc_t = torch.rand(n_aux, 1, device=device) * 1.8
            loss_fn.extra = {'ic_points': ic_p, 'bc_points': bc_p, 'bc_t': bc_t}
        optimizer.zero_grad()
        loss = loss_fn(model, batch, etching_model)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(loss)
        epoch_times.append(time.perf_counter() - te)
        losses.append(loss.item())
        if variant == 'soft':
            ic_hist.append(getattr(loss_fn, 'last_ic', float('nan')))
            bc_hist.append(getattr(loss_fn, 'last_bc', float('nan')))
        if epoch % 100 == 0 or epoch == args.epochs - 1:
            log(f"[{variant}] epoch {epoch:04d}/{args.epochs} "
                f"total={loss.item():.4e} "
                f"ic={ic_hist[-1]:.3e} bc={bc_hist[-1]:.3e}"
                if variant == 'soft' else
                f"[{variant}] epoch {epoch:04d}/{args.epochs} total={loss.item():.4e}")
    wall = time.perf_counter() - t0
    log(f"[{variant}] DONE wall={wall:.2f}s per_epoch="
        f"{np.mean(epoch_times[10:])*1e3:.1f}ms")

    # 保存模型与损失
    torch.save({'model_state_dict': model.state_dict(),
                'variant': variant, 'soft_lam': args.soft_lam,
                'losses': losses, 'ic_hist': ic_hist, 'bc_hist': bc_hist,
                'epoch_times': epoch_times},
               os.path.join(args.outdir, f'model_{variant}.pth'))

    # 多时刻 φ 切片（z=0 平面）——失败模式分析用：伪影/振荡是视觉现象
    model.eval()
    nx, ny = 400, 500
    xs = np.linspace(-0.5, 0.5, nx)
    ys = np.linspace(-1.0, 2.2, ny)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    t_values = [0.5, 1.0, 1.5, cfg.t_end]
    phi_stack = np.zeros((len(t_values), nx, ny))
    with torch.no_grad():
        for k, tv in enumerate(t_values):
            pts = torch.tensor(np.stack([X.ravel(), Y.ravel(),
                                         np.zeros_like(X.ravel())], 1),
                               dtype=torch.float32, device=device)
            tt = torch.full((pts.shape[0], 1), float(tv), device=device)
            phi_stack[k] = model(torch.cat([pts, tt], 1)).cpu().numpy().reshape(nx, ny)
    np.savez(os.path.join(args.outdir, f'phi_slices_{variant}.npz'),
             X=X, Y=Y, t_values=np.array(t_values), phi=phi_stack)

    summary = {'variant': variant, 'soft_lam': args.soft_lam if variant == 'soft' else None,
               'device': device, 'torch': torch.__version__,
               'epochs': args.epochs, 'points': args.points,
               'lambda_pde': cfg.lambda_pde, 'lambda_reg': cfg.lambda_eikonal,
               'wall_s': round(wall, 2),
               'per_epoch_ms': round(float(np.mean(epoch_times[10:])) * 1e3, 2),
               'final_loss': losses[-1], 'host': os.uname().nodename if os.name != 'nt'
               else platform_node()}
    with open(os.path.join(args.outdir, 'timing.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    logf.close()
    return summary


def platform_node():
    import platform
    return platform.node() + '/' + platform.processor()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', type=str, default='hard,soft')
    ap.add_argument('--soft-lam', type=float, default=1.0)
    ap.add_argument('--epochs', type=int, default=2000,
                    help='生产 checkpoint 口径 2000')
    ap.add_argument('--points', type=int, default=5625,
                    help='生产口径：batch 10000 × 9/16 = 5625 PDE 配点')
    ap.add_argument('--lambda-pde', type=float, default=1.1,
                    help='生产配置口径 1.1')
    ap.add_argument('--lambda-reg', type=float, default=0.2)
    ap.add_argument('--device', type=str, default=None)
    ap.add_argument('--outdir', type=str, default='results_main_ablation')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--hidden-dim', type=int, default=None,
                    help='override cfg.hidden_dim (e.g. 64 = paper Table 1 config)')
    ap.add_argument('--hidden-layers', type=int, default=None)
    args = ap.parse_args()

    etching_model = EtchingRateModel(etching_type='integral')
    base_out = args.outdir
    summaries = []
    for variant in args.variant.split(','):
        variant = variant.strip()
        tag = variant if (variant == 'hard' or args.soft_lam == 1.0) \
            else f'{variant}_lam{args.soft_lam:g}'
        args.outdir = os.path.join(base_out, tag)
        s = run_variant(variant, args, etching_model)
        s['tag'] = tag
        summaries.append(s)
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
