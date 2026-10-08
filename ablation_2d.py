# -*- coding: utf-8 -*-
"""JIMS 修稿实验 A+B：2D 硬约束 vs 软约束 PINN 消融（同一配置、精确计时）

回应审稿意见：
  R2#1  2D 加速比未含训练时间 → 本脚本精确测量训练墙钟时间（论文硬件=i7-14650HX CPU
        本机跑；H100 GPU 服务器跑作 GPU/CPU 对比）
  R2#2  消融只报 PDE loss 未报几何精度 → 训练软约束变体到收敛，保存 φ 场供 ε_Γ 评估
  R4#1  （无关）  R4#3 训练成本细节 → 本脚本输出 per-epoch 时间曲线

设计：
  * 基础代码 = "2D neural levelset/main_2D.py"（2D 沟槽生产代码，自包含）
  * 两变体共用同一初始化种子、同一通量场、同一配置（论文 Table 1 口径）：
      4×64 网络、1500 epochs、lr 1e-4、StepLR(0.9, 200)、配点 4000、λ_pde=1.0、λ_reg=0.2
  * hard: φ = (y-y0) + t·N(x,y,t)·B(x)   [原 forward，IC/BC 解析保证，无罚项]
  * soft: φ = N(x,y,t)                   [标准 PINN，IC/BC 罚项 λ=1.0]
  * 保存：loss 历史、墙钟时间、最终模型、t=1.0 的 φ 网格（npz，供轮廓提取/ε_Γ）

用法：
  python ablation_2d.py --epochs 1500 --points 4000 --outdir results_ablation
  python ablation_2d.py --epochs 3 --points 500          # 冒烟测试
"""

import argparse
import importlib.util
import json
import os
import platform
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN_2D_CANDIDATES = [
    os.path.join(HERE, 'main_2D.py'),                                   # 同目录（仓库布局）
    os.path.join(os.path.dirname(HERE), '2D neural levelset', 'main_2D.py'),  # 本仓布局
]
MAIN_2D_PATH = next(p for p in MAIN_2D_CANDIDATES if os.path.exists(p))


def load_main_2d():
    """目录名含空格，用 importlib 按路径加载。"""
    spec = importlib.util.spec_from_file_location('main_2d', MAIN_2D_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['main_2d'] = mod
    spec.loader.exec_module(mod)
    patch_serial_flux(mod)
    return mod


def patch_serial_flux(m2d):
    """串行通量场预计算补丁。

    原实现用 ProcessPoolExecutor：Windows spawn 模式下 importlib 动态加载的
    模块无法在子进程 re-import（BrokenProcessPool）。改为进程内串行，逻辑
    与原版逐位一致（同 batch 划分、同填充顺序），仅去掉进程池。
    """
    def _precompute_flux_serial(self):
        self.logger.info("开始预计算通量场（串行补丁）...")
        start_time = time.time()
        x = np.linspace(self.x_range[0], self.x_range[1], self.grid_size)
        y = np.linspace(self.y_range[0], self.y_range[1], self.grid_size)
        self.X, self.Y = np.meshgrid(x, y)
        points = np.stack([self.X.ravel(), self.Y.ravel()], axis=-1)
        n_points = len(points)
        n_batches = max(1, n_points // self.config.batch_size)
        batches = np.array_split(points, n_batches)
        self.Fx = np.zeros((self.grid_size, self.grid_size))
        self.Fy = np.zeros((self.grid_size, self.grid_size))
        idx = 0
        for i, batch in enumerate(batches):
            fx, fy = m2d.VectorizedFluxCalculator.calculate_flux_batch(batch)
            self.Fx.ravel()[idx:idx + len(fx)] = fx
            self.Fy.ravel()[idx:idx + len(fy)] = fy
            idx += len(fx)
        self.Fx_tensor = torch.FloatTensor(self.Fx).to(self.config.device)
        self.Fy_tensor = torch.FloatTensor(self.Fy).to(self.config.device)
        mag = np.sqrt(self.Fx ** 2 + self.Fy ** 2)
        self.logger.info(f"通量场统计 - min {np.min(mag):.4f} max {np.max(mag):.4f} "
                         f"mean {np.mean(mag):.4f} | {time.time()-start_time:.1f}s")
    m2d.EfficientFluxField._precompute_flux_parallel = _precompute_flux_serial


def make_soft_net_class(m2d):
    """软约束网络：去掉硬约束变换，返回原始网络输出（标准 PINN）。"""

    class _SoftLevelSetNet(m2d.SimplifiedLevelSetNet):
        def forward(self, x):
            hidden = self.initial_layer(x)
            hidden = self.hidden_layers(hidden)
            return self.output_layer(hidden)

    return _SoftLevelSetNet


def make_soft_trainer_class(m2d):
    """软约束训练器：启用 IC 罚 + BC 罚（目标修正为掩膜侧 φ 冻结在初始值）。"""

    class _SoftTrainer(m2d.ImprovedPINNTrainer):
        def compute_boundary_loss(self, boundary_points):
            phi = self.model(boundary_points)
            # 修正目标：掩膜接触区 φ 应回到初始场 φ0 = y - y_interface
            # （原实现 target=0 只在 y=y_interface 的采样带上碰巧等价）
            target = boundary_points[:, 1:2] - self.config.y_interface
            return torch.mean((phi - target) ** 2)

        def compute_losses(self, collocation_points, boundary_points, initial_points):
            losses = {}
            pde_loss, grad_norm = self.compute_pde_loss(collocation_points)
            losses['pde'] = pde_loss
            losses['grad_norm'] = grad_norm.mean()
            losses['reg'] = self.compute_regularization_loss(grad_norm)
            losses['bc'] = self.compute_boundary_loss(boundary_points)
            losses['ic'] = self.compute_initial_loss(initial_points)
            losses['total'] = (self.config.lambda_pde * losses['pde']
                               + self.config.lambda_reg * losses['reg']
                               + self.config.lambda_ic * losses['ic']
                               + self.config.lambda_bc * losses['bc'])
            return losses

    return _SoftTrainer


def build_config(m2d, args):
    cfg = m2d.Config()
    if args.paper_config:
        # 论文 Table 1 声称的配置（实测 2D 生产代码用此配置训练不充分，见
        # results_gpu 里 hard 剖面退化；消融改用生产默认配置，两变体仍完全一致）
        cfg.hidden_layers = 4
        cfg.hidden_dim = 64
        cfg.n_epochs = args.epochs
        cfg.lr = 1e-4
        cfg.n_points = args.points
        cfg.lambda_pde = 1.0
        cfg.lambda_reg = 0.2
    else:
        # main_2D.py 生产默认配置（3x256, lr 1e-3, 3000 epochs, 1000 配点）
        cfg.n_epochs = args.epochs
        cfg.n_points = args.points
    cfg.lambda_ic = 0.0          # hard 变体不用
    cfg.lambda_bc = 0.0          # hard 变体不用
    cfg.checkpoint_interval = 10 ** 9   # 中途不存 checkpoint（时间对比不受 IO 干扰）
    if args.device:
        cfg.device = args.device
    return cfg


def run_variant(m2d, cfg, variant, flux_field, outdir, args, soft_lam=1.0):
    """训练一个变体，返回计时与损失摘要。"""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(outdir, exist_ok=True)
    logger = m2d.setup_logging(outdir)
    logger.info(f"=== variant={variant} soft_lam={soft_lam} device={cfg.device} "
                f"epochs={cfg.n_epochs} points={cfg.n_points} ===")

    if variant == 'hard':
        model = m2d.SimplifiedLevelSetNet(cfg)
        trainer = m2d.ImprovedPINNTrainer(model, flux_field, cfg)
    else:
        model = make_soft_net_class(m2d)(cfg)
        trainer = make_soft_trainer_class(m2d)(model, flux_field, cfg)
        cfg.lambda_ic = soft_lam   # 软约束：IC/BC 罚项启用（λ 可调做鲁棒性检查）
        cfg.lambda_bc = soft_lam

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.9)

    t0 = time.perf_counter()
    epoch_times = []
    for epoch in range(cfg.n_epochs):
        te = time.perf_counter()
        loss_dict = trainer.train_epoch(optimizer)
        epoch_times.append(time.perf_counter() - te)
        scheduler.step()
        for k, v in loss_dict.items():
            if k in trainer.loss_history:
                trainer.loss_history[k].append(v)
        if epoch % 100 == 0 or epoch == cfg.n_epochs - 1:
            logger.info(f"[{variant}] epoch {epoch:04d} total={loss_dict['total']:.4e} "
                        f"pde={loss_dict['pde']:.4e} ic={loss_dict['ic']:.4e} "
                        f"bc={loss_dict.get('bc', float('nan')):.4e} "
                        f"lr={optimizer.param_groups[0]['lr']:.2e}")
    wall = time.perf_counter() - t0

    # 保存模型与损失
    torch.save({'model_state_dict': model.state_dict(),
                'config': cfg.__dict__,
                'loss_history': trainer.loss_history},
               os.path.join(outdir, f'model_{variant}.pth'))

    # 最终时刻 φ 场（供轮廓提取 / ε_Γ 评估）
    evaluator = m2d.LevelSetEvaluator(model, cfg, flux_field)
    res = evaluator.evaluate_grid(x_range=(-1.5, 1.5), y_range=(-1.5, 0.5),
                                  t_values=[1.0], grid_size=400)
    d = res[1.0]
    np.savez(os.path.join(outdir, f'phi_final_{variant}.npz'),
             X=d['X'], Y=d['Y'], phi=d['phi'])

    summary = {
        'variant': variant,
        'soft_lam': soft_lam if variant == 'soft' else None,
        'device': str(cfg.device),
        'torch': torch.__version__,
        'epochs': cfg.n_epochs,
        'n_points': cfg.n_points,
        'hidden': f"{cfg.hidden_layers}x{cfg.hidden_dim}",
        'wall_s': round(wall, 3),
        'per_epoch_ms': round(float(np.mean(epoch_times[10:] if len(epoch_times) > 10
                                              else epoch_times)) * 1e3, 3),  # 去预热
        'final_losses': {k: float(v[-1]) for k, v in trainer.loss_history.items()
                         if len(v)},
        'host': platform.node(),
        'cpu': platform.processor(),
    }
    with open(os.path.join(outdir, f'timing_{variant}.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    np.save(os.path.join(outdir, f'epoch_times_{variant}.npy'), np.array(epoch_times))
    logger.info(f"[{variant}] DONE wall={wall:.2f}s "
                f"per_epoch={summary['per_epoch_ms']}ms "
                f"final_total={summary['final_losses'].get('total')}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=1500)
    ap.add_argument('--points', type=int, default=4000)
    ap.add_argument('--device', type=str, default=None,
                    help="'cpu' 或 'cuda'；缺省自动")
    ap.add_argument('--outdir', type=str, default='results_ablation')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--variants', type=str, default='hard,soft')
    ap.add_argument('--soft-lam', type=float, default=1.0,
                    help='soft 变体的 IC/BC 罚权重（鲁棒性检查用）')
    ap.add_argument('--paper-config', action='store_true',
                    help='用论文 Table 1 配置（缺省用 main_2D 生产默认配置）')
    args = ap.parse_args()

    m2d = load_main_2d()
    cfg = build_config(m2d, args)

    print(f"device: {cfg.device} | torch {torch.__version__} | "
          f"cuda_available={torch.cuda.is_available()}")

    os.makedirs(args.outdir, exist_ok=True)
    m2d.setup_logging(args.outdir)

    # 通量场只算一次，两变体共享
    print("预计算通量场 ...")
    tf = time.perf_counter()
    flux_field = m2d.EfficientFluxField(cfg)
    flux_prep_s = time.perf_counter() - tf
    print(f"通量场就绪 ({flux_prep_s:.1f}s)")

    summaries = []
    for variant in args.variants.split(','):
        variant = variant.strip()
        tag = variant if (variant == 'hard' or args.soft_lam == 1.0) \
            else f'{variant}_lam{args.soft_lam:g}'
        outdir = os.path.join(args.outdir, tag)
        s = run_variant(m2d, cfg, variant, flux_field, outdir, args,
                        soft_lam=args.soft_lam)
        s['tag'] = tag
        summaries.append(s)

    allf = os.path.join(args.outdir, 'ablation_summary.json')
    with open(allf, 'w') as f:
        json.dump({'flux_prep_s': round(flux_prep_s, 2),
                   'runs': summaries}, f, indent=2)
    print("\n=== 汇总 ===")
    for s in summaries:
        print(f"{s['variant']:>5}: wall={s['wall_s']:>8.1f}s  "
              f"per_epoch={s['per_epoch_ms']:>7.1f}ms  "
              f"final_total={s['final_losses'].get('total'):.4e}")
    print(f"已写入 {allf}")


if __name__ == '__main__':
    main()
