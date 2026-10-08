# -*- coding: utf-8 -*-
"""软约束失败原因分析（多面板诊断）：针对 R2#2。

面板（每变体一张 2x2 图）：
  (a) φ(t=2.0) 热图 + 全部零等值线 —— 视觉伪影/振荡
  (b) |∇φ| 热图 —— 平坦区（|∇φ|→0）与振荡带
  (c) 每列过零计数（开口带 |x|<=0.3, y>y0）—— 假轮廓拓扑
  (d) loss 曲线 —— 佐证「收敛了但解是错的」

数值汇总 JSON：过零列占比、最大过零数、前沿 TV（振荡度量）、轴心深度、终损。
用法：python soft_failure_analysis.py <slices_dir> <production_npz> <out_png> <out_json>
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

Y0 = -0.7
OPEN_HALF = 0.30      # 开口带半宽（孔壁 x≈±0.3）


def zero_crossings_per_col(phi, ys):
    """每列（固定 x，沿 y）过零次数。phi: (nx, ny)，y 沿 axis1。"""
    s = np.sign(phi)
    return (s[:, :-1] * s[:, 1:] < 0).sum(axis=1)   # (nx,) 沿 y 的过零数


def col_front(col_y, col_phi):
    """底部(小 y)连续负值块顶端 = 前沿；无负值→NaN。"""
    neg = col_phi < 0
    if not neg.any():
        return np.nan
    j = len(col_phi) - 1
    while j > 0 and neg[j - 1]:
        j -= 1
    p1, p2 = col_phi[j - 1], col_phi[j]
    y1, y2 = col_y[j - 1], col_y[j]
    if abs(p2 - p1) < 1e-4:
        return 0.5 * (y1 + y2)
    return y1 - p1 * (y2 - y1) / (p2 - p1)


def analyze(npz_path, label):
    d = np.load(npz_path)
    X, Y, ts, phi = d['X'], d['Y'], d['t_values'], d['phi']
    k = len(ts) - 1                       # t=t_end 切片
    P = phi[k]                            # (nx, ny)，x 沿 axis0
    gx, gy = np.gradient(P, X[:, 0], Y[0], axis=(0, 1))
    gnorm = np.sqrt(gx ** 2 + gy ** 2)

    nx = P.shape[0]
    xs = X[:, 0]
    ncr = zero_crossings_per_col(P, Y[0])          # 沿 y 过零数（每 x 列）
    in_open = np.abs(xs) <= OPEN_HALF
    ncr_open = ncr[in_open]
    front = np.full(nx, np.nan)
    for i in range(nx):
        front[i] = col_front(Y[i], P[i])
    depth = front - Y0
    depth[~in_open] = np.nan
    tv = np.nansum(np.abs(np.diff(depth[np.isfinite(depth)]))) \
        if np.isfinite(depth).sum() > 2 else np.nan
    stats = {
        'label': label,
        'spurious_col_frac': round(float((ncr_open > 1).mean()), 4),
        'max_crossings': int(ncr_open.max()),
        'mean_crossings': round(float(ncr_open.mean()), 3),
        'front_TV_nm': None if np.isnan(tv) else round(float(tv * 333.33), 1),
        'axis_depth': None if not np.isfinite(depth[nx // 2]) else round(float(depth[nx // 2]), 3),
        'depth_max': None if np.all(np.isnan(depth)) else round(float(np.nanmax(depth)), 3),
        'phi_min': round(float(P.min()), 3), 'phi_max': round(float(P.max()), 3),
        'grad_norm_min': round(float(gnorm.min()), 4),
    }
    return dict(X=X, Y=Y, P=P, gnorm=gnorm, ncr=ncr, xs=xs, stats=stats)


def loss_curve(model_pth):
    try:
        ck = torch_load(model_pth)
        return ck.get('losses', [])
    except Exception:
        return []


def torch_load(p):
    import torch
    return torch.load(p, map_location='cpu', weights_only=False)


def main(slices_dir, prod_npz, out_png, out_json):
    variants = []
    for label, fn in [('hard (lreg=0.2)', 'phi_slices_hard_lreg02.npz'),
                      ('hard lreg=0', 'phi_slices_hard_lreg0.npz'),
                      ('hard lreg=1.0', 'phi_slices_hard_lreg1.npz'),
                      ('soft lam=1', 'phi_slices_soft_lam1.npz'),
                      ('soft lam=10', 'phi_slices_soft_lam10.npz'),
                      ('soft lam=100', 'phi_slices_soft_lam100.npz'),
                      ('PRODUCTION ckpt', 'phi_slices_production.npz')]:
        f = os.path.join(slices_dir, fn)
        m = os.path.join(slices_dir, fn.replace('phi_slices_', 'model_').replace('.npz', '.pth'))
        if os.path.exists(f):
            variants.append((label, f, m if os.path.exists(m) else None))

    n = len(variants)
    fig, axes = plt.subplots(n, 3, figsize=(15, 3.4 * n), squeeze=False)
    all_stats = {}
    for r, (label, npz_f, model_f) in enumerate(variants):
        res = analyze(npz_path=npz_f, label=label)
        res['loss_final'] = None
        lc = loss_curve(model_f)
        if lc:
            res['loss_final'] = round(float(lc[-1]), 5)
        all_stats[label] = res['stats'] | {'loss_final': res['loss_final']}
        ax = axes[r]
        im = ax[0].contourf(res['X'], res['Y'], res['P'], levels=40, cmap='RdBu_r')
        cs = ax[0].contour(res['X'], res['Y'], res['P'], levels=[0], colors='k',
                           linewidths=0.9)
        ax[0].clabel(cs, inline=False, fontsize=6)
        ax[0].set_ylabel(f'{label}\ny')
        ax[0].set_title(f'phi(t=2.0)  [crossings/open-col={res["stats"]["mean_crossings"]}]',
                        fontsize=9)
        plt.colorbar(im, ax=ax[0], fraction=0.04)
        im2 = ax[1].contourf(res['X'], res['Y'], res['gnorm'], levels=40,
                             cmap='viridis')
        ax[1].contour(res['X'], res['Y'], res['P'], levels=[0], colors='r',
                      linewidths=0.8)
        ax[1].set_title('|grad phi| (red: zero contour)', fontsize=9)
        plt.colorbar(im2, ax=ax[1], fraction=0.04)
        ax[2].plot(res['xs'], res['ncr'], lw=1)
        ax[2].set_ylim(0, max(4, res['ncr'].max() + 1))
        ax[2].set_xlabel('x'); ax[2].set_ylabel('zero-crossings per column')
        ax[2].set_title('topology (1 = clean front)', fontsize=9)
        ax[2].grid(alpha=0.3)
        if lc:
            ax2 = ax[2].twiny()
            ax2.semilogy(lc, 'g-', lw=0.8, alpha=0.6)
            ax2.set_xlabel('loss (semilogy)', fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches='tight')
    with open(out_json, 'w') as f:
        json.dump(all_stats, f, indent=2)
    print(json.dumps(all_stats, indent=2))
    print('saved:', out_png)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
