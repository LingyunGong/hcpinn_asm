"""刻蚀速率物理模型：使用预计算通量场计算刻蚀速率"""

import torch
import numpy as np
from rate_precompute import PositionalFluxField


class EtchingRateModel:
    """刻蚀速率模型：速度场 = 位置通量 · 法向量"""

    def __init__(self, coefficient=0.8, flux_field=None):
        self.coefficient = coefficient
        self.flux_field = flux_field or PositionalFluxField()

    def _compute_normal_components(self, normals):
        normals_np = normals.detach().cpu().numpy() if torch.is_tensor(normals) else normals
        n_x = np.sqrt(normals_np[:, 0] ** 2 + normals_np[:, 2] ** 2)
        n_y = normals_np[:, 1]
        norm = np.sqrt(n_x ** 2 + n_y ** 2)
        mask = norm > 1e-6
        n_x[mask] = n_x[mask] / norm[mask]
        n_y[mask] = n_y[mask] / norm[mask]
        return n_x, n_y

    def __call__(self, points, normals, curvatures=None, t=None):
        """计算刻蚀速率 = 位置通量 · 法向量"""
        device = points.device if torch.is_tensor(points) else 'cpu'

        positional_flux = self.flux_field.get_positional_flux(points)  # [N, 2]
        n_x, n_y = self._compute_normal_components(normals)            # [N]

        flux_direct = positional_flux[:, 0] * n_x + positional_flux[:, 1] * n_y
        flux_direct = np.maximum(flux_direct, 0) * self.coefficient

        return torch.from_numpy(flux_direct).float().to(device).view(-1, 1)