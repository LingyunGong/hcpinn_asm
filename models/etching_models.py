"""刻蚀速率物理模型：使用预计算通量场计算刻蚀速率"""

import torch
import numpy as np
from rate_precompute import PositionalFluxField


class EtchingRateModel:
    """刻蚀速率物理模型（速度场 = 位置通量 · 法向量）"""

    def __init__(self, etching_type='integral', coefficient=0.8, flux_field=None):
        """
        Args:
            etching_type: 刻蚀类型 ('integral', 'isotropic', 'reflect', 'anisotropic')
            coefficient: 通量系数
            flux_field: 预计算的位置通量场 (PositionalFluxField 实例)
        """
        self.etching_type = etching_type
        self.coefficient = coefficient
        self.flux_field = flux_field or PositionalFluxField()

    def _compute_normal_components(self, normals):
        """计算法向量的旋转对称分量 (n_x, n_y)"""
        normals_np = normals.detach().cpu().numpy() if torch.is_tensor(normals) else normals
        n_x = np.sqrt(normals_np[:, 0] ** 2 + normals_np[:, 2] ** 2)
        n_y = normals_np[:, 1]
        norm_magnitude = np.sqrt(n_x ** 2 + n_y ** 2)
        mask = norm_magnitude > 1e-6
        n_x[mask] = n_x[mask] / norm_magnitude[mask]
        n_y[mask] = n_y[mask] / norm_magnitude[mask]
        return n_x, n_y

    def __call__(self, points, normals, curvatures=None, t=None):
        """计算刻蚀速率 = 位置通量 · 法向量"""
        if self.etching_type == 'isotropic':
            return torch.ones_like(points[:, :1]) * 0.5

        elif self.etching_type == 'integral':
            device = points.device if torch.is_tensor(points) else 'cpu'

            # 1. 位置通量 [N, 2]
            positional_flux = self.flux_field.get_positional_flux(points)

            # 2. 法向量分量 [N]
            n_x, n_y = self._compute_normal_components(normals)

            # 3. 点乘
            flux_direct = (positional_flux[:, 0] * n_x + positional_flux[:, 1] * n_y)
            flux_direct = np.maximum(flux_direct, 0) * self.coefficient

            flux_tensor = torch.from_numpy(flux_direct).float().to(device).view(-1, 1)
            return flux_tensor

        elif self.etching_type == 'reflect':
            directional_effect = 0.8
            distance = torch.sqrt(points[:, 0:1] ** 2 + points[:, 2:3] ** 2)
            r = torch.abs(distance)
            region1 = 1.0 + (20 / 81) * r ** 2
            one_minus_r = 1 - r
            region2 = (3280 / 9) * one_minus_r ** 2 - (22000 / 9) * one_minus_r ** 3
            result = torch.zeros_like(distance)
            result = torch.where(r <= 0.9, region1, result)
            result = torch.where((r > 0.9) & (r <= 1.0), region2, result)
            return directional_effect * result

        elif self.etching_type == 'anisotropic':
            directional_effect = 0.49
            distance = torch.sqrt(points[:, 0:1] ** 2 + points[:, 2:3] ** 2)
            mask = distance < 1.0
            trans = torch.where(mask, 1 - 3 * distance ** 2 + 2 * distance ** 3,
                                torch.zeros_like(distance))
            return directional_effect * trans

        else:
            raise ValueError(f"未知的刻蚀类型: {self.etching_type}")