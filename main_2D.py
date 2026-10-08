import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib

matplotlib.use('Agg')  # 设置非交互式后端
import matplotlib.pyplot as plt
import os
import json
import time
from datetime import datetime
from typing import Tuple, Optional, Dict, List, Any
from dataclasses import dataclass
import logging
from scipy.integrate import quad
from concurrent.futures import ProcessPoolExecutor
import warnings

warnings.filterwarnings('ignore')


# ==================== 配置类 ====================
@dataclass
class Config:
    """配置参数"""
    # 几何参数
    Lx: float = 0.92
    Ly: float = 3.0
    y_interface: float = -0.7
    mask_width: float = 0.16
    mask_height: float = 1.0
    mask_ratio: float = 0.1

    # 通量场参数
    flux_center: float = np.pi / 2
    flux_sigma1: float = 0.035
    flux_sigma2: float = 0.14
    flux_ratio: float = 0.07

    # 训练参数
    n_epochs: int = 3000
    lr: float = 1e-3
    n_points: int = 1000
    t_max: float = 1.0
    lambda_pde: float = 1.0
    lambda_reg: float = 0.1
    lambda_bc: float = 0.0  # 设置为0
    lambda_ic: float = 0.1

    # 网络参数
    hidden_layers: int = 3
    hidden_dim: int = 256
    use_symmetry: bool = True
    use_fourier: bool = False  # 暂时关闭Fourier特征以简化

    # 通量场计算参数
    grid_size: int = 150
    batch_size: int = 1000
    num_workers: int = 4

    # 其他
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_interval: int = 500
    adaptive_sampling: bool = True


# ==================== 日志设置 ====================
def setup_logging(results_dir: str):
    """设置日志"""
    log_path = os.path.join(results_dir, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


# ==================== 向量化通量计算 ====================
class VectorizedFluxCalculator:
    """向量化通量计算器"""

    @staticmethod
    def g(theta, center=np.pi / 2, sigma1=0.035, sigma2=0.14, ratio=0.07):
        """向量化通量函数"""
        return np.exp(-(theta - center) ** 2 / (2 * sigma1 ** 2))

    @staticmethod
    def g2(theta, center=0, sigma=0.035):
        """向量化反射通量函数"""
        return np.exp(-(theta - center) ** 2 / (2 * sigma ** 2))

    @staticmethod
    def calculate_flux_batch(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """批量计算通量（修复反射通量）"""
        x_vals = points[:, 0]
        y_vals = points[:, 1]
        y0 = -0.7
        h = 1.0
        upper_bound = y0 - h

        # 计算积分上下限
        denominator_lower = x_vals + 1.0
        denominator_upper = x_vals - 1.0

        # 向量化计算积分限
        lower_limits = np.arctan((y_vals - upper_bound) / (denominator_lower + 1e-12))

        # 处理分母接近0的情况
        mask_small = np.abs(denominator_upper) < 1e-10
        upper_limits = np.where(
            mask_small,
            np.pi / 2,
            np.where(
                denominator_upper > 0,
                -np.arctan((y_vals - y0) / (denominator_upper + 1e-12)),
                np.pi + np.arctan((y_vals - upper_bound) / (denominator_upper + 1e-12))
            )
        )

        # 交换上下限如果lower > upper
        swap_mask = lower_limits > upper_limits
        temp = lower_limits[swap_mask].copy()
        lower_limits[swap_mask] = upper_limits[swap_mask]
        upper_limits[swap_mask] = temp

        # 数值积分
        n_points = len(points)
        flux_x = np.zeros(n_points)
        flux_y = np.zeros(n_points)

        for i in range(n_points):
            if lower_limits[i] >= upper_limits[i]:
                continue

            try:
                # 使用向量化积分
                def integrand_x(theta):
                    return VectorizedFluxCalculator.g(theta) * np.cos(theta)

                def integrand_y(theta):
                    return VectorizedFluxCalculator.g(theta) * np.sin(theta)

                integral_x, _ = quad(integrand_x, lower_limits[i], upper_limits[i])
                integral_y, _ = quad(integrand_y, lower_limits[i], upper_limits[i])

                # 添加反射通量（确保反射通量被正确计算）
                reflect_x, reflect_y = VectorizedFluxCalculator.calculate_reflect_flux_single(
                    x_vals[i], y_vals[i]
                )

                # 重要：反射通量乘以系数并加上直接通量
                flux_x[i] = (integral_x + reflect_x ) * 4 * np.pi
                flux_y[i] = (integral_y + reflect_y ) * 4 * np.pi

            except Exception as e:
                # 记录错误但不中断
                pass

        return flux_x, flux_y

    @staticmethod
    def calculate_reflect_flux_single(x_val: float, y_val: float) -> Tuple[float, float]:
        """计算单个点的反射通量（修复版本）"""
        y0 = -0.7
        h = 1.0
        upper_bound = y0 - h

        # 计算积分上下限
        lower_limit = max(y_val - y0, 0)
        upper_limit = y_val - upper_bound

        if lower_limit >= upper_limit or np.isclose(lower_limit, upper_limit, atol=1e-12):
            return 0.0, 0.0

        r = 1.0
        eps = 1e-8

        def reflect_integrand(y):
            rl = np.sqrt((r + x_val) ** 2 + y ** 2 + eps)
            rr = np.sqrt((r - x_val) ** 2 + y ** 2 + eps)

            # 防止除零
            y_rl_ratio = np.clip(y / rl, -1.0, 1.0)
            y_rr_ratio = np.clip(y / rr, -1.0, 1.0)

            theta_1 = np.arccos(y_rl_ratio)
            theta_2 = np.arccos(y_rr_ratio)

            # 使用g2函数计算反射通量
            left_x = VectorizedFluxCalculator.g2(theta_1) / (rl + eps) * np.sin(theta_1)
            right_x = -VectorizedFluxCalculator.g2(theta_2) / (rr + eps) * np.sin(theta_2)
            left_y = VectorizedFluxCalculator.g2(theta_1) / (rl + eps) * np.cos(theta_1)
            right_y = VectorizedFluxCalculator.g2(theta_2) / (rr + eps) * np.cos(theta_2)

            # 根据x_val位置决定使用哪些分量
            if -1 <= x_val <= 1:
                return left_x + right_x, left_y + right_y
            elif x_val < -1:
                return right_x, right_y  # 只考虑右源点
            else:  # x_val > 1
                return left_x, left_y  # 只考虑左源点

        try:
            # 数值积分
            reflect_integrand_x, reflect_integrand_y = reflect_integrand
            integral_reflect_x= quad(reflect_integrand_x, lower_limit, upper_limit)[0]
            integral_reflect_y= quad(reflect_integrand_y, lower_limit, upper_limit)[0]
            return integral_reflect_x, integral_reflect_y

        except Exception as e:
            # 如果积分失败，返回0
            return 0.0, 0.0


# ==================== 高效通量场 ====================
class EfficientFluxField:
    """高效通量场，使用并行计算和GPU加速"""

    def __init__(self, config: Config):
        self.config = config
        self.x_range = (-1.5, 1.5)
        self.y_range = (-1.5, 1.0)
        self.grid_size = config.grid_size

        self.logger = logging.getLogger(__name__)
        self._precompute_flux_parallel()

    def _precompute_flux_parallel(self):
        """并行预计算通量场"""
        self.logger.info("开始并行预计算通量场...")
        start_time = time.time()

        # 创建网格
        x = np.linspace(self.x_range[0], self.x_range[1], self.grid_size)
        y = np.linspace(self.y_range[0], self.y_range[1], self.grid_size)
        self.X, self.Y = np.meshgrid(x, y)

        points = np.stack([self.X.ravel(), self.Y.ravel()], axis=-1)
        n_points = len(points)

        # 分批并行计算
        n_batches = max(1, n_points // self.config.batch_size)
        batches = np.array_split(points, n_batches)

        self.Fx = np.zeros((self.grid_size, self.grid_size))
        self.Fy = np.zeros((self.grid_size, self.grid_size))

        # 使用进程池并行计算
        with ProcessPoolExecutor(max_workers=self.config.num_workers) as executor:
            futures = []
            for batch in batches:
                future = executor.submit(VectorizedFluxCalculator.calculate_flux_batch, batch)
                futures.append(future)

            # 收集结果
            idx = 0
            for i, future in enumerate(futures):
                flux_x_batch, flux_y_batch = future.result()
                batch_size = len(flux_x_batch)

                start_idx = idx
                end_idx = idx + batch_size

                self.Fx.ravel()[start_idx:end_idx] = flux_x_batch
                self.Fy.ravel()[start_idx:end_idx] = flux_y_batch
                idx += batch_size

                if (i + 1) % 10 == 0:
                    self.logger.info(f"进度: {min(idx, n_points)}/{n_points}")

        # 转换为PyTorch张量并移到设备
        self.Fx_tensor = torch.FloatTensor(self.Fx).to(self.config.device)
        self.Fy_tensor = torch.FloatTensor(self.Fy).to(self.config.device)

        # 计算通量统计信息
        flux_magnitude = np.sqrt(self.Fx ** 2 + self.Fy ** 2)
        self.logger.info(f"通量场统计 - 最小: {np.min(flux_magnitude):.4f}, "
                         f"最大: {np.max(flux_magnitude):.4f}, "
                         f"平均: {np.mean(flux_magnitude):.4f}")

        self.logger.info(f"通量场计算完成，耗时: {time.time() - start_time:.2f}秒")

    def get_flux_gpu(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """在GPU上双线性插值获取通量值"""
        # 归一化到[0, 1]
        x_norm = (x - self.x_range[0]) / (self.x_range[1] - self.x_range[0])
        y_norm = (y - self.y_range[0]) / (self.y_range[1] - self.y_range[0])

        # 缩放到网格索引
        x_idx = x_norm * (self.grid_size - 1)
        y_idx = y_norm * (self.grid_size - 1)

        # 双线性插值
        x0 = torch.floor(x_idx).long().clamp(0, self.grid_size - 2)
        x1 = x0 + 1
        y0 = torch.floor(y_idx).long().clamp(0, self.grid_size - 2)
        y1 = y0 + 1

        # 计算权重
        x_frac = x_idx - x0.float()
        y_frac = y_idx - y0.float()

        # 收集四个角点的值
        f00_x = self.Fx_tensor[y0, x0]
        f01_x = self.Fx_tensor[y0, x1]
        f10_x = self.Fx_tensor[y1, x0]
        f11_x = self.Fx_tensor[y1, x1]

        f00_y = self.Fy_tensor[y0, x0]
        f01_y = self.Fy_tensor[y0, x1]
        f10_y = self.Fy_tensor[y1, x0]
        f11_y = self.Fy_tensor[y1, x1]

        # 双线性插值
        flux_x = (1 - x_frac) * (1 - y_frac) * f00_x + \
                 x_frac * (1 - y_frac) * f01_x + \
                 (1 - x_frac) * y_frac * f10_x + \
                 x_frac * y_frac * f11_x

        flux_y = (1 - x_frac) * (1 - y_frac) * f00_y + \
                 x_frac * (1 - y_frac) * f01_y + \
                 (1 - x_frac) * y_frac * f10_y + \
                 x_frac * y_frac * f11_y

        return flux_x.unsqueeze(-1), flux_y.unsqueeze(-1)


# ==================== SIREN正弦激活层 ====================
class SineLayer(nn.Module):
    """SIREN正弦激活层"""

    def __init__(self, in_features, out_features, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)

        # SIREN特殊初始化
        with torch.no_grad():
            bound = np.sqrt(6 / in_features) / omega_0
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


# ==================== 简化的网络结构 ====================
class SimplifiedLevelSetNet(nn.Module):
    """简化的水平集网络，修复维度问题"""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # 输入维度: (x, y, t)
        self.input_dim = 3

        # 第一层：SIREN正弦层
        self.initial_layer = SineLayer(self.input_dim, config.hidden_dim, omega_0=30)

        # 隐藏层：使用SIREN
        layers = []
        for i in range(config.hidden_layers - 1):
            layers.append(SineLayer(config.hidden_dim, config.hidden_dim, omega_0=30))

            # 添加dropout（除了最后一层）
            if i < config.hidden_layers - 2:
                layers.append(nn.Dropout(0.05))

        self.hidden_layers = nn.Sequential(*layers)

        # 输出层
        self.output_layer = nn.Linear(config.hidden_dim, 1)

        # 输出层特殊初始化
        with torch.no_grad():
            bound = np.sqrt(6 / config.hidden_dim) / 30
            self.output_layer.weight.uniform_(-bound, bound)
            self.output_layer.bias.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 分离坐标
        x_coord = x[:, 0:1]
        y_coord = x[:, 1:2]
        t_coord = x[:, 2:3]

        # 网络前向传播
        hidden = self.initial_layer(x)
        hidden = self.hidden_layers(hidden)
        output = self.output_layer(hidden)

        # 应用物理约束
        # 初始条件: φ(x,y,0) = y - y_interface
        initial_condition = y_coord - self.config.y_interface

        # 边界条件:
        r = torch.abs(x_coord)
        boundary_mask = r < 1.02

        # 硬约束: φ = (y - y_interface) + t * network_output * boundary_function
        # 边界函数在边界附近衰减
        boundary_func = torch.where(boundary_mask, 1 - 0.6 * r ** 8, torch.zeros_like(r))

        # 当t=0时，确保满足初始条件
        final_output = initial_condition + t_coord * output * boundary_func

        return final_output

    def compute_gradients(self, x: torch.Tensor, phi: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """高效计算梯度"""
        # 确保x需要梯度
        x.requires_grad_(True)

        # 计算梯度
        gradients = torch.autograd.grad(
            outputs=phi,
            inputs=x,
            grad_outputs=torch.ones_like(phi),
            create_graph=True,
            retain_graph=True
        )[0]

        phi_x = gradients[:, 0:1]
        phi_y = gradients[:, 1:2]
        phi_t = gradients[:, 2:3]

        return phi_x, phi_y, phi_t


# ==================== 自适应采样器 ====================
class AdaptiveSampler:
    """自适应重要性采样"""

    def __init__(self, config: Config):
        self.config = config
        self.device = config.device

        # 采样权重
        self.boundary_weight = 0.0
        self.interface_weight = 0.7
        self.uniform_weight = 0.3

    def sample_points(self, n_points: int, t_max: float = 1.0) -> Tuple[torch.Tensor, ...]:
        """采样训练点"""
        n_boundary = int(n_points * self.boundary_weight)
        n_interface = int(n_points * self.interface_weight)
        n_uniform = n_points - n_boundary - n_interface

        # 1. 边界区域采样
        # 采样x在(1.2, 1.5)和(-1.5, -1.2)范围内的点
        # 使用随机数决定每个点的x符号
        x_sign = torch.where(torch.rand(n_boundary, 1, device=self.device) > 0.5, 1.0, -1.0)
        x_abs = 1.2 + torch.rand(n_boundary, 1, device=self.device) * 0.3  # 1.2到1.5
        x_boundary = x_sign * x_abs
        # y固定在y_interface
        y_boundary = torch.full_like(x_boundary, self.config.y_interface)
        t_boundary = torch.rand(n_boundary, 1, device=self.device) * t_max

        # 2. 界面附近采样
        x_interface = torch.rand(n_interface, 1, device=self.device) * 2.2 - 1.1
        y_interface_val = self.config.y_interface + 0.6 * (torch.rand(n_interface, 1, device=self.device) - 0.5)
        t_interface = torch.rand(n_interface, 1, device=self.device) * t_max

        # 3. 均匀采样
        x_uniform = torch.rand(n_uniform, 1, device=self.device) * 2.2 - 1.1
        y_uniform = torch.rand(n_uniform, 1, device=self.device) * 2 - 1
        t_uniform = torch.rand(n_uniform, 1, device=self.device) * t_max

        # 合并
        x = torch.cat([x_boundary, x_interface, x_uniform], dim=0)
        y = torch.cat([y_boundary, y_interface_val, y_uniform], dim=0)
        t = torch.cat([t_boundary, t_interface, t_uniform], dim=0)

        # 创建组合张量
        points = torch.cat([x, y, t], dim=1)
        points.requires_grad_(True)

        return points

    def sample_collocation_points(self, n_points: int, t_max: float = 1.0) -> torch.Tensor:
        """采样PDE残差点"""
        return self.sample_points(n_points, t_max)

    def sample_boundary_points(self, n_points: int,
                               x_range=(1.2, 1.5),
                               y_fixed=None) -> torch.Tensor:
        """
        采样边界点
        Args:
            n_points: 采样点数
            x_range: x坐标的范围（绝对值）
            y_fixed: 固定的y坐标，如果不指定则使用config.y_interface
        """
        if y_fixed is None:
            y_fixed = self.config.y_interface

        # 解构x范围
        x_min, x_max = x_range

        # 创建基础随机数
        rand_nums = torch.rand(n_points, 2, device=self.device)

        # 决定每个点在左边界还是右边界
        boundary_side = torch.rand(n_points, 1, device=self.device) > 0.5

        # 计算x坐标
        x_base = x_min + rand_nums[:, 0:1] * (x_max - x_min)
        x_sign = torch.where(boundary_side, 1.0, -1.0)
        x_coords = x_sign * x_base

        # y方向固定
        y_coords = torch.full((n_points, 1), y_fixed, device=self.device)

        # 时间采样
        t_coords = rand_nums[:, 1:2] * self.config.t_max

        # 合并成(x, y, t)格式
        points = torch.cat([x_coords, y_coords, t_coords], dim=1)
        points.requires_grad_(True)

        return points

    def sample_initial_points(self, n_points: int) -> torch.Tensor:
        """采样初始点(t=0)"""
        x = torch.rand(n_points, 1, device=self.device) * 2.2 - 1.1
        y = torch.rand(n_points, 1, device=self.device) * 2 - 0.8
        t = torch.zeros(n_points, 1, device=self.device)

        points = torch.cat([x, y, t], dim=1)
        points.requires_grad_(True)
        return points


# ==================== 改进的训练器 ====================
class ImprovedPINNTrainer:
    """改进的PINN训练器"""

    def __init__(self, model: nn.Module, flux_field: EfficientFluxField, config: Config):
        self.model = model.to(config.device)
        self.flux_field = flux_field
        self.config = config
        self.device = config.device

        self.logger = logging.getLogger(__name__)
        self.sampler = AdaptiveSampler(config)

        # 损失历史
        self.loss_history = {
            'total': [], 'pde': [], 'reg': [], 'bc': [], 'ic': [],
            'grad_norm': [], 'lr': []
        }

    def compute_pde_loss(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算PDE损失"""
        # 分离坐标
        x = points[:, 0:1]
        y = points[:, 1:2]
        t = points[:, 2:3]

        # 获取通量
        flux_x, flux_y = self.flux_field.get_flux_gpu(x, y)

        # 前向传播
        phi = self.model(points)

        # 计算梯度
        phi_x, phi_y, phi_t = self.model.compute_gradients(points, phi)

        # PDE残差: φ_t + F·∇φ = 0
        pde_residual = phi_t + flux_x * phi_x + flux_y * phi_y

        # 梯度模长
        grad_norm = torch.sqrt(phi_x ** 2 + phi_y ** 2 + 1e-8)

        # 损失
        pde_loss = torch.mean(pde_residual ** 2)

        return pde_loss, grad_norm

    def compute_regularization_loss(self, grad_norm: torch.Tensor) -> torch.Tensor:
        """计算正则化损失"""
        # 梯度模长正则化（保持|∇φ|接近1）
        reg_loss = torch.mean((grad_norm - 1.0) ** 2)

        return reg_loss

    def compute_boundary_loss(self, boundary_points: torch.Tensor) -> torch.Tensor:
        """计算边界条件损失（保持定义但不使用）"""
        phi = self.model(boundary_points)

        # 在边界上，水平集函数应满足某些条件
        # 这里假设边界上φ=0（零等值线在边界上）
        target = torch.zeros_like(phi)
        bc_loss = torch.mean((phi - target) ** 2)

        return bc_loss

    def compute_initial_loss(self, initial_points: torch.Tensor) -> torch.Tensor:
        """计算初始条件损失"""
        phi = self.model(initial_points)

        # 初始条件: φ(x,y,0) = y - y_interface
        y_coord = initial_points[:, 1:2]
        target = y_coord - self.config.y_interface

        ic_loss = torch.mean((phi - target) ** 2)

        return ic_loss

    def compute_losses(self, collocation_points: torch.Tensor,
                       boundary_points: torch.Tensor,
                       initial_points: torch.Tensor) -> Dict[str, torch.Tensor]:
        """计算所有损失"""
        losses = {}

        # 1. PDE损失
        pde_loss, grad_norm = self.compute_pde_loss(collocation_points)
        losses['pde'] = pde_loss
        losses['grad_norm'] = grad_norm.mean()

        # 2. 正则化损失
        reg_loss = self.compute_regularization_loss(grad_norm)
        losses['reg'] = reg_loss

        # 3. 边界条件损失（计算但不加入总损失）
        bc_loss = self.compute_boundary_loss(boundary_points)
        losses['bc'] = bc_loss

        # 4. 初始条件损失
        ic_loss = self.compute_initial_loss(initial_points)
        losses['ic'] = ic_loss

        # 总损失 - 完全不包含边界损失
        total_loss = (self.config.lambda_pde * pde_loss +
                      self.config.lambda_reg * reg_loss +
                      self.config.lambda_ic * ic_loss)
        # 注意：lambda_bc * bc_loss 被排除在外

        losses['total'] = total_loss

        return losses

    def train_epoch(self, optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()

        # 采样点
        collocation_points = self.sampler.sample_collocation_points(self.config.n_points)
        boundary_points = self.sampler.sample_boundary_points(self.config.n_points // 32)
        initial_points = self.sampler.sample_initial_points(self.config.n_points // 32)

        # 计算损失
        losses = self.compute_losses(collocation_points, boundary_points, initial_points)

        # 反向传播
        optimizer.zero_grad()
        losses['total'].backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        optimizer.step()

        # 记录损失
        loss_dict = {k: v.item() if isinstance(v, torch.Tensor) else v
                     for k, v in losses.items()}
        loss_dict['lr'] = optimizer.param_groups[0]['lr']

        return loss_dict

    def train(self, save_dir: str) -> Dict[str, List[float]]:
        """训练循环"""
        os.makedirs(save_dir, exist_ok=True)

        # 优化器
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=1e-4
        )

        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=50
        )

        self.logger.info(f"开始训练，设备: {self.device}")
        self.logger.info(f"模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")

        # 训练循环
        for epoch in range(self.config.n_epochs):
            start_time = time.time()

            # 训练一个epoch
            loss_dict = self.train_epoch(optimizer)

            # 更新学习率
            scheduler.step(loss_dict['total'])

            # 记录历史
            for key, value in loss_dict.items():
                if key in self.loss_history:
                    self.loss_history[key].append(value)

            # 日志
            if epoch % 500 == 0 or epoch == self.config.n_epochs - 1:
                elapsed = time.time() - start_time
                self.logger.info(
                    f"Epoch {epoch:04d}/{self.config.n_epochs} | "
                    f"Total: {loss_dict['total']:.2e} | "
                    f"PDE: {loss_dict['pde']:.2e} | "
                    f"Reg: {loss_dict['reg']:.2e} | "
                    f"BC: {loss_dict['bc']:.2e} | "
                    f"IC: {loss_dict['ic']:.2e} | "
                    f"Time: {elapsed:.2f}s | "
                    f"LR: {loss_dict['lr']:.2e}"
                )

            # 保存检查点
            if epoch % self.config.checkpoint_interval == 0 or epoch == self.config.n_epochs - 1:
                self._save_checkpoint(epoch, optimizer, save_dir)

        # 保存最终模型
        self._save_model(save_dir)

        return self.loss_history

    def _save_checkpoint(self, epoch: int, optimizer: torch.optim.Optimizer,
                         save_dir: str):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_history': self.loss_history,
            'config': self.config
        }

        checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch:04d}.pth')
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"检查点保存到: {checkpoint_path}")

    def _save_model(self, save_dir: str):
        """保存完整模型"""
        model_path = os.path.join(save_dir, 'model_final.pth')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'loss_history': self.loss_history
        }, model_path)
        self.logger.info(f"最终模型保存到: {model_path}")


# ==================== 评估和可视化 ====================
class LevelSetEvaluator:
    """水平集评估器"""

    def __init__(self, model: nn.Module, config: Config, flux_field: EfficientFluxField = None):
        self.model = model
        self.config = config
        self.flux_field = flux_field
        self.device = config.device
        self.model.eval()

        # 设置matplotlib使用Agg后端（无GUI）
        import matplotlib
        matplotlib.use('Agg')

    def evaluate_grid(self, x_range=(-1.5, 1.5), y_range=(-1.5, 1.0),
                      t_values=None, grid_size=200):
        """在网格上评估"""
        if t_values is None:
            t_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        x = np.linspace(x_range[0], x_range[1], grid_size)
        y = np.linspace(y_range[0], y_range[1], grid_size)
        X, Y = np.meshgrid(x, y)

        results = {}

        with torch.no_grad():
            for t in t_values:
                # 创建输入
                x_tensor = torch.FloatTensor(X.ravel()).unsqueeze(1).to(self.device)
                y_tensor = torch.FloatTensor(Y.ravel()).unsqueeze(1).to(self.device)
                t_tensor = torch.ones_like(x_tensor) * t

                inputs = torch.cat([x_tensor, y_tensor, t_tensor], dim=1)

                # 预测
                phi_pred = self.model(inputs)
                phi_grid = phi_pred.cpu().numpy().reshape(grid_size, grid_size)

                # 获取通量（如果可用）
                flux_x_grid = None
                flux_y_grid = None
                if self.flux_field is not None:
                    flux_x, flux_y = self.flux_field.get_flux_gpu(
                        x_tensor, y_tensor
                    )
                    flux_x_grid = flux_x.cpu().numpy().reshape(grid_size, grid_size)
                    flux_y_grid = flux_y.cpu().numpy().reshape(grid_size, grid_size)

                results[t] = {
                    'X': X,
                    'Y': Y,
                    'phi': phi_grid,
                    'flux_x': flux_x_grid,
                    'flux_y': flux_y_grid
                }

        return results

    def create_visualization(self, results, save_dir: str):
        """创建可视化（使用纯文件保存，无交互）"""
        os.makedirs(save_dir, exist_ok=True)

        # 1. 2D等高线图
        t_values = list(results.keys())
        n_times = min(6, len(t_values))

        # 创建子图
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        if n_times == 1:
            fig, axes = plt.subplots(1, 1, figsize=(10, 8))
            axes = np.array([[axes]])

        axes_flat = axes.ravel()

        for idx in range(n_times):
            t = t_values[idx]
            data = results[t]
            X, Y = data['X'], data['Y']
            phi = data['phi']

            ax = axes_flat[idx]

            # 等高线填充
            contour = ax.contourf(X, Y, phi, levels=50, cmap='RdBu_r')

            # 零等值线
            zero_contour = ax.contour(X, Y, phi, levels=[0], colors='black', linewidths=2)

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title(f't = {t:.2f}')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.invert_yaxis()
            # 颜色条
            if idx == 0:  # 只在第一个子图添加颜色条
                plt.colorbar(contour, ax=ax, shrink=0.8)

        # 隐藏多余的子图
        for idx in range(n_times, len(axes_flat)):
            axes_flat[idx].axis('off')

        plt.tight_layout()
        contour_path = os.path.join(save_dir, 'levelset_contours.png')
        plt.savefig(contour_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        # 2. 创建通量场可视化
        if self.flux_field is not None:
            self.create_flux_visualization(results, save_dir)

        # 3. 创建动画GIF（可选）
        try:
            self.create_animation(results, save_dir)
        except Exception as e:
            print(f"创建动画失败: {e}")

        return {
            'contour_plot': contour_path
        }

    def create_flux_visualization(self, results, save_dir: str):
        """创建通量场可视化"""
        t_values = list(results.keys())

        # 选择一个时间点（例如t=0.5）来显示通量场
        if 0.5 in t_values:
            t = 0.5
        else:
            t = t_values[len(t_values) // 2]

        data = results[t]
        X, Y = data['X'], data['Y']
        phi = data['phi']
        flux_x = data['flux_x']
        flux_y = data['flux_y']

        if flux_x is None or flux_y is None:
            return

        # 计算通量大小
        flux_magnitude = np.sqrt(flux_x ** 2 + flux_y ** 2)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. 通量大小
        ax1 = axes[0, 0]
        contour1 = ax1.contourf(X, Y, flux_magnitude, levels=50, cmap='viridis')
        ax1.set_title('Flux Magnitude')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_aspect('equal')
        plt.colorbar(contour1, ax=ax1)

        # 2. 通量流线图
        ax2 = axes[0, 1]
        ax2.streamplot(X, Y, flux_x, flux_y, color='black', linewidth=0.5, density=2)
        ax2.set_title('Flux Streamlines')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_aspect('equal')

        # 3. 通量x分量
        ax3 = axes[1, 0]
        contour3 = ax3.contourf(X, Y, flux_x, levels=50, cmap='RdBu_r')
        ax3.set_title('Flux X Component')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_aspect('equal')
        plt.colorbar(contour3, ax=ax3)

        # 4. 通量y分量
        ax4 = axes[1, 1]
        contour4 = ax4.contourf(X, Y, flux_y, levels=50, cmap='RdBu_r')
        ax4.set_title('Flux Y Component')
        ax4.set_xlabel('X')
        ax4.set_ylabel('Y')
        ax4.set_aspect('equal')
        plt.colorbar(contour4, ax=ax4)

        plt.tight_layout()
        flux_path = os.path.join(save_dir, 'flux_field.png')
        plt.savefig(flux_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def create_animation(self, results, save_dir: str, fps: int = 5):
        """创建动画GIF"""
        try:
            from matplotlib.animation import FuncAnimation, PillowWriter

            t_values = sorted(results.keys())
            data_t0 = results[t_values[0]]
            X, Y = data_t0['X'], data_t0['Y']

            fig, ax = plt.subplots(figsize=(10, 8))

            # 初始化
            contour = ax.contourf(X, Y, data_t0['phi'], levels=50, cmap='RdBu_r')
            zero_line = ax.contour(X, Y, data_t0['phi'], levels=[0], colors='black', linewidths=2)


            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title(f'Level Set Evolution (t={t_values[0]:.2f})')
            ax.invert_yaxis()
            ax.set_aspect('equal')
            plt.colorbar(contour, ax=ax)

            def update(frame_idx):
                t = t_values[frame_idx]
                data = results[t]

                # 清除当前图形
                ax.clear()

                # 重新绘制
                contour = ax.contourf(X, Y, data['phi'], levels=50, cmap='RdBu_r')
                ax.contour(X, Y, data['phi'], levels=[0], colors='black', linewidths=2)

                circle = plt.Circle((0, 0), 1.0, color='green', fill=False,
                                    linestyle='--', linewidth=2)
                ax.add_patch(circle)

                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_title(f'Level Set Evolution (t={t:.2f})')
                ax.set_aspect('equal')

                return contour,

            # 创建动画
            anim = FuncAnimation(fig, update, frames=len(t_values), interval=1000 / fps, blit=False)

            # 保存为GIF
            gif_path = os.path.join(save_dir, 'levelset_evolution.gif')
            anim.save(gif_path, writer=PillowWriter(fps=fps), dpi=100)
            print(f"动画保存到: {gif_path}")

            plt.close(fig)

        except ImportError:
            print("Pillow未安装，跳过动画创建")
        except Exception as e:
            print(f"创建动画时出错: {e}")

    def plot_training_history(self, loss_history: Dict[str, List[float]], save_dir: str):
        """绘制训练历史"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 总损失
        axes[0, 0].semilogy(loss_history['total'], 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Total Loss')
        axes[0, 0].set_title('Total Training Loss')
        axes[0, 0].grid(True, alpha=0.3)

        # PDE损失
        axes[0, 1].semilogy(loss_history['pde'], 'r-', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('PDE Loss')
        axes[0, 1].set_title('PDE Residual Loss')
        axes[0, 1].grid(True, alpha=0.3)

        # 正则损失
        axes[0, 2].semilogy(loss_history['reg'], 'g-', linewidth=2)
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Regularization Loss')
        axes[0, 2].set_title('Regularization Loss')
        axes[0, 2].grid(True, alpha=0.3)

        # 边界损失（仍然绘制但注意lambda_bc=0）
        axes[1, 0].semilogy(loss_history['bc'], 'm-', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Boundary Loss')
        axes[1, 0].set_title('Boundary Condition Loss (λ=0)')
        axes[1, 0].grid(True, alpha=0.3)

        # 初始条件损失
        axes[1, 1].semilogy(loss_history['ic'], 'c-', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Initial Condition Loss')
        axes[1, 1].set_title('Initial Condition Loss')
        axes[1, 1].grid(True, alpha=0.3)

        # 梯度模长
        if 'grad_norm' in loss_history:
            axes[1, 2].plot(loss_history['grad_norm'], 'k-', linewidth=2)
            axes[1, 2].axhline(y=1.0, color='r', linestyle='--', alpha=0.7)
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Gradient Norm')
            axes[1, 2].set_title('Gradient Norm Evolution')
            axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        loss_plot_path = os.path.join(save_dir, 'training_history.png')
        plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        return loss_plot_path


# ==================== 主程序 ====================
def main():
    """主训练流程"""
    import matplotlib
    matplotlib.use('Agg')  # 设置非交互式后端

    # 创建配置
    config = Config()

    # 创建结果目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f'./results_levelset_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)

    # 设置日志
    logger = setup_logging(results_dir)
    logger.info(f"结果目录: {results_dir}")
    logger.info(f"设备: {config.device}")

    # 保存配置
    config_path = os.path.join(results_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config.__dict__, f, indent=2)

    try:
        # 步骤1: 创建通量场
        logger.info("=" * 50)
        logger.info("步骤1: 创建高效通量场")
        logger.info("=" * 50)

        flux_field = EfficientFluxField(config)

        # 步骤2: 创建和训练模型
        logger.info("\n" + "=" * 50)
        logger.info("步骤2: 创建和训练PINN模型")
        logger.info("=" * 50)

        model = SimplifiedLevelSetNet(config)
        trainer = ImprovedPINNTrainer(model, flux_field, config)

        # 训练
        loss_history = trainer.train(results_dir)

        # 步骤3: 评估模型
        logger.info("\n" + "=" * 50)
        logger.info("步骤3: 评估模型")
        logger.info("=" * 50)

        evaluator = LevelSetEvaluator(model, config, flux_field)
        results = evaluator.evaluate_grid(
            x_range=(-1.5, 1.5),
            y_range=(-1.5, 0.5),
            t_values=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            grid_size=200
        )

        # 步骤4: 可视化
        logger.info("\n" + "=" * 50)
        logger.info("步骤4: 生成可视化")
        logger.info("=" * 50)

        viz_paths = evaluator.create_visualization(results, results_dir)

        # 步骤5: 训练曲线
        logger.info("\n" + "=" * 50)
        logger.info("步骤5: 绘制训练曲线")
        logger.info("=" * 50)

        loss_plot_path = evaluator.plot_training_history(loss_history, results_dir)

        # 步骤6: 保存结果数据
        logger.info("\n" + "=" * 50)
        logger.info("步骤6: 保存结果数据")
        logger.info("=" * 50)

        # 保存评估结果
        results_data = {}
        for t, data in results.items():
            results_data[float(t)] = {
                'phi_min': float(np.min(data['phi'])),
                'phi_max': float(np.max(data['phi'])),
                'phi_mean': float(np.mean(data['phi'])),
                'zero_contour_area': float(np.sum(np.abs(data['phi']) < 0.1) / data['phi'].size)
            }

        results_json_path = os.path.join(results_dir, 'evaluation_results.json')
        with open(results_json_path, 'w') as f:
            json.dump(results_data, f, indent=2)

        logger.info(f"评估结果保存到: {results_json_path}")

        # 总结
        logger.info("\n" + "=" * 50)
        logger.info("训练完成!")
        logger.info(f"结果保存在: {results_dir}")
        logger.info(f"最终损失: {loss_history['total'][-1]:.2e}")
        logger.info("=" * 50)

        # 输出文件列表
        logger.info("生成的文件:")
        for file in os.listdir(results_dir):
            if file.endswith(('.png', '.gif', '.json', '.pth', '.log')):
                logger.info(f"  - {file}")

        return {
            'model': model,
            'flux_field': flux_field,
            'results_dir': results_dir,
            'loss_history': loss_history
        }

    except Exception as e:
        logger.error(f"训练过程中出错: {e}")
        import traceback
        traceback.print_exc()
        raise


# ==================== 命令行接口 ====================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='2D Level Set PINN 训练和推理')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'inference'],
                        help='运行模式: train (训练) 或 inference (推理)')
    parser.add_argument('--model-path', type=str, default=None,
                        help='推理模式下的模型路径')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='输出目录')

    args = parser.parse_args()

    if args.mode == 'train':
        results = main()
    elif args.mode == 'inference':
        if args.model_path is None:
            print("请提供模型路径: --model-path /path/to/model.pth")
        else:
            # 轻量级推理函数
            def run_inference(model_path: str, output_dir: str = None):
                import matplotlib
                matplotlib.use('Agg')

                if output_dir is None:
                    output_dir = './inference_results'
                os.makedirs(output_dir, exist_ok=True)

                # 加载模型
                checkpoint = torch.load(model_path, map_location='cpu')
                config_dict = checkpoint['config']

                # 重新创建配置
                config = Config()
                for key, value in config_dict.items():
                    if hasattr(config, key):
                        setattr(config, key, value)

                # 创建模型
                model = SimplifiedLevelSetNet(config)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()

                print(f"模型加载成功: {model_path}")

                # 评估
                evaluator = LevelSetEvaluator(model, config)
                results = evaluator.evaluate_grid()

                # 可视化
                viz_paths = evaluator.create_visualization(results, output_dir)

                print(f"推理完成，结果保存在: {output_dir}")

                return evaluator, results


            run_inference(args.model_path, args.output_dir)
    else:
        print(f"未知模式: {args.mode}")