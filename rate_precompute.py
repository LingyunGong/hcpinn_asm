"""
速度场预计算：通量积分与位置相关通量场
核心：物理计算函数 + PositionalFluxField（预计算 + 插值查询）
"""

import numpy as np
from scipy.integrate import quad
from multiprocessing import Pool, cpu_count
import pickle
import os
from config.default_config import TrainingConfig

config = TrainingConfig()
radius = config.radius
mask_h = radius * 2 * config.h
rate_0 = config.rate


def g(theta, center=np.pi / 2, sigma_1=config.sigma):
    """高斯分布函数，模拟离子通量角度分布"""
    return np.exp(-(theta - center) ** 2 / (2 * sigma_1 ** 2))


def g2(theta, center=0, sigma=config.sigma):
    return np.exp(-(theta - center) ** 2 / (2 * sigma ** 2))


def neutral_etch_rate(x, y, R0=0.5, alpha=3.0, D=0.15, ks=0.16, w0=0.5):
    """中性粒子扩散贡献的刻蚀速率，正比于浓度梯度"""
    y -= -0.7
    x = np.abs(x)
    lambda_val = np.sqrt(2.0 * ks / (D * w0))
    r_x = R0 * np.exp(-alpha * lambda_val * x + -alpha * lambda_val * y)
    r_y = R0 * np.exp(-alpha * lambda_val * y)
    return r_x, r_y


def calculate_integration_limits(x_val, y_val, y_0=-0.7, h=mask_h, r=radius):
    """计算直接通量积分的上下限"""
    upper_bound = y_0 - h
    denominator_lower = x_val + r
    denominator_upper = x_val - r

    if denominator_upper > 0:
        lower_limit = np.arctan((y_val - upper_bound) / denominator_lower)
        upper_limit = -np.arctan(np.abs(y_val - y_0) / denominator_upper) if abs(denominator_upper) >= 1e-8 else np.pi / 2
    else:
        lower_limit = np.arctan((y_val - upper_bound) / denominator_lower)
        upper_limit = (np.pi / 2 if abs(denominator_upper) < 1e-8
                       else np.pi + np.arctan((y_val - upper_bound) / denominator_upper))
    return lower_limit, upper_limit


def calculate_reflect_flux(x_val, y_val, y_0=-0.7, h=mask_h, r=radius):
    """通过一维积分计算离子镜面反射通量"""
    drift = -min(np.abs(y_val - y_0) * 0.072, 0.12)
    upper_bound = y_0 - h
    upper_limit = y_val - upper_bound
    lower_limit = 0.5 * np.abs(x_val + radius - drift) / (2 * radius + x_val - drift) * upper_limit

    if lower_limit >= upper_limit or x_val > r:
        return 0.0, 0.0

    eps = 1e-10

    def reflect_integrand_x(y):
        rl = np.sqrt((r - x_val + drift) ** 2 + y ** 2 + eps)
        rr = np.sqrt((r + x_val - drift) ** 2 + y ** 2 + eps)
        y_rl = np.clip(y / rl, -1.0, 1.0)
        y_rr = np.clip(y / rr, -1.0, 1.0)
        theta_1 = np.arccos(y_rl)
        theta_2 = np.arccos(y_rr)
        left_x = g2(theta_1) / (rl + eps) * np.sin(theta_1)
        right_x = g2(theta_1) / rr * np.sin(theta_2)
        if 0 <= x_val < r - drift:
            return (-left_x + right_x) * np.exp(-theta_2 - theta_1) * 0.01
        else:
            return right_x * np.exp(-theta_2 - theta_1) * 0.01

    def reflect_integrand_y(y):
        rl = np.sqrt((r - x_val + drift) ** 2 + y ** 2 + eps)
        rr = np.sqrt((r + x_val - drift) ** 2 + y ** 2 + eps)
        y_rl = np.clip(y / rl, -1.0, 1.0)
        y_rr = np.clip(y / rr, -1.0, 1.0)
        theta_1 = np.arccos(y_rl)
        theta_2 = np.arccos(y_rr)
        left_y = g2(theta_1) / (rl + eps) * np.cos(theta_1)
        right_y = g2(theta_1) / rr * np.cos(theta_2)
        if 0 <= x_val <= r - drift:
            return left_y * np.exp(-theta_1) + right_y * np.exp(-theta_2)
        else:
            return right_y * np.exp(-theta_2)

    try:
        ix, _ = quad(reflect_integrand_x, lower_limit, upper_limit)
        iy, _ = quad(reflect_integrand_y, lower_limit, upper_limit)
        return np.abs(ix), iy
    except:
        return 0.0, 0.0


def calculate_positional_flux_integral(point_data, r=radius):
    """计算单个位置点的通量积分 (integral_x, integral_y)"""
    x_val, y_val = point_data
    lower_limit, upper_limit = calculate_integration_limits(x_val, y_val)
    if lower_limit > upper_limit:
        lower_limit, upper_limit = upper_limit, lower_limit

    def integrand_x(theta):
        return g(theta) * np.abs(np.cos(theta))

    def integrand_y(theta):
        return g(theta) * np.sin(theta)

    try:
        integral_x = quad(integrand_x, lower_limit, upper_limit)[0]
        integral_y = quad(integrand_y, lower_limit, upper_limit)[0]
        reflect_x, reflect_y = calculate_reflect_flux(x_val, y_val)
        neutral_x, neutral_y = neutral_etch_rate(x_val, y_val)
        ratio = 0.002
        integral_x = integral_x * neutral_x * 32 / (integral_x * 32 + neutral_x)
        integral_y = integral_y * (1 - ratio) * 1.3 + reflect_y * ratio
        integral_x *= np.pi * rate_0
        integral_y *= np.pi * rate_0
        return integral_x, integral_y
    except:
        return 0.0, 0.0


def _calculate_batch_positional_flux(batch_data):
    """批量计算位置通量（多进程用）"""
    return [calculate_positional_flux_integral(data) for data in batch_data]


class PositionalFluxField:
    """位置通量场：预计算 (r, y) 网格上的通量积分，运行时插值查询"""

    def __init__(self, cache_file='models/flux_field_cache.pkl',
                 r_range=(0, 2 * config.radius), y_range=(-1.0, 2.2), resolution=200):
        self.cache_file = cache_file
        self.r_range = r_range
        self.y_range = y_range
        self.resolution = resolution
        self.flux_x_grid = None
        self.flux_y_grid = None
        self.r_grid = None
        self.y_grid = None
        self.interpolator_x = None
        self.interpolator_y = None

        if not self.load_cache():
            raise FileNotFoundError(
                f"缓存文件 {self.cache_file} 不存在，请先运行预计算生成通量场。"
                f"\n  调用方式: PositionalFluxField().precompute_flux_field(n_processes=N)")

    def precompute_flux_field(self, n_processes=None):
        """预计算通量场并缓存"""
        if self.load_cache():
            print(f"从缓存 {self.cache_file} 加载通量场")
            return

        print("开始预计算通量场...")
        n_processes = n_processes or max(1, cpu_count() - 1)

        self.r_grid = np.linspace(self.r_range[0], self.r_range[1], self.resolution)
        self.y_grid = np.linspace(self.y_range[0], self.y_range[1], self.resolution)
        self.flux_x_grid = np.zeros((self.resolution, self.resolution))
        self.flux_y_grid = np.zeros((self.resolution, self.resolution))

        grid_points = [(r, y) for r in self.r_grid for y in self.y_grid]
        indices = [(i, j) for i in range(self.resolution) for j in range(self.resolution)]

        batch_size = max(1, len(grid_points) // n_processes)
        batches = [grid_points[i:i + batch_size] for i in range(0, len(grid_points), batch_size)]

        print(f"使用 {n_processes} 个进程计算 {len(grid_points)} 个网格点...")
        with Pool(processes=n_processes) as pool:
            batch_results = pool.map(_calculate_batch_positional_flux, batches)

        flat_idx = 0
        for batch_result in batch_results:
            for flux_x, flux_y in batch_result:
                i, j = indices[flat_idx]
                self.flux_x_grid[i, j] = flux_x
                self.flux_y_grid[i, j] = flux_y
                flat_idx += 1

        self.save_cache()
        print("通量场计算完成并已缓存")

    def get_positional_flux(self, points):
        """获取位置相关的通量积分 [N, 2]"""
        if self.interpolator_x is None:
            self._create_interpolators()

        points_np = points.detach().cpu().numpy() if hasattr(points, 'detach') else np.asarray(points)
        r_vals = np.sqrt(points_np[:, 0] ** 2 + points_np[:, 2] ** 2)
        y_vals = points_np[:, 1]
        r_vals = np.clip(r_vals, self.r_range[0] + 1e-6, self.r_range[1] - 1e-6)
        y_vals = np.clip(y_vals, self.y_range[0] + 1e-6, self.y_range[1] - 1e-6)

        flux_x = self.interpolator_x(np.column_stack([r_vals, y_vals]))
        flux_y = self.interpolator_y(np.column_stack([r_vals, y_vals]))
        return np.column_stack([flux_x, flux_y])

    def get_positional_flux_tensor(self, points):
        """获取位置通量的 PyTorch 张量版本"""
        flux_np = self.get_positional_flux(points)
        import torch
        if hasattr(points, 'device'):
            return torch.from_numpy(flux_np).float().to(points.device)
        return torch.from_numpy(flux_np).float()

    def _create_interpolators(self):
        from scipy.interpolate import LinearNDInterpolator
        R, Y = np.meshgrid(self.r_grid, self.y_grid, indexing='ij')
        points_grid = np.column_stack([R.ravel(), Y.ravel()])
        self.interpolator_x = LinearNDInterpolator(points_grid, self.flux_x_grid.ravel())
        self.interpolator_y = LinearNDInterpolator(points_grid, self.flux_y_grid.ravel())

    def save_cache(self):
        cache_data = {
            'r_grid': self.r_grid, 'y_grid': self.y_grid,
            'flux_x_grid': self.flux_x_grid, 'flux_y_grid': self.flux_y_grid,
            'r_range': self.r_range, 'y_range': self.y_range, 'resolution': self.resolution
        }
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        with open(self.cache_file, 'wb') as f:
            pickle.dump(cache_data, f)

    def load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                self.r_grid = cache_data['r_grid']
                self.y_grid = cache_data['y_grid']
                self.flux_x_grid = cache_data['flux_x_grid']
                self.flux_y_grid = cache_data['flux_y_grid']
                self.r_range = cache_data['r_range']
                self.y_range = cache_data['y_range']
                self.resolution = cache_data['resolution']
                self._create_interpolators()
                return True
            except:
                print("缓存文件损坏，重新计算通量场")
                return False
        return False


def plot_flux_field(flux_field=None, save_path='flux_field.png'):
    """
    简化速度通量场可视化：2x2 子图 (integral_x, integral_y, 幅值, 向量场)
    如不传 flux_field，自动从默认缓存加载。
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if flux_field is None:
        flux_field = PositionalFluxField()

    R, Y = np.meshgrid(flux_field.r_grid, flux_field.y_grid, indexing='ij')
    fx = flux_field.flux_x_grid
    fy = flux_field.flux_y_grid
    fmag = np.sqrt(fx ** 2 + fy ** 2)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle('Velocity flux field (r-y plane)', fontsize=14)

    titles = [r'$\Phi_x$', r'$\Phi_y$', r'$|\Phi|$', 'Vector field']
    data = [fx, fy, fmag, None]
    cmaps = ['viridis', 'plasma', 'hot', None]

    for idx in range(3):
        im = axes[idx // 2, idx % 2].imshow(
            data[idx].T, origin='lower', aspect='auto', cmap=cmaps[idx],
            extent=[*flux_field.r_range, *flux_field.y_range])
        axes[idx // 2, idx % 2].set_title(titles[idx])
        axes[idx // 2, idx % 2].set_xlabel('r')
        axes[idx // 2, idx % 2].set_ylabel('y')
        plt.colorbar(im, ax=axes[idx // 2, idx % 2])

    # 向量场（下采样）
    step = max(1, flux_field.resolution // 20)
    axes[1, 1].quiver(R[::step, ::step], Y[::step, ::step],
                      fx[::step, ::step], fy[::step, ::step],
                      fmag[::step, ::step], cmap='coolwarm')
    axes[1, 1].set_title(titles[3])
    axes[1, 1].set_xlabel('r')
    axes[1, 1].set_ylabel('y')
    axes[1, 1].set_xlim(flux_field.r_range)
    axes[1, 1].set_ylim(flux_field.y_range)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'通量场可视化已保存到: {save_path}')