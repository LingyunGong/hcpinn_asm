import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.renderers.default = "notebook_connected"  # or "svg"
from scipy.integrate import quad
from multiprocessing import Pool, cpu_count
import pickle
import os
from config.default_config import TrainingConfig

config = TrainingConfig()
radius =config.radius
mask_h = radius * 2 * config.h
rate_0 = config.rate # 速率系数
# Define angular distribution of incident ions
def g(theta, center=np.pi / 2, sigma_1=config.sigma):
    """高斯分布函数，模拟离子通量分布"""
    return np.exp(-(theta - center) ** 2 / (2 * sigma_1 ** 2))

def g2(theta, center=0, sigma=config.sigma):
    return np.exp(-(theta - center) ** 2 / (2 * sigma ** 2))   # 高斯分布

# Neural transport
def neutral_etch_rate(x,y, R0=0.5, alpha=3.0, D=0.15, ks=0.16, w0=0.5):
    """
    计算中性粒子扩散贡献的刻蚀速率随深度的变化，正比于浓度梯度

    参数:
        y: 深度坐标 (μm)，标量或numpy数组
        R0: 刻蚀速率贡献系数 默认: 0.1
        alpha: 反应级数 默认: 1.0
        D: 扩散系数 (μm²/μs) 默认: 0.15
        ks: 表面反应速率常数 (μm/μs) 默认: 0.08
        w0: 沟槽开口宽度 (μm) 默认: 0.5
    返回:
        浓度梯度
    """
    y -=-0.7
    x= np.abs(x)
    # 计算衰减系数 λ = sqrt(2*ks/(D*w0))
    lambda_val = np.sqrt(2.0 * ks / (D * w0))
    # 计算刻蚀速率 R(y) = R0 * exp(-α * λ * y)
    r_x = R0 * np.exp(-alpha * lambda_val *x + -alpha * lambda_val *y)
    r_y =  R0 * np.exp(-alpha * lambda_val *y )
    return r_x, r_y

# Direct flux
def calculate_integration_limits(x_val, y_val, y_0=-0.7, h=mask_h, r =radius):
    """计算积分上下限，考虑x和y坐标"""
    # 默认初始界面y_0=-0.7, 掩膜高0.6
    upper_bound = y_0 - h
    denominator_lower = x_val + r
    denominator_upper = x_val - r

    if denominator_upper > 0:
        lower_limit = np.arctan((y_val - upper_bound) / denominator_lower)
        if abs(denominator_upper) < 1e-8:
            upper_limit = np.pi / 2
        else:
            upper_limit = -np.arctan(np.abs(y_val - y_0) / denominator_upper)
    else:
        lower_limit = np.arctan((y_val - upper_bound) / denominator_lower)
        if abs(denominator_upper) < 1e-8:
            upper_limit = np.pi / 2
        else:
            upper_limit = np.pi + np.arctan((y_val - upper_bound) / denominator_upper)

    return lower_limit, upper_limit

# Reflect flux
def calculate_reflect_flux(x_val, y_val, y_0=-0.7, h=mask_h, r =radius):
    """通过一维积分计算离子镜面反射通量"""
    # 计算积分上下限
    drift = -min(np.abs(y_val - y_0) * 0.072, 0.12)  # 侧壁倾斜导致的底部宽度变化 -min(np.abs(y_val - y_0) * 0.02, 0.05)
    upper_bound = y_0 - h
    upper_limit =  y_val - upper_bound
    lower_limit = 0.5 *np.abs(x_val+radius- drift)/(2 * radius +x_val - drift ) * upper_limit#max(y_val-y_0, 0)
    # 检查积分区间是否有效
    if lower_limit >= upper_limit or x_val> r : # or y_val<= y_0:
        return 0.0, 0.0
    eps = 1e-10  # 避免除零

    # 定义反射通量的被积函数
    def reflect_integrand_x(y):
        rl = np.sqrt((r - x_val+ drift) ** 2 + y ** 2 + eps)
        rr = np.sqrt((r + x_val-drift) ** 2 + y ** 2 + eps)

        # 限制 arcsin 参数在有效范围内
        y_rl_ratio = y / rl
        y_rr_ratio = y / rr

        # 使用 np.clip 限制参数在 [-1, 1] 范围内
        y_rl_ratio = np.clip(y_rl_ratio, -1.0, 1.0)
        y_rr_ratio = np.clip(y_rr_ratio, -1.0, 1.0)
        theta_1 = np.arccos(y_rl_ratio)
        theta_2 = np.arccos(y_rr_ratio)
        left_x = g2(theta_1) / (rl + eps)  * np.sin(theta_1)
        right_x = g2(theta_1) / rr  * np.sin(theta_2)
        if 0 <= x_val < r- drift:
            return (-left_x + right_x) * np.exp(-theta_2-theta_1)*0.01
        else:
            return right_x * np.exp(-theta_2-theta_1)*0.01

    def reflect_integrand_y(y):
        rl = np.sqrt((r - x_val+drift) ** 2 + y ** 2 + eps)
        rr = np.sqrt((r + x_val-drift) ** 2 + y ** 2 + eps)
        # 限制 arcsin 参数在有效范围内
        y_rl_ratio = y / rl
        y_rr_ratio = y / rr

        # 使用 np.clip 限制参数在 [-1, 1] 范围内
        y_rl_ratio = np.clip(y_rl_ratio, -1.0, 1.0)
        y_rr_ratio = np.clip(y_rr_ratio, -1.0, 1.0)
        theta_1 = np.arccos(y_rl_ratio)
        theta_2 = np.arccos(y_rr_ratio)
        left_y = g2(theta_1) / (rl + eps)  * np.cos(theta_1)
        right_y = g2(theta_1) / rr  * np.cos(theta_2)
        if 0 <= x_val <= r-drift:
            return left_y * np.exp(-theta_1) + right_y* np.exp(-theta_2)
        else:
            return right_y* np.exp(-theta_2)

    # 计算反射通量积分
    try:
        integral_reflect_x, err = quad(reflect_integrand_x, lower_limit, upper_limit)
        integral_reflect_y, err = quad(reflect_integrand_y, lower_limit, upper_limit)
        return np.abs(integral_reflect_x), integral_reflect_y # 负表示方向向内
    except:
        # 如果积分失败，返回0
        return 0.0, 0.0

# Effective flux at single point
def calculate_positional_flux_integral(point_data, r =radius):
    """计算单个位置点的通量积分（返回integral_x, integral_y，不包含法向量）"""
    x_val, y_val = point_data

    # 计算积分上下限
    lower_limit, upper_limit = calculate_integration_limits(x_val, y_val)

    # 确保下限小于上限
    if lower_limit > upper_limit:
        lower_limit, upper_limit = upper_limit, lower_limit

    # 定义被积函数
    def integrand_x(theta):
        return g(theta) * np.abs(np.cos(theta)) # 和法向夹角一定为锐角（凸图形），通量贡献为正，X法向量在速率模型里也取为了绝对值

    def integrand_y(theta):
        return g(theta) * np.sin(theta)

    # 计算积分
    try:
        integral_x = quad(integrand_x, lower_limit, upper_limit)[0]
        integral_y = quad(integrand_y, lower_limit, upper_limit)[0]
        reflect_x, reflect_y =calculate_reflect_flux(x_val, y_val)
        neutral_x, neutral_y = neutral_etch_rate(x_val, y_val)
        ratio = 0.002 #0.028
        integral_x = (integral_x * neutral_x *32/ (integral_x*32 + neutral_x)) #integral_x*(1-ratio) +reflect_x*ratio  #*config.side_p
        integral_y = integral_y*(1-ratio)*1.3 +reflect_y*ratio #+(integral_y * neutral_y *3/ (integral_y*3 + neutral_y))#+neutral_y
        # 乘以系数，但不乘以法向量
        integral_x *= np.pi * rate_0
        integral_y *= np.pi * rate_0

        return integral_x, integral_y
    except:
        # 如果积分失败，返回0
        return 0.0, 0.0


# 批量计算位置通量（用于多进程）
def calculate_batch_positional_flux(batch_data):
    """批量计算位置通量"""
    results = []
    for data in batch_data:
        results.append(calculate_positional_flux_integral(data))
    return results


class PositionalFluxField:
    """位置通量场计算器（只计算位置相关的通量积分）"""

    def __init__(self, r=1.0, cache_file='flux_field_cache.pkl',
                 r_range=(0, 0.8), y_range=(-0.8, 1.0), resolution=200):
        """
        初始化通量场计算器

        Args:
            r: 积分半径参数
            cache_file: 缓存文件名
            r_range: r坐标范围
            y_range: y坐标范围
            resolution: 网格分辨率
        """
        self.r = r
        self.cache_file = cache_file
        self.r_range = r_range
        self.y_range = y_range
        self.resolution = resolution

        # 通量场数据
        self.flux_x_grid = None  # integral_x 网格
        self.flux_y_grid = None  # integral_y 网格
        self.r_grid = None
        self.y_grid = None

        # 插值器
        self.interpolator_x = None
        self.interpolator_y = None

    def precompute_flux_field(self, n_processes=None):
        """预计算通量场（只需计算一次）"""
        if self.load_cache():
            print(f"从缓存 {self.cache_file} 加载通量场")
            return

        print("开始预计算通量场...")

        n_processes = n_processes or max(1, cpu_count() - 1)

        # 创建网格
        self.r_grid = np.linspace(self.r_range[0], self.r_range[1], self.resolution)
        self.y_grid = np.linspace(self.y_range[0], self.y_range[1], self.resolution)

        # 初始化通量网格
        self.flux_x_grid = np.zeros((self.resolution, self.resolution))
        self.flux_y_grid = np.zeros((self.resolution, self.resolution))

        # 准备所有网格点数据
        grid_points = []
        indices = []

        for i, r_val in enumerate(self.r_grid):
            for j, y_val in enumerate(self.y_grid):
                grid_points.append((r_val, y_val))
                indices.append((i, j))

        # 使用多进程计算
        batch_size = max(1, len(grid_points) // n_processes)
        batches = [grid_points[i:i + batch_size] for i in range(0, len(grid_points), batch_size)]

        print(f"使用 {n_processes} 个进程计算 {len(grid_points)} 个网格点...")

        with Pool(processes=n_processes) as pool:
            batch_results = pool.map(calculate_batch_positional_flux, batches)

        # 合并结果并填充网格
        flat_idx = 0
        for batch_result in batch_results:
            for flux_x, flux_y in batch_result:
                i, j = indices[flat_idx]
                self.flux_x_grid[i, j] = flux_x
                self.flux_y_grid[i, j] = flux_y
                flat_idx += 1

        # 保存缓存
        self.save_cache()
        print("通量场计算完成并已缓存")

    def get_positional_flux(self, points):
        """
        获取位置相关的通量积分

        Args:
            points: [N, 3] 形状的点坐标

        Returns:
            flux_integrals: [N, 2] 形状的通量积分 (integral_x, integral_y)
        """
        if self.interpolator_x is None:
            self._create_interpolators()

        # 计算r和y坐标
        points_np = points.detach().cpu().numpy() if torch.is_tensor(points) else points

        # 旋转对称：r = sqrt(x^2 + z^2)
        r_vals = np.sqrt(points_np[:, 0] ** 2 + points_np[:, 2] ** 2)
        y_vals = points_np[:, 1]

        # 确保在网格范围内
        r_vals = np.clip(r_vals, self.r_range[0] + 1e-6, self.r_range[1] - 1e-6)
        y_vals = np.clip(y_vals, self.y_range[0] + 1e-6, self.y_range[1] - 1e-6)

        # 插值得到每个点的通量积分
        flux_x = self.interpolator_x(np.column_stack([r_vals, y_vals]))
        flux_y = self.interpolator_y(np.column_stack([r_vals, y_vals]))

        # 组合成 [N, 2] 数组
        flux_integrals = np.column_stack([flux_x, flux_y])

        return flux_integrals

    def get_positional_flux_tensor(self, points):
        """获取位置通量的PyTorch张量版本"""
        flux_np = self.get_positional_flux(points)

        if torch.is_tensor(points):
            device = points.device
            return torch.from_numpy(flux_np).float().to(device)
        else:
            return torch.from_numpy(flux_np).float()

    def _create_interpolators(self):
        """创建插值器"""
        if self.flux_x_grid is None:
            self.precompute_flux_field()

        from scipy.interpolate import LinearNDInterpolator

        # 创建网格点
        R, Y = np.meshgrid(self.r_grid, self.y_grid, indexing='ij')
        points_grid = np.column_stack([R.ravel(), Y.ravel()])

        # 创建插值器
        self.interpolator_x = LinearNDInterpolator(points_grid, self.flux_x_grid.ravel())
        self.interpolator_y = LinearNDInterpolator(points_grid, self.flux_y_grid.ravel())

    def save_cache(self):
        """保存通量场缓存"""
        cache_data = {
            'r_grid': self.r_grid,
            'y_grid': self.y_grid,
            'flux_x_grid': self.flux_x_grid,
            'flux_y_grid': self.flux_y_grid,
            'r_range': self.r_range,
            'y_range': self.y_range,
            'resolution': self.resolution
        }

        with open(self.cache_file, 'wb') as f:
            pickle.dump(cache_data, f)

    def load_cache(self):
        """加载通量场缓存"""
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

                # 创建插值器
                self._create_interpolators()
                return True
            except:
                print("缓存文件损坏，重新计算通量场")
                return False
        return False


class SeparatedEtchingRateCalculator:
    """分离的通量场刻蚀速率计算器"""

    def __init__(self, etching_type='integral', coefficient=0.8,
                 flux_field=None, n_processes=None):
        """
        初始化刻蚀速率计算器

        Args:
            etching_type: 刻蚀类型，'isotropic' 或 'integral'
            coefficient: 通量系数
            flux_field: 预计算的位置通量场（PositionalFluxField实例）
            n_processes: 多进程数量
        """
        self.etching_type = etching_type
        self.coefficient = coefficient
        self.n_processes = n_processes or cpu_count()

        # 位置通量场
        if flux_field is None:
            self.flux_field = PositionalFluxField()
        else:
            self.flux_field = flux_field

        # 确保通量场已计算
        if self.flux_field.flux_x_grid is None:
            self.flux_field.precompute_flux_field(n_processes=self.n_processes)

    def compute_normal_components(self, normals):
        """
        计算法向量的旋转对称分量

        Args:
            normals: [N, 3] 形状的法向量

        Returns:
            n_x: [N] 旋转平面内的法向量分量
            n_y: [N] y方向的法向量分量
        """
        if torch.is_tensor(normals):
            normals_np = normals.detach().cpu().numpy()
        else:
            normals_np = normals

        # 计算旋转对称的法向量分量
        n_x = np.sqrt(normals_np[:, 0] ** 2 + normals_np[:, 2] ** 2)
        n_y = normals_np[:, 1]

        # 归一化（如果需要）
        norm_magnitude = np.sqrt(n_x ** 2 + n_y ** 2)
        mask = norm_magnitude > 1e-6
        n_x[mask] = n_x[mask] / norm_magnitude[mask]
        n_y[mask] = n_y[mask] / norm_magnitude[mask]

        return n_x, n_y

    def __call__(self, points, normals, curvatures, t=None):
        """计算刻蚀速率 = 位置通量 · 法向量"""
        if self.etching_type == 'isotropic':
            return torch.ones_like(points[:, :1]) * 0.5

        elif self.etching_type == 'integral':
            device = points.device if torch.is_tensor(points) else 'cpu'

            # 1. 获取位置相关的通量积分
            positional_flux = self.flux_field.get_positional_flux(points)  # [N, 2]

            # 2. 计算法向量分量
            n_x, n_y = self.compute_normal_components(normals)  # 各为[N]

            # 3. 计算点乘：通量 · 法向量
            flux_direct = positional_flux[:, 0] * n_x + positional_flux[:, 1] * n_y

            # 4. 应用系数并确保非负
            flux_direct = np.maximum(flux_direct, 0) * self.coefficient

            # 5. 转换为张量
            if torch.is_tensor(points):
                flux_tensor = torch.from_numpy(flux_direct).float().to(device)
            else:
                flux_tensor = torch.from_numpy(flux_direct).float()

            # 重塑为 [N, 1]
            flux_tensor = flux_tensor.view(-1, 1)

            return flux_tensor

        else:
            raise ValueError(f"未知的刻蚀类型: {self.etching_type}")

    def get_flux_components(self, points):
        """
        分别获取通量分量和法向量分量（用于调试和分析）

        Returns:
            flux_integrals: [N, 2] 位置通量积分 (integral_x, integral_y)
            normal_components: [N, 2] 法向量分量 (n_x, n_y)
        """
        # 获取位置通量
        positional_flux = self.flux_field.get_positional_flux(points)

        # 假设使用默认法向量计算
        from scipy.spatial.transform import Rotation as R
        import random

        # 生成一些随机法向量用于演示
        N = len(points) if hasattr(points, '__len__') else 1
        random_normals = []

        for _ in range(N):
            # 随机旋转生成法向量
            rotation = R.random(random_state=random.randint(0, 1000))
            normal = rotation.apply([0, 1, 0])  # 从[0,1,0]开始随机旋转
            random_normals.append(normal)

        random_normals = np.array(random_normals)
        n_x, n_y = self.compute_normal_components(random_normals)
        normal_components = np.column_stack([n_x, n_y])

        return positional_flux, normal_components


# 使用示例
def example_usage():
    # 创建通量场计算器（只需一次）
    flux_field = PositionalFluxField(
        cache_file='my_flux_field.pkl',
        r_range=(0, 3),  # 根据实际需求调整
        y_range=(-2, 2),
        resolution=150
    )

    # 预计算通量场（如果缓存不存在）
    flux_field.precompute_flux_field()

    # 创建刻蚀速率计算器
    etcher = SeparatedEtchingRateCalculator(
        etching_type='integral',
        coefficient=0.8,
        flux_field=flux_field
    )

    # 生成测试数据
    batch =1000
    points = torch.randn(batch, 3)  # 1000个随机点
    normals = torch.randn(batch, 3)
    normals = normals / torch.norm(normals, dim=1, keepdim=True)  # 归一化

    # 计算刻蚀速率
    etching_rates = etcher(points, normals, None)

    print(f"刻蚀速率形状: {etching_rates.shape}")
    print(f"刻蚀速率范围: [{etching_rates.min():.4f}, {etching_rates.max():.4f}]")

    # 获取通量分量进行分析
    flux_integrals, normal_comps = etcher.get_flux_components(points[:10])
    print(f"\n前10个点的通量积分:")
    print(f"integral_x: {flux_integrals[:5, 0]}")
    print(f"integral_y: {flux_integrals[:5, 1]}")

    return etcher, etching_rates


class EtchingRateVisualizer:
    """刻蚀速率可视化工具"""

    def __init__(self, calculator=None):
        """
        初始化可视化工具

        Args:
            calculator: EtchingRateCalculator实例
        """
        self.calculator = calculator

    def visualize_2d_flux_field(self, flux_field=None, save_path='flux_field_2d.png'):
        """
        可视化2D通量场（r-y平面）

        Args:
            flux_field: PositionalFluxField实例
            save_path: 保存路径
        """
        if flux_field is None and self.calculator is not None:
            flux_field = self.calculator.flux_field

        if flux_field is None:
            raise ValueError("需要提供flux_field或calculator")

        # 获取通量场数据
        r_grid, y_grid = np.meshgrid(flux_field.r_grid, flux_field.y_grid, indexing='ij')
        flux_x = flux_field.flux_x_grid
        flux_y = flux_field.flux_y_grid
        flux_magnitude = np.sqrt(flux_x ** 2 + flux_y ** 2)

        # 创建2x2的子图
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('2D flux field (r-y plane)', fontsize=16)

        # 1. integral_x 分量
        im1 = axes[0, 0].imshow(flux_x.T, extent=[flux_field.r_range[0], flux_field.r_range[1],
                                                  flux_field.y_range[0], flux_field.y_range[1]],
                                origin='lower', aspect='auto', cmap='viridis')
        axes[0, 0].set_title('integral_x')
        axes[0, 0].set_xlabel('r = sqrt(x²+z²)')
        axes[0, 0].set_ylabel('y')
        plt.colorbar(im1, ax=axes[0, 0])

        # 2. integral_y 分量
        im2 = axes[0, 1].imshow(flux_y.T, extent=[flux_field.r_range[0], flux_field.r_range[1],
                                                  flux_field.y_range[0], flux_field.y_range[1]],
                                origin='lower', aspect='auto', cmap='plasma')
        axes[0, 1].set_title('integral_y')
        axes[0, 1].set_xlabel('r = sqrt(x²+z²)')
        axes[0, 1].set_ylabel('y')
        plt.colorbar(im2, ax=axes[0, 1])

        # 3. 通量大小
        im3 = axes[1, 0].imshow(flux_magnitude.T, extent=[flux_field.r_range[0], flux_field.r_range[1],
                                                          flux_field.y_range[0], flux_field.y_range[1]],
                                origin='lower', aspect='auto', cmap='hot')
        axes[1, 0].set_title('|flux| = sqrt(integral_x² + integral_y²)')
        axes[1, 0].set_xlabel('r = sqrt(x²+z²)')
        axes[1, 0].set_ylabel('y')
        plt.colorbar(im3, ax=axes[1, 0])

        # 4. 通量方向（向量场）
        # 下采样显示向量
        step = max(1, flux_field.resolution // 20)
        r_sub = r_grid[::step, ::step]
        y_sub = y_grid[::step, ::step]
        flux_x_sub = flux_x[::step, ::step]
        flux_y_sub = flux_y[::step, ::step]

        axes[1, 1].quiver(r_sub, y_sub, flux_x_sub, flux_y_sub,
                          np.sqrt(flux_x_sub ** 2 + flux_y_sub ** 2),
                          cmap='coolwarm', angles='xy', scale_units='xy')
        axes[1, 1].set_title('Flux directions（vector field）')
        axes[1, 1].set_xlabel('r = sqrt(x²+z²)')
        axes[1, 1].set_ylabel('y')
        axes[1, 1].set_xlim(flux_field.r_range)
        axes[1, 1].set_ylim(flux_field.y_range)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"2D通量场已保存到: {save_path}")
        plt.close()

        return fig

    def visualize_3d_etching_surface(self, points=None, normals=None, save_path='etching_3d.png'):
        """
        可视化3D刻蚀表面

        Args:
            points: 点云坐标 [N, 3]
            normals: 法向量 [N, 3]
            save_path: 保存路径
        """
        if points is None:
            # 生成示例点云（圆柱表面）
            theta = np.linspace(0, 2 * np.pi, 50)
            y = np.linspace(-2, 2, 40)
            theta_grid, y_grid = np.meshgrid(theta, y)
            r = 1.0  # 圆柱半径

            # 转换为3D点
            x = r * np.cos(theta_grid).flatten()
            z = r * np.sin(theta_grid).flatten()
            y = y_grid.flatten()
            points = np.column_stack([x, y, z])

            # 生成法向量（圆柱的法向量指向径向）
            normals = np.column_stack([np.cos(theta_grid).flatten(),
                                       np.zeros_like(y_grid.flatten()),
                                       np.sin(theta_grid).flatten()])

        # 计算刻蚀速率
        if self.calculator:
            rates = self.calculator(points, normals, None)
            rates_np = rates.cpu().numpy().flatten() if torch.is_tensor(rates) else rates.flatten()
        else:
            # 如果没有计算器，使用模拟数据
            rates_np = np.exp(-0.5 * points[:, 1] ** 2)  # 高斯分布

        # 创建3D图
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 3D散点图，颜色表示刻蚀速率
        sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                        c=rates_np, cmap='jet', s=20, alpha=0.8)

        # 添加法向量箭头（只显示一部分）
        N = len(points)
        arrow_indices = np.random.choice(N, min(100, N), replace=False)

        for idx in arrow_indices:
            ax.quiver(points[idx, 0], points[idx, 1], points[idx, 2],
                      normals[idx, 0] * 0.1, normals[idx, 1] * 0.1, normals[idx, 2] * 0.1,
                      color='white', alpha=0.5, linewidth=0.5)

        ax.set_title('3D刻蚀表面可视化')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # 添加颜色条
        cbar = plt.colorbar(sc, ax=ax, shrink=0.6)
        cbar.set_label('刻蚀速率')

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"3D刻蚀表面已保存到: {save_path}")
        plt.close()

        return fig

    def visualize_histogram(self, rates, save_path='etching_histogram.png'):
        """
        可视化刻蚀速率的直方图

        Args:
            rates: 刻蚀速率数组
            save_path: 保存路径
        """
        if torch.is_tensor(rates):
            rates_np = rates.cpu().numpy().flatten()
        else:
            rates_np = rates.flatten()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 直方图
        axes[0].hist(rates_np, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        axes[0].set_title('刻蚀速率分布直方图')
        axes[0].set_xlabel('刻蚀速率')
        axes[0].set_ylabel('频率')
        axes[0].grid(True, alpha=0.3)

        # 添加统计信息
        stats_text = f"""
        统计信息:
        均值: {rates_np.mean():.4f}
        标准差: {rates_np.std():.4f}
        最小值: {rates_np.min():.4f}
        最大值: {rates_np.max():.4f}
        中位数: {np.median(rates_np):.4f}
        """
        axes[0].text(0.95, 0.95, stats_text, transform=axes[0].transAxes,
                     verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 箱线图
        axes[1].boxplot(rates_np, vert=True, patch_artist=True,
                        boxprops=dict(facecolor='lightgreen'))
        axes[1].set_title('刻蚀速率箱线图')
        axes[1].set_ylabel('刻蚀速率')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"直方图已保存到: {save_path}")
        plt.close()

        return fig

    def visualize_cross_section(self, calculator=None, plane='xz', y_value=0.0,
                                save_path='cross_section.png'):
        """
        可视化截面上的刻蚀速率

        Args:
            calculator: 刻蚀速率计算器
            plane: 截面平面 ('xz', 'xy', 'yz')
            y_value: 截面的y值（用于xz平面）
            save_path: 保存路径
        """
        if calculator is None:
            calculator = self.calculator

        if calculator is None:
            raise ValueError("需要提供calculator")

        # 创建网格
        if plane == 'xz':
            # xz平面，固定y值
            x = np.linspace(-2, 2, 100)
            z = np.linspace(-2, 2, 100)
            X, Z = np.meshgrid(x, z)
            Y = np.full_like(X, y_value)

            # 创建点云
            points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

            # 计算法向量（假设表面垂直）
            normals = np.tile([0, 1, 0], (len(points), 1))

        elif plane == 'xy':
            # xy平面，固定z值
            x = np.linspace(-2, 2, 100)
            y = np.linspace(-2, 2, 100)
            X, Y = np.meshgrid(x, y)
            Z = np.zeros_like(X)

            points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
            normals = np.tile([0, 0, 1], (len(points), 1))

        else:  # yz平面
            y = np.linspace(-2, 2, 100)
            z = np.linspace(-2, 2, 100)
            Y, Z = np.meshgrid(y, z)
            X = np.zeros_like(Y)

            points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
            normals = np.tile([1, 0, 0], (len(points), 1))

        # 计算刻蚀速率
        rates = calculator(points, normals, None)
        rates_np = rates.cpu().numpy().flatten() if torch.is_tensor(rates) else rates.flatten()

        # 重塑为网格形状
        rates_grid = rates_np.reshape(X.shape)

        # 绘制热图
        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(rates_grid, extent=[x.min(), x.max(), z.min(), z.max()] if plane == 'xz' else
        [x.min(), x.max(), y.min(), y.max()] if plane == 'xy' else
        [y.min(), y.max(), z.min(), z.max()],
                       origin='lower', cmap='viridis', aspect='auto')

        ax.set_title(f'{plane.upper()}平面截面 (y={y_value if plane == "xz" else "0"})')
        ax.set_xlabel('X' if plane in ['xz', 'xy'] else 'Y')
        ax.set_ylabel('Z' if plane in ['xz', 'yz'] else 'Y')

        plt.colorbar(im, ax=ax, label='刻蚀速率')

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"截面图已保存到: {save_path}")
        plt.close()

        return fig

    def create_interactive_html(self, points, normals, rates, save_path='etching_interactive.html'):
        """
        创建交互式HTML可视化（使用plotly，可在浏览器中查看）

        Args:
            points: 点云坐标
            normals: 法向量
            rates: 刻蚀速率
            save_path: 保存路径
        """
        # 转换为numpy数组
        if torch.is_tensor(points):
            points_np = points.cpu().numpy()
        else:
            points_np = points

        if torch.is_tensor(normals):
            normals_np = normals.cpu().numpy()
        else:
            normals_np = normals

        if torch.is_tensor(rates):
            rates_np = rates.cpu().numpy().flatten()
        else:
            rates_np = rates.flatten()

        # 下采样以提高性能
        N = len(points_np)
        if N > 5000:
            indices = np.random.choice(N, 5000, replace=False)
            points_np = points_np[indices]
            normals_np = normals_np[indices]
            rates_np = rates_np[indices]

        # 创建交互式3D图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('3D刻蚀表面', 'X-Y投影', 'X-Z投影', 'Y-Z投影'),
            specs=[[{'type': 'scene'}, {'type': 'scatter'}],
                   [{'type': 'scatter'}, {'type': 'scatter'}]]
        )

        # 1. 3D散点图
        fig.add_trace(
            go.Scatter3d(
                x=points_np[:, 0], y=points_np[:, 1], z=points_np[:, 2],
                mode='markers',
                marker=dict(
                    size=3,
                    color=rates_np,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="刻蚀速率", x=1.0)
                ),
                name='刻蚀表面'
            ),
            row=1, col=1
        )

        # 2. X-Y投影
        fig.add_trace(
            go.Scatter(
                x=points_np[:, 0], y=points_np[:, 1],
                mode='markers',
                marker=dict(
                    size=3,
                    color=rates_np,
                    colorscale='Viridis',
                    showscale=False
                ),
                name='X-Y投影'
            ),
            row=1, col=2
        )

        # 3. X-Z投影
        fig.add_trace(
            go.Scatter(
                x=points_np[:, 0], y=points_np[:, 2],
                mode='markers',
                marker=dict(
                    size=3,
                    color=rates_np,
                    colorscale='Viridis',
                    showscale=False
                ),
                name='X-Z投影'
            ),
            row=2, col=1
        )

        # 4. Y-Z投影
        fig.add_trace(
            go.Scatter(
                x=points_np[:, 1], y=points_np[:, 2],
                mode='markers',
                marker=dict(
                    size=3,
                    color=rates_np,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="刻蚀速率", x=1.3)
                ),
                name='Y-Z投影'
            ),
            row=2, col=2
        )

        # 更新布局
        fig.update_layout(
            height=800,
            showlegend=False,
            title_text="刻蚀速率交互式可视化"
        )

        # 更新轴标签
        fig.update_xaxes(title_text="X", row=1, col=2)
        fig.update_yaxes(title_text="Y", row=1, col=2)
        fig.update_xaxes(title_text="X", row=2, col=1)
        fig.update_yaxes(title_text="Z", row=2, col=1)
        fig.update_xaxes(title_text="Y", row=2, col=2)
        fig.update_yaxes(title_text="Z", row=2, col=2)

        # 保存为HTML文件
        fig.write_html(save_path)
        print(f"交互式HTML可视化已保存到: {save_path}")

        return fig

    def visualize_all(self, points=None, normals=None, save_dir='etching_visualizations'):
        """
        执行所有可视化

        Args:
            points: 点云坐标
            normals: 法向量
            save_dir: 保存目录
        """
        import os
        os.makedirs(save_dir, exist_ok=True)

        print("开始执行刻蚀速率可视化...")

        # 1. 可视化2D通量场
        try:
            self.visualize_2d_flux_field(save_path=f'{save_dir}/flux_field_2d.png')
        except Exception as e:
            print(f"2D通量场可视化失败: {e}")

        # 2. 生成示例数据（如果没有提供）
        if points is None:
            # 生成球面点云
            phi = np.linspace(0, np.pi, 30)
            theta = np.linspace(0, 2 * np.pi, 60)
            phi_grid, theta_grid = np.meshgrid(phi, theta)

            r = 1.5  # 球半径
            x = r * np.sin(phi_grid) * np.cos(theta_grid)
            y = r * np.sin(phi_grid) * np.sin(theta_grid)
            z = r * np.cos(phi_grid)

            points = np.column_stack([x.flatten(), y.flatten(), z.flatten()])

            # 球面法向量（指向球心）
            normals = -points / np.linalg.norm(points, axis=1, keepdims=True)

        # 3. 计算刻蚀速率
        if self.calculator:
            rates = self.calculator(points, normals, None)
        else:
            # 模拟数据
            rates = np.exp(-0.3 * (points[:, 0] ** 2 + points[:, 2] ** 2))
            rates = torch.from_numpy(rates).float().view(-1, 1)

        # 4. 可视化3D刻蚀表面
        try:
            self.visualize_3d_etching_surface(
                points, normals, save_path=f'{save_dir}/etching_3d.png'
            )
        except Exception as e:
            print(f"3D刻蚀表面可视化失败: {e}")

        # 5. 可视化直方图
        try:
            self.visualize_histogram(rates, save_path=f'{save_dir}/etching_histogram.png')
        except Exception as e:
            print(f"直方图可视化失败: {e}")

        # 6. 可视化截面
        try:
            self.visualize_cross_section(
                plane='xz', y_value=-0.1, save_path=f'{save_dir}/cross_section_xz.png'
            )
        except Exception as e:
            print(f"截面可视化失败: {e}")

        # 7. 创建交互式HTML（可选）
        try:
            self.create_interactive_html(
                points, normals, rates, save_path=f'{save_dir}/etching_interactive.html'
            )
        except Exception as e:
            print(f"交互式HTML可视化失败: {e}")
            print("你可能需要安装plotly: pip install plotly")

        print(f"所有可视化结果已保存到: {save_dir}")

        # 显示汇总信息
        rates_np = rates.cpu().numpy().flatten() if torch.is_tensor(rates) else rates.flatten()
        print("\n刻蚀速率统计:")
        print(f"  均值: {rates_np.mean():.6f}")
        print(f"  标准差: {rates_np.std():.6f}")
        print(f"  最小值: {rates_np.min():.6f}")
        print(f"  最大值: {rates_np.max():.6f}")
        print(f"  中位数: {np.median(rates_np):.6f}")


# 设置中文字体
def setup_chinese_font():
    """设置中文字体，避免中文显示问题"""
    # 获取系统字体路径
    system_fonts = []

    # Windows系统
    if os.name == 'nt':
        import winreg
        try:
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts")
            for i in range(winreg.QueryInfoKey(key)[1]):
                name, value, _ = winreg.EnumValue(key, i)
                if 'ttf' in value.lower() or 'ttc' in value.lower():
                    font_path = os.path.join(r"C:\Windows\Fonts", value)
                    if os.path.exists(font_path):
                        system_fonts.append(font_path)
        except:
            pass

    # Linux系统
    elif os.name == 'posix':
        font_dirs = [
            '/usr/share/fonts',
            '/usr/local/share/fonts',
            os.path.expanduser('~/.fonts'),
            os.path.expanduser('~/.local/share/fonts')
        ]
        for font_dir in font_dirs:
            if os.path.exists(font_dir):
                for root, dirs, files in os.walk(font_dir):
                    for file in files:
                        if file.endswith(('.ttf', '.ttc', '.otf')):
                            system_fonts.append(os.path.join(root, file))

    # macOS系统
    else:
        font_dirs = [
            '/System/Library/Fonts',
            '/Library/Fonts',
            os.path.expanduser('~/Library/Fonts')
        ]
        for font_dir in font_dirs:
            if os.path.exists(font_dir):
                for root, dirs, files in os.walk(font_dir):
                    for file in files:
                        if file.endswith(('.ttf', '.ttc', '.otf')):
                            system_fonts.append(os.path.join(root, file))

    # 优先选择中文字体
    chinese_fonts = []
    for font_path in system_fonts:
        font_name = os.path.basename(font_path).lower()
        # 常见的中文字体名称
        if any(keyword in font_name for keyword in [
            'simhei', 'simkai', 'simsun', 'microsoftyahei',
            'msyh', 'pingfang', 'heiti', 'stkaiti', 'stsong'
        ]):
            chinese_fonts.append(font_path)

    if chinese_fonts:
        # 添加字体
        matplotlib.font_manager.fontManager.addfont(chinese_fonts[0])
        font_name = matplotlib.font_manager.FontProperties(fname=chinese_fonts[0]).get_name()
        matplotlib.rcParams['font.sans-serif'] = [font_name]
        matplotlib.rcParams['axes.unicode_minus'] = False
        print(f"已设置中文字体: {font_name}")
    else:
        print("未找到中文字体，使用默认字体")


# 使用示例
def main():
    # 1. 创建通量场（如果需要）
    flux_field = PositionalFluxField(
        cache_file='models/flux_field_cache.pkl',
        r_range=(0, radius * 2),
        y_range=(config.region_y[0], config.region_y[1]),
        resolution=160
    )

    # 预计算通量场（如果缓存不存在）
    if not os.path.exists('models/flux_field_cache.pkl'):
        print("正在计算通量场...")
        flux_field.precompute_flux_field(n_processes=4)
    else:
        flux_field.load_cache()

    # 2. 创建刻蚀速率计算器
    calculator = SeparatedEtchingRateCalculator(
        etching_type='integral',
        coefficient=0.8,
        flux_field=flux_field
    )

    # 3. 创建可视化工具
    setup_chinese_font()
    visualizer = EtchingRateVisualizer(calculator=calculator)

    # 4. 执行所有可视化
    visualizer.visualize_all(save_dir='visualizations/flux_field')

    print("可视化完成！")
    print("检查 'visualizations/flux_field' 文件夹中的输出文件。")


# 简单的命令行界面
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='刻蚀速率可视化工具')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'flux', '3d', 'hist', 'cross', 'html'],
                        help='可视化模式')
    parser.add_argument('--save_dir', type=str, default='etching_visualizations',
                        help='保存目录')

    args = parser.parse_args()

    # 运行主函数
    main()