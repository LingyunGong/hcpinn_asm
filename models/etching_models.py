import torch
import pickle
import numpy as np
import torch
from config.default_config import TrainingConfig
from multiprocessing import Pool, cpu_count
import os
config=TrainingConfig()
class PositionalFluxField:
    """位置通量场计算器（只计算位置相关的通量积分）"""

    def __init__(self, cache_file='models/flux_field_cache.pkl',
                 r_range=(0, 2* config.radius), y_range=(-1, 5), resolution=200):
        """
        初始化通量场计算器

        Args:
            cache_file: 缓存文件名
            r_range: r坐标范围
            y_range: y坐标范围
            resolution: 网格分辨率
        """
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

        # 尝试加载缓存，如果失败则提示用户
        if not self.load_cache():
            print(f"错误：缓存文件 {self.cache_file} 不存在或损坏！")
            print("请先运行 'rate_precompute.py' 生成通量场缓存文件。")
            raise FileNotFoundError(f"缓存文件 {self.cache_file} 未找到，请先运行预计算脚本。")

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
            raise ValueError("通量场数据未加载，请检查缓存文件")

        from scipy.interpolate import LinearNDInterpolator

        # 创建网格点
        R, Y = np.meshgrid(self.r_grid, self.y_grid, indexing='ij')
        points_grid = np.column_stack([R.ravel(), Y.ravel()])

        # 创建插值器
        self.interpolator_x = LinearNDInterpolator(points_grid, self.flux_x_grid.ravel())
        self.interpolator_y = LinearNDInterpolator(points_grid, self.flux_y_grid.ravel())

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

                return True
            except Exception as e:
                print(f"缓存文件损坏: {e}")
                return False
        return False

class EtchingRateModel:
    """刻蚀速率物理模型"""

    def __init__(self, etching_type='integral', coefficient=0.8,
                 flux_field=None, n_processes=None):
        """
        初始化刻蚀速率计算器

        Args:
            etching_type: 刻蚀类型
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

    def __call__(self, points, normals, curvatures=None, t=None):
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

        elif self.etching_type == 'reflect':
            directional_effect = 0.8

            # 计算xz平面上的模长
            distance_from_center = torch.sqrt((points[:, 0:1]) ** 2 + (points[:, 2:3]) ** 2)

            # 使用构造的光滑样条函数作为过渡函数
            r = torch.abs(distance_from_center)

            # 计算第一区域的值
            region1_value = 1.0  + (20 / 81) * r ** 2

            # 计算第二区域的值
            one_minus_r = 1 - r
            region2_value = (3280 / 9) * one_minus_r ** 2 - (22000 / 9) * one_minus_r ** 3   # +0.1 使得在边界不光滑

            # 组合结果：当r≤0.9时使用第一区域，当0.9<r≤1时使用第二区域，当r>1时为0
            transition_mask_region1 = (r <= 0.9)
            transition_mask_region2 = (r > 0.9) & (r <= 1.0)

            transition_func = torch.zeros_like(distance_from_center)
            transition_func = torch.where(transition_mask_region1, region1_value, transition_func)
            transition_func = torch.where(transition_mask_region2, region2_value, transition_func)

            # 应用过渡函数
            return directional_effect * transition_func

        elif self.etching_type == 'anisotropic':
            directional_effect = 0.49
            # 计算xz平面上的模长
            distance_from_center = torch.sqrt((points[:, 0:1]) ** 2 + (points[:, 2:3]) ** 2)
            # 平滑过渡函数 (三次样条)
            # 当distance_from_center >= 1时为0，<1时为平滑过渡
            transition_mask = distance_from_center < 1.0
            transition_func = torch.where(
                transition_mask,
                1 - 3 * distance_from_center ** 2 + 2 * distance_from_center ** 3,
                torch.zeros_like(distance_from_center)
            )
            # 应用过渡函数
            return directional_effect* transition_func

        else:
            raise ValueError(f"未知的刻蚀类型: {self.etching_type}")



    def save(self, filepath):
        """保存刻蚀模型"""
        with open(filepath, 'wb') as f:
            pickle.dump({'etching_type': self.etching_type}, f)
        print(f"刻蚀模型已保存到: {filepath}")

    @classmethod
    def load(cls, filepath):
        """加载刻蚀模型"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        model = cls(etching_type=data['etching_type'])
        print(f"刻蚀模型已从 {filepath} 加载")
        return model