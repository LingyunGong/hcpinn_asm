import torch
import numpy as np


class InitialSphere:
    """初始球体表面"""

    def __init__(self, radius=0.8):
        self.radius = radius
        self.bounding_box = [0, 3]

    def sample_points(self, n_points):
        """在球面上均匀采样点"""
        # 使用球面均匀采样
        points = torch.randn(n_points, 3)
        points = points / torch.norm(points, dim=1, keepdim=True) * self.radius
        # 添加球心偏移
        center = torch.tensor([1.5, 1.5, 1.5], device=points.device, dtype=points.dtype)
        points = points + center
        return points

    def sdf_values(self, points):
        """计算点到球面的符号距离函数（球心在(1.5,1.5,1.5)）"""
        # 定义球心
        center = torch.tensor([1.5, 1.5, 1.5], device=points.device, dtype=points.dtype)

        # 计算点到球心的距离，然后减去半径
        return torch.norm(points - center, dim=1, keepdim=True) - self.radius

class InitialPlane2:
    """初始平面表面（基于传统水平集初始化）"""

    def __init__(self, y_interface=-0.7):
        self.y_interface = y_interface
        self.range = 0.1

    def sample_points(self, n_points):
        """在平面上采样点"""
        # 均匀采样
        x_coords = (2*torch.rand(n_points, 1)-1)* self.range
        z_coords = (2*torch.rand(n_points, 1)-1)* self.range
        y_coords = torch.ones(n_points, 1) * self.y_interface #+(2*torch.rand(n_points, 1)-1) * 0.05

        points = torch.cat([x_coords, y_coords, z_coords], dim=1)
        return points

    def sdf_values(self, points):
        """计算点到平面的符号距离函数"""
        # 平面方程: y = y_interface
        # SDF: y - y_interface
        return points[:, 1:2] - self.y_interface




class InitialSurfaceFromArray:
    """从numpy数组初始化表面"""

    def __init__(self, phi_array, bbox=[0, 3.0, 0, 3.0, 0, 6.0]):
        """
        phi_array: 3D数组，表示初始水平集函数
        bbox: [x_min, x_max, y_min, y_max, z_min, z_max]
        """
        self.phi_array = phi_array
        self.bbox = bbox
        self.shape = phi_array.shape

    def sample_points(self, n_points):
        """在零等值面附近采样点"""
        from skimage import measure

        # 提取零等值面
        vertices, faces, _, _ = measure.marching_cubes(self.phi_array, level=0)

        # 将顶点坐标缩放到实际物理坐标
        scale_x = (self.bbox[1] - self.bbox[0]) / (self.shape[0] - 1)
        scale_y = (self.bbox[3] - self.bbox[2]) / (self.shape[1] - 1)
        scale_z = (self.bbox[5] - self.bbox[4]) / (self.shape[2] - 1)

        vertices[:, 0] = vertices[:, 0] * scale_x + self.bbox[0]
        vertices[:, 1] = vertices[:, 1] * scale_y + self.bbox[2]
        vertices[:, 2] = vertices[:, 2] * scale_z + self.bbox[4]

        # 从顶点中随机采样
        if len(vertices) > n_points:
            indices = np.random.choice(len(vertices), n_points, replace=False)
            points = vertices[indices]
        else:
            points = vertices
            # 如果需要更多点，重复采样
            if len(points) < n_points:
                additional_points = vertices[np.random.choice(len(vertices), n_points - len(points))]
                points = np.vstack([points, additional_points])

        return torch.tensor(points).float()

    def sdf_values(self, points):
        """通过插值计算SDF值"""
        points_np = points.detach().numpy()

        # 计算网格坐标
        xi = (points_np[:, 0] - self.bbox[0]) / (self.bbox[1] - self.bbox[0]) * (self.shape[0] - 1)
        yi = (points_np[:, 1] - self.bbox[2]) / (self.bbox[3] - self.bbox[2]) * (self.shape[1] - 1)
        zi = (points_np[:, 2] - self.bbox[4]) / (self.bbox[5] - self.bbox[4]) * (self.shape[2] - 1)

        # 三线性插值
        from scipy.ndimage import map_coordinates
        coords = np.array([xi, yi, zi])
        sdf_values = map_coordinates(self.phi_array, coords, order=1, mode='nearest')

        return torch.tensor(sdf_values).float().unsqueeze(1)