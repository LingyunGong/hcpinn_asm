"""硬约束 PINN 网络架构：SpaceTimeSIREN（时空神经隐式函数）"""

import torch
import torch.nn as nn
import numpy as np


class SineLayer(nn.Module):
    """SIREN 正弦激活层"""

    def __init__(self, in_features, out_features, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)
        with torch.no_grad():
            bound = np.sqrt(6 / in_features) / omega_0
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class SwishLayer(nn.Module):
    def __init__(self, in_features, out_features, beta=1.0):
        super().__init__()
        self.beta = beta
        self.linear = nn.Linear(in_features, out_features)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        z = self.linear(x)
        return z * torch.sigmoid(self.beta * z)


class SpaceTimeSIREN(nn.Module):
    """时空神经隐式函数：输入 (x,y,z,t) → 输出 硬约束水平集值 φ"""

    def __init__(self, hidden_layers=4, hidden_dim=256, r=0.25, alpha=2.0):
        super().__init__()
        self.hidden_layers = hidden_layers
        self.hidden_dim = hidden_dim
        self.radius = r
        self.alpha = alpha

        # 输入 4 维 (x, y, z, t)
        layers = [SineLayer(4, hidden_dim)]
        for _ in range(hidden_layers - 1):
            layers.append(SwishLayer(hidden_dim, hidden_dim))
        self.network = nn.Sequential(*layers)
        self.final_layer = nn.Linear(hidden_dim, 1)

        with torch.no_grad():
            bound = np.sqrt(6 / hidden_dim) / 30
            self.final_layer.weight.uniform_(-bound, bound)
            self.final_layer.bias.uniform_(-bound, bound)

    def initial_condition(self, x):
        """初始条件 I(x) = y - y_0"""
        y_0 = -0.7
        return x[:, 1:2] - y_0

    def boundary_condition(self, x, flag='1'):
        """边值硬约束：根据 flag 选择过渡函数"""
        x_coord = x[:, 0:1]
        y_coord = x[:, 1:2]
        z_coord = x[:, 2:3]

        if flag == '0':
            distance = torch.sqrt(x_coord ** 2 + z_coord ** 2) / self.radius
            mask = distance < 1
            trans = torch.where(mask, 1 - 0.4 * distance ** 8, torch.zeros_like(distance))
        else:
            y_0 = -0.7
            f = y_coord - y_0 - self.alpha * (x_coord ** 2 + z_coord ** 2 - self.radius ** 2)
            bias = -self.radius * 0.01
            trans = torch.where(f < bias, torch.zeros_like(f), torch.sigmoid(50.0 * f))

        return trans

    def forward(self, x):
        """硬约束前向传播：φ(x,t) = e^{-2t}·I(x) + t·u·B(x)"""
        x_coord = x[:, 0:1]
        y_coord = x[:, 1:2]
        z_coord = x[:, 2:3]
        t_coord = x[:, 3:4]

        features = self.network(x)
        u = self.final_layer(features)

        I_x = self.initial_condition(x)
        B_x = self.boundary_condition(x, flag='1')

        constrained_output = torch.exp(-t_coord * 2) * I_x + t_coord * u * B_x
        return constrained_output

    def save(self, filepath):
        torch.save({
            'model_state_dict': self.state_dict(),
            'hidden_layers': self.hidden_layers,
            'hidden_dim': self.hidden_dim
        }, filepath)
        print(f"模型已保存到: {filepath}")

    @classmethod
    def load(cls, filepath, device='cpu'):
        checkpoint = torch.load(filepath, map_location=device)
        model = cls(
            hidden_layers=checkpoint['hidden_layers'],
            hidden_dim=checkpoint['hidden_dim']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        print(f"模型已从 {filepath} 加载")
        return model