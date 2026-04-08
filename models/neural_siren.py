import torch
import torch.nn as nn
import numpy as np



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


class SwishLayer(nn.Module):
    def __init__(self, in_features, out_features,  beta=1.0):
        super().__init__()
        self.beta = beta
        self.linear = nn.Linear(in_features, out_features)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        z = self.linear(x)
        return z * torch.sigmoid(self.beta * z)

class SpaceTimeSIREN(nn.Module):
    """时空神经隐式函数"""

    def __init__(self, hidden_layers=4, hidden_dim=256, r = 0.25, alpha = 2.0):
        super().__init__()
        self.hidden_layers = hidden_layers
        self.hidden_dim = hidden_dim
        self.radius = r
        self.alpha =alpha

        # 输入: (x, y, z, t) - 4维
        layers = [SineLayer(4, hidden_dim)]

        # 隐藏层
        for _ in range(hidden_layers - 1):
            layers.append(SwishLayer(hidden_dim, hidden_dim))

        # 输出层
        self.network = nn.Sequential(*layers)
        self.final_layer = nn.Linear(hidden_dim, 1)

        # 输出层初始化
        with torch.no_grad():
            bound = np.sqrt(6 / hidden_dim) / 30
            self.final_layer.weight.uniform_(-bound, bound)
            self.final_layer.bias.uniform_(-bound, bound)



    def initial_condition(self, x):
        """
        初始条件函数: I(x) = y - y_0

        参数:
            x: 输入张量 [batch_size, 4]，其中最后一维是 (x, y, z, t)

        返回:
            初始条件值
        """
        y_0 = -0.7
        y_coord = x[:, 1:2]  # y坐标
        return y_coord - y_0

    def boundary_condition(self, x , flag = '0'):  # 根据不同速率模型设计不同边界硬约束，对应不同演化趋势
        x_coord = x[:, 0:1]  # x坐标
        y_coord = x[:, 1:2]  # y坐标
        z_coord = x[:, 2:3]  # z坐标
        # 边值约束
        radius = self.radius #开口半径
        y_0 =-0.7
        if flag =='0':
            distance_from_center = torch.sqrt(x_coord ** 2 + z_coord ** 2)/radius
            # 平滑过渡函数
            # 当distance_from_center >= 1时为0，<1时为平滑过渡
            transition_mask = distance_from_center < 1
            transition_func = torch.where(
                transition_mask,
                1 - 0.4*distance_from_center ** 8,                  # 不一定需要平滑过渡，幂次可调整
                torch.zeros_like(distance_from_center)
            )
        else:
            # 计算指示函数
            f = y_coord-y_0 - self.alpha * (x_coord ** 2 + z_coord ** 2 - radius ** 2)
            #f = torch.abs(y_coord - y_0) - self.alpha*(torch.sqrt(x_coord ** 2 + z_coord ** 2)- radius)
            beta = 50.0  # beta越大，过渡越陡峭，接近阶跃函数
            bias = -radius * 0.01
            # 结合where确保f<0时为0
            transition_func = torch.where(
                f < bias,
                torch.zeros_like(f),
                torch.sigmoid(beta * f )
            )
            # #f = y_coord - y_0 - 1.5* (x_coord ** 2 + z_coord ** 2- radius ** 2)
            # bias =radius * 0.04
            # f = y_coord - y_0 - 1.0*(torch.sqrt(x_coord ** 2 + z_coord ** 2)- radius-bias)
            # delta = 0.1  # 过渡区间的宽度，可调整
            # transition_func = torch.where(
            #     f <= 0,
            #     torch.zeros_like(f),
            #     torch.where(
            #         f < delta,
            #         3 * (f / delta) ** 2 - 2 * (f / delta) ** 3,  # 三次Hermite插值，f'(0)=f'(delta)=0
            #         torch.ones_like(f)
            #     )
            # )


        return transition_func

    def forward(self, x):
        # 分离坐标分量
        x_coord = x[:, 0:1]  # x坐标
        y_coord = x[:, 1:2]  # y坐标
        z_coord = x[:, 2:3]  # z坐标
        t_coord = x[:, 3:4]  # 时间坐标
        # # 神经网络输出 u
        features = self.network(x)
        u = self.final_layer(features)
        # 构造关于x取反的输入
        # x_neg = torch.cat([-x_coord, y_coord, -z_coord, t_coord], dim=1)
        #
        # # 分别计算原始输入和取反输入的网络输出
        # u_pos = self.final_layer(self.network(x))
        # u_neg = self.final_layer(self.network(x_neg))
        #
        # # 平均
        # u = (u_pos + u_neg) / 2

        # 应用初值硬约束: I(x) + (t - T_0) * u
        I_x = self.initial_condition(x)
        # 边值约束
        B_x = self.boundary_condition(x , flag='1') # flag 取值0或1
        #t_p = (torch.cos(t_coord*0.5)+1)/2
        constrained_output = torch.exp(- t_coord*2)*I_x + t_coord * u * B_x

        return constrained_output
    def save(self, filepath):
        """保存模型"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'hidden_layers': self.hidden_layers,
            'hidden_dim': self.hidden_dim
        }, filepath)
        print(f"模型已保存到: {filepath}")

    @classmethod
    def load(cls, filepath, device='cpu'):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=device)
        model = cls(
            hidden_layers=checkpoint['hidden_layers'],
            hidden_dim=checkpoint['hidden_dim']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        print(f"模型已从 {filepath} 加载")
        return model