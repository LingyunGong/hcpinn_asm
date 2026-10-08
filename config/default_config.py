class TrainingConfig:
    """训练配置参数"""

    def __init__(self):
        # 模型参数
        self.hidden_layers = 4
        self.hidden_dim = 256
        # 硬约束参数
        self.alpha = 1.0
        self.t_end = 2.0
        # 训练参数
        self.num_epochs = 2000
        self.batch_size = 10000
        self.save_interval = 1000

        # 损失权重
        self.lambda_data = 0.0
        self.lambda_pde = 1.1
        self.lambda_eikonal = 0.2
        self.lambda_temporal_data = 0.0

        # Physical parameters (simplified)
        self.etching_type = 'integral'  # isotropic, anisotropic, reflect, stochastic ,integral
        self.radius = 0.25  # 沟槽开口半宽（模型单位）
        self.h = 1.6  # 掩膜高/开口直径比 -> 掩膜高 = 2*radius*h = 0.8（模型单位）
        self.sigma = 0.02 # 论文 Table 1: sigma（离子角分布宽度）0.02 (0.01--0.04)
        self.side_p = 0.1 # Side wall reaction parameter - side wall protection  0.01
        self.rate = 8 # 论文 Table 1: k_i / k_i0（离子通道速率常数）8 (4--16)

        # 可视化参数
        self.visualization_resolution = 80
        self.region_x =(-2*self.radius, 2*self.radius) # Set the Computational domain, in which we precompute the rate
        self.region_y =(-1.0, 2.2)