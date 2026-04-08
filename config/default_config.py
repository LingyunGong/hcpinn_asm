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
        self.radius = 0.25
        self.h = 1.6  # Mask height / diameter of opening
        self.sigma = 0.02 # Parameter of Gauss function (Angle distribution)
        self.side_p = 0.1 # Side wall reaction parameter - side wall protection  0.01
        self.rate = 8 # Rate strength

        # 可视化参数
        self.visualization_resolution = 80
        self.region_x =(-2*self.radius, 2*self.radius) # Set the Computational domain, in which we precompute the rate
        self.region_y =(-1.0, 2.2)