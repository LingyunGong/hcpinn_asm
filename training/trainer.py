import torch
import os
import time
import numpy as np
import pickle
import h5py


class EtchingTrainer:
    """刻蚀模拟训练器"""

    def __init__(self, model, loss_fn, etching_rate_model, temporal_data_path=None, r = 0.25, alpha = 2.0, rej ='1'):
        self.model = model
        self.loss_fn = loss_fn
        self.etching_rate_model = etching_rate_model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.radius = r # 开口半径
        self.alpha = alpha
        #self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        self.rej =rej # 是否拒绝采样 '1','0' yes & no
        self.temporal_data_path = temporal_data_path

        # 如果提供了时序数据路径，则加载数据
        if temporal_data_path and os.path.exists(temporal_data_path):
            self.temporal_data = self.load_temporal_data(temporal_data_path)
            print(f"已加载时序数据: {temporal_data_path}")
            print(f"时间步数量: {len(self.temporal_data['sdf_data'])}")
        else:
            self.temporal_data = None
            print("未使用时序数据")

        # 移除 verbose 参数以兼容不同版本的 PyTorch
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=500, factor=0.5
        )

    def load_temporal_data(self, data_path):
        """加载时序数据"""
        evolution_data = {
            'metadata': {},
            'sdf_data': {}
        }

        with h5py.File(data_path, 'r') as f:
            # 加载元数据
            meta_grp = f['metadata']
            for key in meta_grp.attrs:
                evolution_data['metadata'][key] = meta_grp.attrs[key]

            for key in meta_grp:
                evolution_data['metadata'][key] = meta_grp[key][()]

            # 加载SDF数据
            sdf_grp = f['sdf_data']
            for time_key in sdf_grp:
                evolution_data['sdf_data'][time_key] = sdf_grp[time_key][()]

        return evolution_data

    def sample_from_temporal_data(self, num_samples):
        """从时序数据中采样"""
        if self.temporal_data is None:
            return None, None, None

        grid_shape = self.temporal_data['metadata']['grid_shape']
        domain_bounds = self.temporal_data['metadata']['domain_bounds']

        # 生成网格坐标
        x = np.linspace(domain_bounds[0][0], domain_bounds[0][1], grid_shape[0])
        y = np.linspace(domain_bounds[1][0], domain_bounds[1][1], grid_shape[1])
        z = np.linspace(domain_bounds[2][0], domain_bounds[2][1], grid_shape[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        coordinates = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

        # 收集所有时空点和对应的SDF值
        all_points = []
        all_times = []
        all_sdf_values = []

        for time_key, sdf_array in self.temporal_data['sdf_data'].items():
            t_value = float(time_key[1:])

            # 为每个时间点随机采样
            time_indices = np.random.choice(
                coordinates.shape[0],
                num_samples // len(self.temporal_data['sdf_data']),
                replace=False
            )

            all_points.append(coordinates[time_indices])
            all_times.append(np.full((len(time_indices), 1), t_value))
            all_sdf_values.append(sdf_array.ravel()[time_indices])

        # 合并所有时空数据
        space_time_points = torch.FloatTensor(np.vstack(all_points))
        space_time_t = torch.FloatTensor(np.vstack(all_times))
        space_time_sdf = torch.FloatTensor(np.concatenate(all_sdf_values))

        return space_time_points, space_time_t, space_time_sdf

    def sample_pde_points(self, n_points, time_interval=(0, 1.5)):
        """专门为PDE损失采样的点"""
        # 时空域采样
        #时间均匀采样
        # t_min, t_max = time_interval
        # times = torch.rand(n_points, 1) * (t_max - t_min) + t_min
        """
            采样服从分布 f(t) ∝ exp(4λ(t_max - t)/3) 的时间点
        """
        t_min, t_max = time_interval
        n_candidates = 2
        # 步骤1：生成[0,1]均匀分布的随机数
        u = torch.rand(n_points* n_candidates, 1)

        # 步骤2：计算累积分布函数的逆函数
        # 对于分布 f(t) ∝ exp(4λ(t_max - t)/3)，其CDF的逆函数为：
        # t = t_max - (3/(4λ)) * ln(1 + (exp(4λΔt/3) - 1) * (1 - u))
        # 其中 Δt = t_max - t_min
        delta_t = t_max - t_min
        lambd = torch.tensor(1.0)
        exp_term = torch.exp(4 * lambd * delta_t / 3) - 1
        # 计算采样时间
        times = t_max - (3 / (4 * lambd)) * torch.log(1 + exp_term * (1 - u))

        if self.rej == '0':
            theta = torch.rand(n_points, 1) * 2 * np.pi
            r = torch.sqrt(torch.rand(n_points, 1)) * (self.radius * 1.1)

            x_coords = r * torch.cos(theta)
            z_coords = r * torch.sin(theta)
            lb = -1.0  # 采样下界
            y_coords = torch.rand(n_points, 1) * 2.0 + lb
            spatial_points = torch.cat([x_coords, y_coords, z_coords], dim=1)
        else:
            # 生成更多候选点
            n_total = n_points * n_candidates

            # theta = torch.rand(n_total, 1) * 2 * np.pi
            # r = torch.sqrt(torch.rand(n_total, 1)) * (self.radius * 2.0)
            #
            # x_coords = r * torch.cos(theta)
            # z_coords = r * torch.sin(theta)
            x_coords = self.radius * 2.0 * (2 * torch.rand(n_total, 1) - 1)
            z_coords = self.radius * 2.0 * (2 * torch.rand(n_total, 1) - 1)
            lb = -0.8 - self.alpha * self.radius ** 2 # 采样下界
            y_coords = torch.rand(n_total, 1)* (0.8+ times)  + lb

            # 计算f值
            y_0 = -0.7
            f = y_coords - y_0 - self.alpha * (x_coords ** 2 + z_coords ** 2 - self.radius ** 2)
            #f = torch.abs(y_coords - y_0) - self.alpha * (torch.sqrt(x_coords ** 2 + z_coords ** 2) - self.radius)
            # 筛选满足条件的点
            mask = f > -self.radius * 0.04
            valid_points = torch.cat([x_coords, y_coords, z_coords], dim=1)[mask.squeeze()]

            spatial_points = valid_points[:n_points]
        time_points = times[:n_points]


        # 需要梯度计算
        spatial_points = spatial_points.clone().detach().requires_grad_(True)
        time_points =  time_points.clone().detach().requires_grad_(True)

        return spatial_points,  time_points
    def sample_training_batch(self, initial_surface, batch_size=10000, time_interval=(0, 1.8)):
        batch = {}

        # Initial condition sampling
        n_initial = batch_size * 1 // 10000  # Unused

        if hasattr(initial_surface, 'sample_points'):
            initial_points = initial_surface.sample_points(n_initial)
            if isinstance(initial_points, torch.Tensor):
                initial_points = initial_points.clone().detach().requires_grad_(True)
            else:
                initial_points = torch.tensor(initial_points).float().requires_grad_(True)

            initial_sdf = initial_surface.sdf_values(initial_points)
        else:
            idx = np.random.choice(len(initial_surface), n_initial)
            initial_points = torch.tensor(initial_surface[idx]).float().requires_grad_(True)
            initial_sdf = torch.zeros(n_initial, 1)

        batch['initial_points'] = initial_points
        batch['initial_sdf'] = initial_sdf

        # Data sampling
        n_temporal = batch_size*1 // 10000    # Unused
        if self.temporal_data is not None:
            temporal_points, temporal_t, temporal_sdf = self.sample_from_temporal_data(n_temporal)
            batch['space_time_points'] = temporal_points
            batch['space_time_t'] = temporal_t
            batch['space_time_sdf'] = temporal_sdf

        # Collaboration points sampling for PDE loss
        n_pde = batch_size * 9 // 16
        pde_points, pde_t = self.sample_pde_points(n_pde, time_interval)
        batch['pde_points'] = pde_points
        batch['pde_t'] = pde_t


        return batch

    def train_epoch(self, initial_surface, batch_size=10000):
        """Training of one epoch"""
        self.model.train()

        batch = self.sample_training_batch(initial_surface, batch_size)

        self.optimizer.zero_grad()
        loss = self.loss_fn(self.model, batch, self.etching_rate_model)

        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        return loss.item()

    def train(self, initial_surface, num_epochs=10000, batch_size=10000, save_interval=1000):
        """Whole training process"""
        os.makedirs('checkpoints', exist_ok=True)
        losses = []
        start_time = time.time()

        print(f"开始训练，总轮数: {num_epochs}")
        if self.temporal_data:
            print("训练模式: 初始条件 + 时序数据 + PDE约束")
        else:
            print("训练模式: 初始条件 + PDE约束")

        for epoch in range(num_epochs):
            loss = self.train_epoch(initial_surface, batch_size)
            losses.append(loss)
            self.scheduler.step(loss)

            if epoch % 500 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                elapsed_time = time.time() - start_time
                # 手动打印学习率变化信息
                lr_info = f", LR变化: {current_lr:.2e}" if epoch > 0 and current_lr != self.optimizer.param_groups[
                    0].get('initial_lr', 1e-4) else ""
                print(f"Epoch {epoch:05d}/{num_epochs}, Loss: {loss:.6f}{lr_info}, Time: {elapsed_time:.2f}s")

            if (epoch + 1) % save_interval == 0:
                checkpoint_path = f'checkpoints/model_epoch_{epoch + 1}.pth'
                self.model.save(checkpoint_path)

        # 保存最终模型和训练记录
        os.makedirs('models', exist_ok=True)
        final_model_path = 'models/final_model.pth'
        self.model.save(final_model_path)

        # 保存训练损失
        loss_data = {'losses': losses, 'epochs': num_epochs}
        with open('models/training_losses.pkl', 'wb') as f:
            pickle.dump(loss_data, f)

        total_time = time.time() - start_time
        print(f"训练完成! 总时间: {total_time:.2f}秒")

        return losses