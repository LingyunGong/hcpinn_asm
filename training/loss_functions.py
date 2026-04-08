import torch
import torch.nn as nn


class LevelSetLoss(nn.Module):
    """水平集演化损失函数"""

    def __init__(self, lambda_data=1.0, lambda_pde=1.0, lambda_temporal_data=0.0,lambda_eikonal=0.1):
        super().__init__()
        self.lambda_data = lambda_data
        self.lambda_temporal_data=lambda_temporal_data
        self.lambda_pde = lambda_pde
        self.lambda_eikonal = lambda_eikonal

    def compute_derivatives(self, model, points, t_values, create_graph=True):
        """计算时空导数"""
        points = points.clone().detach().requires_grad_(True)
        t_values = t_values.clone().detach().requires_grad_(True)

        inputs = torch.cat([points, t_values], dim=1)
        sdf_values = model(inputs)

        # 自动微分
        gradients = torch.autograd.grad(
            sdf_values, [points, t_values],
            grad_outputs=torch.ones_like(sdf_values),
            create_graph=create_graph,
            retain_graph=True,
            allow_unused=True
        )

        spatial_gradients = gradients[0] if gradients[0] is not None else torch.zeros_like(points)
        time_derivatives = gradients[1] if gradients[1] is not None else torch.zeros_like(t_values)

        return sdf_values, spatial_gradients, time_derivatives

    def compute_dr1_flow_term(self, model, points, t_values, mu=1.0, epsilon=1e-6):
        """
        简化的DR1正则化梯度流项计算（针对2D/3D优化）
        """
        points = points.clone().detach().requires_grad_(True)
        t_values = t_values.clone().detach().requires_grad_(True)

        inputs = torch.cat([points, t_values], dim=1)
        phi = model(inputs)

        # 计算梯度
        gradients = torch.autograd.grad(
            outputs=phi,
            inputs=points,
            grad_outputs=torch.ones_like(phi),
            create_graph=True,
            retain_graph=True
        )[0]
        # 计算梯度模长
        grad_norm = torch.norm(gradients, dim=1, keepdim=True) + epsilon
        # 系数 (1 - 1/|∇φ|)
        coeff = 1.0 - 1.0 / grad_norm

        # 散度计算：∂/∂x(coeff * ∂φ/∂x) + ∂/∂y(coeff * ∂φ/∂y) [+ ∂/∂z(coeff * ∂φ/∂z) for 3D]
        d = points.shape[1]

        # 使用向量化方式计算散度
        if d == 2:  # 2D情况
            # coeff * gradients
            scaled_grad_x = coeff * gradients[:, 0]
            scaled_grad_y = coeff * gradients[:, 1]

            # 计算每个分量的梯度
            grad_scaled_x = torch.autograd.grad(
                outputs=scaled_grad_x,
                inputs=points,
                grad_outputs=torch.ones_like(scaled_grad_x),
                create_graph=False,
                retain_graph=True
            )[0]

            grad_scaled_y = torch.autograd.grad(
                outputs=scaled_grad_y,
                inputs=points,
                grad_outputs=torch.ones_like(scaled_grad_y),
                create_graph=False,
                retain_graph=True
            )[0]

            # 散度 = ∂(coeff*∂φ/∂x)/∂x + ∂(coeff*∂φ/∂y)/∂y
            dr1_term = grad_scaled_x[:, 0] + grad_scaled_y[:, 1]

        elif d == 3:  # 3D情况
            scaled_grad_x = coeff * gradients[:, 0]
            scaled_grad_y = coeff * gradients[:, 1]
            scaled_grad_z = coeff * gradients[:, 2]

            grad_scaled_x = torch.autograd.grad(
                outputs=scaled_grad_x,
                inputs=points,
                grad_outputs=torch.ones_like(scaled_grad_x),
                create_graph=False,
                retain_graph=True
            )[0]

            grad_scaled_y = torch.autograd.grad(
                outputs=scaled_grad_y,
                inputs=points,
                grad_outputs=torch.ones_like(scaled_grad_y),
                create_graph=False,
                retain_graph=True
            )[0]

            grad_scaled_z = torch.autograd.grad(
                outputs=scaled_grad_z,
                inputs=points,
                grad_outputs=torch.ones_like(scaled_grad_z),
                create_graph=False,
                retain_graph=True
            )[0]

            dr1_term = grad_scaled_x[:, 0] + grad_scaled_y[:, 1] + grad_scaled_z[:, 2]

        else:
            # 高维情况，使用通用方法
            scaled_gradients = coeff * gradients

            # 为每个维度创建分量
            scaled_components = [scaled_gradients[:, i] for i in range(d)]

            # 一次性计算所有分量的梯度
            grad_components = torch.autograd.grad(
                outputs=scaled_components,
                inputs=points,
                grad_outputs=[torch.ones_like(comp) for comp in scaled_components],
                create_graph=False,
                retain_graph=True
            )[0]

            # 梯度张量形状是(batch_size, d)，每列对应一个分量的梯度
            # 我们需要每个分量梯度的对角线元素
            dr1_term = torch.sum(grad_components[:, :d], dim=1)

        # 重新形状并乘以权重
        dr1_term = mu * dr1_term.unsqueeze(1)

        return dr1_term

    def compute_mbe_term(self, model, points, t_values, alpha=10.0, mu=1.0):
        """
        实用的MBE正则化项计算
        在速度和准确性之间取得平衡
        """
        batch_size, d = points.shape

        # 1. 使用向量化计算一阶和二阶导数
        points = points.clone().detach().requires_grad_(True)
        t_values = t_values.clone().detach().requires_grad_(True)

        inputs = torch.cat([points, t_values], dim=1)
        phi = model(inputs)

        # 计算一阶导数
        gradients = torch.autograd.grad(
            phi, points,
            grad_outputs=torch.ones_like(phi),
            create_graph=True,
            retain_graph=True
        )[0]

        # 预分配Hessian存储
        hessian_list = []

        # 计算二阶导数（Hessian）
        for i in range(d):
            # 计算第i个梯度分量的梯度
            grad_i = gradients[:, i].reshape(-1, 1)
            hessian_i = torch.autograd.grad(
                grad_i, points,
                grad_outputs=torch.ones_like(grad_i),
                create_graph=True,
                retain_graph=True
            )[0]
            hessian_list.append(hessian_i)

        # 堆叠得到Hessian矩阵
        hessian = torch.stack(hessian_list, dim=2)  # (batch_size, d, d)

        # 2. 计算拉普拉斯（Hessian的迹）
        laplacian = torch.einsum('bii->b', hessian).reshape(-1, 1)

        # 3. 计算双拉普拉斯（使用有限差分近似以提高速度）
        eps = 1e-3
        bilaplacian = torch.zeros_like(laplacian)

        for i in range(d):
            # 创建扰动点
            points_plus = points.clone()
            points_plus[:, i] += eps

            points_minus = points.clone()
            points_minus[:, i] -= eps

            # 计算扰动点的拉普拉斯
            inputs_plus = torch.cat([points_plus, t_values], dim=1)
            laplacian_plus = torch.autograd.grad(
                torch.autograd.grad(
                    model(inputs_plus), points_plus,
                    grad_outputs=torch.ones_like(phi),
                    create_graph=True,
                    retain_graph=True
                )[0][:, i].reshape(-1, 1),
                points_plus,
                grad_outputs=torch.ones_like(phi),
                create_graph=False,
                retain_graph=True
            )[0][:, i].reshape(-1, 1)

            inputs_minus = torch.cat([points_minus, t_values], dim=1)
            laplacian_minus = torch.autograd.grad(
                torch.autograd.grad(
                    model(inputs_minus), points_minus,
                    grad_outputs=torch.ones_like(phi),
                    create_graph=True,
                    retain_graph=True
                )[0][:, i].reshape(-1, 1),
                points_minus,
                grad_outputs=torch.ones_like(phi),
                create_graph=False,
                retain_graph=True
            )[0][:, i].reshape(-1, 1)

            # 中心差分近似二阶导数
            bilaplacian += (laplacian_plus - 2 * laplacian + laplacian_minus) / (eps ** 2)

        # 4. 计算非线性项
        grad_norm_sq = torch.sum(gradients ** 2, dim=1, keepdim=True)

        # 使用向量化计算非线性项
        nonlinear_term = torch.zeros_like(phi)

        # 预计算 (|∇φ|² - 1) * gradients
        scaled_gradients = (grad_norm_sq - 1.0) * gradients

        for i in range(d):
            # 计算每个分量的散度
            comp_i = scaled_gradients[:, i].reshape(-1, 1)
            div_comp_i = torch.autograd.grad(
                comp_i, points,
                grad_outputs=torch.ones_like(comp_i),
                create_graph=False,
                retain_graph=True
            )[0][:, i].reshape(-1, 1)

            nonlinear_term += div_comp_i

        # 5. 组合结果
        regularization_term = mu * (-alpha * bilaplacian + nonlinear_term)

        return regularization_term
    def compute_curvature(self, spatial_gradients, points):
        """计算平均曲率"""
        if not points.requires_grad:
            points = points.clone().detach().requires_grad_(True)

        norm_grad = torch.norm(spatial_gradients, dim=1, keepdim=True) + 1e-6
        unit_normal = spatial_gradients / norm_grad

        curvature = torch.zeros(points.shape[0], 1).to(points.device)
        for i in range(3):
            div_component = torch.autograd.grad(
                unit_normal[:, i], points,
                grad_outputs=torch.ones_like(unit_normal[:, i]),
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )[0]
            if div_component is not None:
                curvature += div_component[:, i:i + 1]

        return curvature

    def forward(self, model, samples, etching_rate_model):
        """总损失计算"""
        total_loss = 0

        # 数据损失 - 初始条件  t = 0
        if 'initial_points' in samples:
            init_points = samples['initial_points']
            init_t = torch.zeros(init_points.shape[0], 1).to(init_points.device)

            init_points = init_points.clone().detach().requires_grad_(True)
            init_t = init_t.clone().detach().requires_grad_(True)

            init_inputs = torch.cat([init_points, init_t], dim=1)
            pred_sdf = model(init_inputs)
            target_sdf = samples['initial_sdf']

            data_loss = torch.mean(torch.abs(pred_sdf - target_sdf))
            total_loss += self.lambda_data * data_loss
            # # compute boundary loss
            # r = torch.sqrt(torch.rand(2000, 1)) * 0.5 + 1.1
            # theta = torch.rand(2000, 1) * 2 * torch.pi
            # x_coords = r * torch.cos(theta)
            # z_coords = r * torch.sin(theta)
            # y_coords = torch.ones(2000, 1) * -0.7
            # t_evo = torch.rand(2000, 1) * 1.5
            # boundary_inputs = torch.cat([x_coords, y_coords, z_coords,t_evo], dim=1)
            # boundary_inputs = boundary_inputs.clone().detach().requires_grad_(True)
            # pred_bs = model(boundary_inputs)
            # boundary_loss = torch.mean(torch.abs(pred_bs))
            # total_loss += self.lambda_data * boundary_loss

            # Eikonal正则化
            if init_points.requires_grad:
                init_gradients = torch.autograd.grad(
                    pred_sdf, init_points,
                    grad_outputs=torch.ones_like(pred_sdf),
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True
                )[0]

                if init_gradients is not None:
                    eikonal_loss = torch.mean(torch.abs(torch.norm(init_gradients, dim=1) - 1))
                    total_loss += self.lambda_eikonal * eikonal_loss

        # 时序数据损失 - 各个时间点的监督数据
        if 'space_time_sdf' in samples:
            temp_points = samples['space_time_points']
            temp_t = samples['space_time_t']
            temp_sdf = samples['space_time_sdf']

            temp_points = temp_points.clone().detach().requires_grad_(True)
            temp_t = temp_t.clone().detach().requires_grad_(True)

            temp_inputs = torch.cat([temp_points, temp_t], dim=1)
            pred_temp_sdf = model(temp_inputs)
            target_temp_sdf = temp_sdf

            temporal_data_loss = torch.mean(torch.abs(pred_temp_sdf - target_temp_sdf))
            total_loss += self.lambda_temporal_data * temporal_data_loss


        # PDE损失 - 水平集方程
        if 'pde_points' in samples and 'pde_t' in samples:
            st_points = samples['pde_points']
            st_t = samples['pde_t']

            st_points = st_points.clone().detach().requires_grad_(True)
            st_t = st_t.clone().detach().requires_grad_(True)

            sdf_vals, spatial_grads, time_derivs = self.compute_derivatives(
                model, st_points, st_t
            )
            norm_grad = torch.norm(spatial_grads, dim=1, keepdim=True) + 1e-6
            normals = spatial_grads / norm_grad
            #curvatures = self.compute_curvature(spatial_grads, st_points)

            etching_rates = etching_rate_model(st_points, normals, st_t)
            pde_residual = time_derivs + etching_rates * norm_grad #方向需调整

            # 计算正则化项
            regularization_term =  self.lambda_eikonal * torch.abs( norm_grad-1)
            # 将正则化项加入PDE残差
            pde_residual = pde_residual + regularization_term
            # 权重 越靠近零等值面越大
            lambda_sdf = torch.exp(- torch.abs(sdf_vals))
            pde_residual =pde_residual * lambda_sdf
            pde_loss = torch.mean(torch.abs(pde_residual))
            total_loss += self.lambda_pde * pde_loss

        return total_loss