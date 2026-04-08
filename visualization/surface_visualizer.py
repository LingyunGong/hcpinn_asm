import numpy as np
import torch
import trimesh
from skimage import measure
import plotly.graph_objects as go
import os
import plotly.subplots as sp



class SurfaceExtractor:
    """表面提取和可视化工具"""

    def __init__(self, model, resolution=64):
        self.model = model
        self.resolution = resolution
        self.rbox = (-0.5, 0.5)
        self.ybox = (-1.0, 4.0)
    def extract_isosurface(self, t=0.0, sym='1'):
        """提取指定时间的零等值面"""
        x = np.linspace(self.rbox[0], self.rbox[1], self.resolution)
        y = np.linspace(self.ybox[0], self.ybox[1], self.resolution)
        z = np.linspace(self.rbox[0], self.rbox[1], self.resolution)

        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        grid_points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

        t_array = np.ones((grid_points.shape[0], 1)) * t
        inputs = np.concatenate([grid_points, t_array], axis=1)



        with torch.no_grad():
            inputs_tensor = torch.tensor(inputs).float()
            sdf_values = self.model(inputs_tensor).numpy()

        sdf_grid = sdf_values.reshape(self.resolution, self.resolution, self.resolution)
        # 对称化处理
        if sym == '0':
            # 方法1：分别翻转X轴和Z轴
            sdf_flipped_x = np.flip(sdf_grid, axis=0)  # 翻转X轴
            sdf_flipped_xz = np.flip(sdf_flipped_x, axis=2)  # 翻转Z轴

            # 取原网格和翻转网格的平均值
            sdf_grid = 0.5 * (sdf_grid + sdf_flipped_xz)

            # 方法2：同时翻转多个轴（更简洁）
            # sdf_flipped = np.flip(sdf_grid, axis=(0, 2))  # 同时翻转X轴和Z轴
            # sdf_grid = 0.5 * (sdf_grid + sdf_flipped)
        try:
            vertices, faces, _, _ = measure.marching_cubes(sdf_grid, level=0)

            # 分别计算每个方向的缩放因子和偏移
            scale_x = (self.rbox[1] - self.rbox[0]) / (self.resolution - 1)
            scale_y = (self.ybox[1] - self.ybox[0]) / (self.resolution - 1)
            scale_z = (self.rbox[1] - self.rbox[0]) / (self.resolution - 1)

            # 分别对每个坐标轴进行缩放和偏移
            vertices[:, 0] = vertices[:, 0] * scale_x + self.rbox[0]  # X轴
            vertices[:, 1] = vertices[:, 1] * scale_y + self.ybox[0]  # Y轴（关键修正！）
            vertices[:, 2] = vertices[:, 2] * scale_z + self.rbox[0]  # Z轴

            return trimesh.Trimesh(vertices=vertices, faces=faces)
        except Exception as e:
            print(f"等值面提取失败 (t={t}): {e}")
            return None

    def compute_sdf_grid(self, t=0.0):
        """计算整个区域在给定时间的SDF值网格"""
        x = np.linspace(self.rbox[0], self.rbox[1], self.resolution)
        y = np.linspace(self.ybox[0], self.ybox[1], self.resolution)
        z = np.linspace(self.rbox[0], self.rbox[1], self.resolution)

        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        grid_points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

        t_array = np.ones((grid_points.shape[0], 1)) * t
        inputs = np.concatenate([grid_points, t_array], axis=1)

        with torch.no_grad():
            inputs_tensor = torch.tensor(inputs).float()
            sdf_values = self.model(inputs_tensor).numpy()

        sdf_grid = sdf_values.reshape(self.resolution, self.resolution, self.resolution)

        return {
            'sdf_grid': sdf_grid,
            'x_coords': x,
            'y_coords': y,
            'z_coords': z,
            'X': X,
            'Y': Y,
            'Z': Z
        }

    def create_interactive_cross_section(self, t=0.0, planes=['x=0', 'y=0', 'z=0'],
                                             save_path=None):
        """
        创建交互式截面图（使用Plotly）  待完善
        """

        # 计算SDF网格
        grid_data = self.compute_sdf_grid(t)
        sdf_grid = grid_data['sdf_grid']
        x_coords = grid_data['x_coords']
        y_coords = grid_data['y_coords']
        z_coords = grid_data['z_coords']

        # 创建子图
        n_planes = len(planes)
        fig = sp.make_subplots(
            rows=1, cols=n_planes,
            subplot_titles=[f'截面 {plane}' for plane in planes],
            horizontal_spacing=0.1
        )

        for idx, plane in enumerate(planes):
            # 解析平面参数
            plane_axis = plane[0]
            plane_value = float(plane.split('=')[1])

            # 找到最接近的网格索引
            if plane_axis == 'x':
                coord_array = x_coords
                slice_axis = 0
                x_plot, y_plot = y_coords, z_coords
                xlabel, ylabel = 'Y', 'Z'
            elif plane_axis == 'y':
                coord_array = y_coords
                slice_axis = 1
                x_plot, y_plot = x_coords, z_coords
                xlabel, ylabel = 'X', 'Z'
            else:  # plane_axis == 'z'
                coord_array = z_coords
                slice_axis = 2
                x_plot, y_plot = x_coords, y_coords
                xlabel, ylabel = 'X', 'Y'

            # 找到最接近的索引
            plane_idx = np.argmin(np.abs(coord_array - plane_value))

            # 提取截面
            if slice_axis == 0:
                sdf_slice = sdf_grid[plane_idx, :, :]
            elif slice_axis == 1:
                sdf_slice = sdf_grid[:, plane_idx, :]
            else:  # slice_axis == 2
                sdf_slice = sdf_grid[:, :, plane_idx]

            # 创建网格
            X_plot, Y_plot = np.meshgrid(x_plot, y_plot)

            # 添加热力图
            heatmap = go.Heatmap(
                x=x_plot,
                y=y_plot,
                z=sdf_slice.T,
                colorscale='RdBu_r',
                zmid=0,
                colorbar=dict(title='SDF值'),
                hovertemplate=f'{xlabel}: %{{x:.2f}}<br>{ylabel}: %{{y:.2f}}<br>SDF: %{{z:.3f}}<extra></extra>'
            )

            # 添加零等高线
            # 需要提取等高线数据
            from scipy import ndimage

            # 创建等高线
            zero_mask = sdf_slice.T > 0
            zero_contours = measure.find_contours(zero_mask, 1)

            for contour in zero_contours:
                # 将轮廓索引转换为实际坐标
                y_contour = x_plot[0] + (x_plot[-1] - x_plot[0]) * contour[:, 1] / (len(x_plot) - 1)
                x_contour = y_plot[0] + (y_plot[-1] - y_plot[0]) * contour[:, 0] / (len(y_plot) - 1)

                contour_line = go.Scatter(
                    x=y_contour,
                    y=x_contour,
                    mode='lines',
                    line=dict(color='black', width=2),
                    showlegend=False,
                    hoverinfo='skip'
                )
                fig.add_trace(contour_line, row=1, col=idx + 1)

            fig.add_trace(heatmap, row=1, col=idx + 1)

            # 更新子图布局
            fig.update_xaxes(title_text=xlabel, row=1, col=idx + 1)
            fig.update_yaxes(title_text=ylabel, row=1, col=idx + 1)

        # 更新整体布局
        fig.update_layout(
            title=f'刻蚀截面图 (时间 t={t})',
            height=500,
            width=300 * n_planes,
            showlegend=False
        )

        if save_path:
            fig.write_html(save_path)
            print(f"交互式截面图已保存到: {save_path}")

        return fig

    def plot_3d_surface(self, t=0.0, title=None, save_path=None):
        """使用Plotly绘制3D表面"""
        mesh = self.extract_isosurface(t)
        if mesh is None:
            print(f"时间 t={t} 时无法提取表面")
            return None

        vertices = mesh.vertices
        faces = mesh.faces

        fig = go.Figure(data=[
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                opacity=0.8,
                color='lightblue',
                flatshading=True,
                lighting=dict(
                    ambient=0.3,
                    diffuse=0.8,
                    fresnel=0.1,
                    specular=0.1,
                    roughness=0.5
                ),
                lightposition=dict(x=100, y=100, z=100)
            )
        ])

        if title is None:
            title = f'刻蚀表面 (t={t})'

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            width=800,
            height=600
        )

        if save_path:
            fig.write_html(save_path)
            print(f"3D图已保存到: {save_path}")

        return fig

    def plot_evolution_sequence(self, time_steps, save_dir='visualizations'):
        """绘制演化序列"""
        os.makedirs(save_dir, exist_ok=True)
        figures = []

        for i, t in enumerate(time_steps):
            fig = self.plot_3d_surface(
                t=t,
                title=f'刻蚀演化 (t={t})',
                save_path=os.path.join(save_dir, f'surface_t_{t:.1f}.html')
            )
            if fig is not None:
                figures.append(fig)

        print(f"生成 {len(figures)} 个演化图")
        return figures


def plot_training_loss(losses, save_path='training_loss', output_format='html', dpi=300, aspect_equal=False):
    """
    绘制训练损失曲线

    参数:
        losses (list): 每个 epoch 的损失值列表
        save_path (str): 保存路径（不含扩展名）
        output_format (str): 'html' 或 'png'
        dpi (int): PNG 分辨率
        aspect_equal (bool): 是否设置等比例坐标
    """
    epochs = list(range(1, len(losses) + 1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs,
        y=losses,
        mode='lines',
        name='Training Loss',
        line=dict(color='blue', width=2)
    ))

    # 基础布局设置（无网格、有边框等）
    font_size = 20
    layout_updates = dict(
        title='Training Loss Curve',
        xaxis_title='Epoch',
        yaxis_title='Loss',
        yaxis_type='log',
        width=800,
        height=500,
        font=dict(
            family="Times New Roman, serif",  # 首选 Times New Roman，若无则用 serif 家族
            size=12,  # 全局基础字号，可根据需要调整
            color="black"
        ),
        title_font=dict(size=font_size+4),
        plot_bgcolor='white',
        xaxis=dict(
            title_font=dict(size=font_size),
            tickfont=dict(size=font_size - 4),
            showgrid=False,
            showline=True,
            mirror=True,
            linewidth=1,
            linecolor='black',
            ticks="outside",  # 明确显示刻度线
            ticklen=6,  # 稍微加长
            tickcolor='black',  # 确保颜色可见
            tickwidth=1
        ),
        yaxis=dict(
            title_font=dict(size=font_size),
            tickfont=dict(size=font_size - 4),
            showgrid=False,
            showline=True,
            mirror=True,
            linewidth=1,
            linecolor='black',
            tickformat=".2f",
            minexponent=3,
            ticks="outside",
            ticklen=6,
            tickcolor='black',
            tickwidth=1
        )
    )

    # 如果需要等比例，添加 scaleanchor 和 scaleratio
    if aspect_equal:
        # 在原有 yaxis 设置基础上添加等比例参数
        layout_updates['yaxis'].update(
            scaleanchor="x",
            scaleratio=1
        )
        # 注意：等比例时可能需要调整图形尺寸，以免图形被压扁或拉长
        # 可以根据数据范围自动计算合适的 width/height，或提示用户手动调整

    fig.update_layout(**layout_updates)

    # 保存代码（同前）
    if output_format == 'html':
        full_path = save_path + '.html'
        fig.write_html(full_path)
        print(f"损失曲线已保存到: {full_path}")
    elif output_format == 'png':
        full_path = save_path + '.png'
        fig.write_image(full_path, scale=dpi/100)
        print(f"损失曲线已保存到: {full_path} (DPI={dpi})")
    else:
        raise ValueError("output_format 必须是 'html' 或 'png'")

    return fig