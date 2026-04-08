# Load model and visualize
import torch
import os
import pickle
import time
import numpy as np
from models.neural_siren import SpaceTimeSIREN
from visualization.surface_visualizer import SurfaceExtractor, plot_training_loss
import plotly.graph_objects as go
import plotly.express as px


def test_loaded_model():
    """测试加载的模型"""

    print("开始测试加载的模型...")

    # 1. 加载模型
    try:
        model = SpaceTimeSIREN.load('models/final_model.pth')
        print("✓ 神经网络模型加载成功")
    except Exception as e:
        print(f"✗ 神经网络模型加载失败: {e}")
        return


    # 2. 加载训练损失
    try:
        with open('models/training_losses.pkl', 'rb') as f:
            loss_data = pickle.load(f)
        losses = loss_data['losses']
        print("✓ 训练损失数据加载成功")
    except Exception as e:
        print(f"✗ 训练损失数据加载失败: {e}")
        losses = []

    # # 3. 测试模型推理
    # print("\n测试模型推理...")
    # test_points = torch.randn(10, 4)
    #
    # # 预热
    # with torch.no_grad():
    #     _ = model(test_points)
    #
    # # 测量推理时间
    # n_tests = 100
    # start = time.perf_counter()
    # with torch.no_grad():
    #     for _ in range(n_tests):
    #         predictions = model(test_points)
    # end = time.perf_counter()
    #
    # avg_time = (end - start) / n_tests
    # print(f"✓ 模型推理成功")
    # print(f"  输出形状: {predictions.shape}")
    # print(f"  平均推理时间: {avg_time * 1000:.2f}ms")
    # print(f"  每秒推理次数: {1 / avg_time:.1f}")

    # 5. 创建表面提取器

    extractor = SurfaceExtractor(model, resolution=64)

    # # 6. 测试表面提取
    # print("\n测试表面提取...")
    #
    # test_times = [0.0, 0.5, 1.0]
    # results = []
    # # 预热一次（避免第一次提取的额外开销）
    # if len(test_times) > 0:
    #     _ = extractor.extract_isosurface(test_times[0])
    # for t in test_times:
    #     timer = time.perf_counter()
    #     mesh = extractor.extract_isosurface(t)
    #     elapsed = time.perf_counter() - timer
    #
    #     status = "成功" if mesh is not None else "失败"
    #     vertices = len(mesh.vertices) if mesh is not None else 0
    #
    #     results.append((t, elapsed, status, vertices))
    #
    #     print(f"t={t:.1f}: {status}, {elapsed * 1000:.1f}ms, {vertices}个顶点")
    #
    # # 计算平均耗时（只计算成功的提取）
    # successful_times = [elapsed for _, elapsed, status, _ in results if status == "成功"]
    # if successful_times:
    #     avg_time = sum(successful_times) / len(successful_times)
    #     print(f"\n平均每个时间点提取耗时: {avg_time * 1000:.1f}ms")
    #     print(f"成功率: {len(successful_times)}/{len(results)}")

    # 7. 生成可视化
    print("\n生成可视化...")
    # 绘制训练损失
    if losses:
        plot_training_loss(losses, 'visualizations/test_results/test_training_loss', output_format='png', dpi=360)

    # 创建新的演化序列可视化
    time_steps = [0.0, 0.5, 1.0, 1.5]
    figures = extractor.plot_evolution_sequence(
        time_steps,
        save_dir='visualizations/test_results'
    )

    print(f"✓ 生成 {len(figures)} 个3D可视化图")

    # 8. 比较不同时间的表面
    print("\n比较不同时间的表面体积:")
    for t in time_steps:
        mesh = extractor.extract_isosurface(t)
        if mesh and mesh.vertices is not None and len(mesh.vertices) > 0:
            volume = mesh.volume if hasattr(mesh, 'volume') else 0
            print(f"  时间 t={t}: 顶点数={len(mesh.vertices)}, 面数={len(mesh.faces)}, 体积≈{volume:.4f}")

    # 9. 创建对比图
    #create_comparison_plot(extractor, time_steps)

    # 时间演化动态图
    create_animated_evolution(extractor, time_points=30,time_end=2.0, save_path='visualizations/test_results/animated_evolution.html')
    # 切片
    # create_sliced_visualization(
    #     extractor,
    #     t=1.5,
    #     slice_plane='z',
    #     slice_value=0,
    #     save_path=f'visualizations/test_results/slice_{t}.html'
    # )
    fig_slice = plot_evolution_slice_curves(
        extractor,
        t_start=0.0,
        t_end=2.0,
        num_curves=9,
        slice_plane='z',
        slice_value=0.0,
        save_path='visualizations/test_results/evolution_5curves.html'
    )
    #最大刻蚀深度
    calculate_highest_point(extractor, t=0.5)
    calculate_highest_point(extractor, t=1.0)
    #calculate_highest_point(extractor, t=2.0)
    print("\n✓ 模型测试完成！")


def create_sliced_visualization(extractor, t=1.0, slice_plane='z', slice_value=0.0,
                                bbox=(-1.5, 1.5), plot_2d=True,
                                save_path='visualizations/test_results/sliced_view.html'):
    """
    创建切片可视化 - 可选择绘制二维截面图或三维切片图

    参数:
        extractor: SurfaceExtractor实例
        t: 时间点
        slice_plane: 切片平面 ('x', 'y', 'z')
        slice_value: 切片值
        bbox: 边界框范围
        plot_2d: True表示绘制二维截面图，False表示绘制三维切片图
        save_path: 保存路径
    """

    if plot_2d:
        # ================== 绘制二维截面图（零等值线） ==================
        print(f"绘制{t}时刻的{slice_plane}={slice_value}二维截面图...")

        # 计算SDF网格
        grid_data = extractor.compute_sdf_grid(t)
        sdf_grid = grid_data['sdf_grid']
        x_coords = grid_data['x_coords']
        y_coords = grid_data['y_coords']
        z_coords = grid_data['z_coords']

        # 根据切片平面提取截面
        if slice_plane == 'x':
            # 找到最接近slice_value的x索引
            slice_idx = np.argmin(np.abs(x_coords - slice_value))
            sdf_slice = sdf_grid[slice_idx, :, :]
            x_plot, y_plot = y_coords, z_coords
            xlabel, ylabel = 'Y', 'Z'
            title = f'X = {slice_value:.2f} 截面 (t={t})'

        elif slice_plane == 'y':
            # 找到最接近slice_value的y索引
            slice_idx = np.argmin(np.abs(y_coords - slice_value))
            sdf_slice = sdf_grid[:, slice_idx, :]
            x_plot, y_plot = x_coords, z_coords
            xlabel, ylabel = 'X', 'Z'
            title = f'Y = {slice_value:.2f} 截面 (t={t})'

        else:  # slice_plane == 'z'
            # 找到最接近slice_value的z索引
            slice_idx = np.argmin(np.abs(z_coords - slice_value))
            sdf_slice = sdf_grid[:, :, slice_idx]
            x_plot, y_plot = x_coords, y_coords
            xlabel, ylabel = 'X', 'Y'
            title = f'Z = {slice_value:.2f} 截面 (t={t})'

        # 创建网格用于等高线
        X_plot, Y_plot = np.meshgrid(x_plot, y_plot)

        # 创建图形
        fig = go.Figure()

        # 添加热力图背景（可选，可以注释掉）
        fig.add_trace(go.Heatmap(
            x=x_plot,
            y=y_plot,
            z=sdf_slice.T,
            colorscale='RdBu_r',
            zmid=0,
            opacity=0.3,
            showscale=True,
            colorbar=dict(title='SDF值'),
            name='SDF值分布',
            hoverinfo='x+y+z'
        ))

        # 提取零等值线
        from skimage import measure
        try:
            # 寻找零等高线（SDF=0）
            zero_contours = measure.find_contours(sdf_slice.T, 0)

            for i, contour in enumerate(zero_contours):
                # 将轮廓索引转换为实际坐标
                x_contour = x_plot[0] + (x_plot[-1] - x_plot[0]) * contour[:, 1] / (len(x_plot) - 1)
                y_contour = y_plot[0] + (y_plot[-1] - y_plot[0]) * contour[:, 0] / (len(y_plot) - 1)

                fig.add_trace(go.Scatter(
                    x=x_contour,
                    y=y_contour,
                    mode='lines',
                    line=dict(color='red', width=3),
                    name=f'零等值线 {i + 1}' if i == 0 else '',
                    showlegend=True if i == 0 else False,
                    hovertemplate=f'{xlabel}: %{{x:.3f}}<br>{ylabel}: %{{y:.3f}}<extra></extra>'
                ))
        except Exception as e:
            print(f"提取等值线时出错: {e}")

        # 更新布局
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=16)
            ),
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            width=700,
            height=600,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            ),
            hovermode='closest'
        )
        fig.update_yaxes(autorange="reversed")   # 翻转y轴
        # 确保坐标轴等比例
        fig.update_xaxes(
            scaleanchor="y",
            scaleratio=1,
            constrain='domain'
        )

        # 添加网格线
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

    else:
        # ================== 绘制三维切片图（原有功能） ==================
        print(f"绘制{t}时刻的{slice_plane}={slice_value}三维切片图...")

        # 提取完整等值面
        mesh = extractor.extract_isosurface(t, bbox)

        if mesh is None:
            print(f"无法在t={t}时提取等值面")
            return None

        vertices = mesh.vertices
        faces = mesh.faces

        # 根据切片平面裁剪顶点
        if slice_plane == 'z':
            mask = vertices[:, 2] > slice_value
            slice_name = f'z > {slice_value}'
        elif slice_plane == 'y':
            mask = vertices[:, 1] > slice_value
            slice_name = f'y > {slice_value}'
        else:  # slice_plane == 'x'
            mask = vertices[:, 0] > slice_value
            slice_name = f'x > {slice_value}'

        # 获取保留的顶点索引
        keep_vertices = np.where(mask)[0]

        # 创建顶点索引映射
        vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(keep_vertices)}

        # 筛选保留的顶点
        new_vertices = vertices[keep_vertices]

        # 筛选保留的面（所有顶点都在保留集合中的面）
        new_faces = []
        for face in faces:
            if all(v in vertex_map for v in face):
                new_faces.append([vertex_map[v] for v in face])

        new_faces = np.array(new_faces) if new_faces else np.array([], dtype=int)

        print(f"切片结果: 原始顶点={len(vertices)}, 切片后顶点={len(new_vertices)}")
        print(f"         原始面={len(faces)}, 切片后面={len(new_faces)}")

        # 创建三维切片可视化
        fig = go.Figure()

        # 添加切片后的网格
        if len(new_vertices) > 0 and len(new_faces) > 0:
            fig.add_trace(go.Mesh3d(
                x=new_vertices[:, 0],
                y=new_vertices[:, 1],
                z=new_vertices[:, 2],
                i=new_faces[:, 0],
                j=new_faces[:, 1],
                k=new_faces[:, 2],
                opacity=0.8,
                color='lightblue',
                name=f'切片: {slice_name}'
            ))

        # 添加参考平面（切片边界）
        if slice_plane == 'z':
            # 创建一个半透明的平面表示切片位置
            x_range = [bbox[0], bbox[1]]
            y_range = [bbox[0], bbox[1]]
            X, Y = np.meshgrid(x_range, y_range)
            Z = np.full_like(X, slice_value)

            fig.add_trace(go.Surface(
                x=X, y=Y, z=Z,
                opacity=0.1,
                colorscale=[[0, 'red'], [1, 'red']],
                showscale=False,
                name='切片平面'
            ))

        fig.update_layout(
            title=f'刻蚀结构切片 (t={t}, {slice_name})',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data',
                camera=dict(eye=dict(x=2.0, y=1.5, z=1.8))
            ),
            width=1000,
            height=700
        )

    # 保存图像
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_html(save_path)
        print(f"✓ 可视化已保存到: {save_path}")

    return fig


def plot_evolution_slice_curves(extractor, t_start=0.0, t_end=1.0, num_curves=5,
                                              slice_plane='z', slice_value=0.0,
                                              bbox=(-1.5, 1.5),
                                              save_path='visualizations/evolution_slice_curves_with_polyline.html'):
    """
    绘制演化截面曲线 - 同时显示多个时间点的零等值线，并添加对称折线段

    参数:
        extractor: SurfaceExtractor实例
        t_start: 起始时间
        t_end: 结束时间
        num_curves: 绘制的曲线数量（时间点数量）
        slice_plane: 切片平面 ('x', 'y', 'z')
        slice_value: 切片值
        bbox: 边界框范围
        save_path: 保存路径
    """

    print(f"绘制{slice_plane}={slice_value}截面的演化曲线，时间范围: [{t_start}, {t_end}]...")

    # 生成等间隔的时间点
    time_points = np.linspace(t_start, t_end, num_curves)
    #time_points = [0,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0] # 或直接定义时间点
    print(f"时间点: {time_points}")

    # 创建图形
    fig = go.Figure()

    # 颜色映射，用于区分不同时间点的曲线
    #colors = px.colors.sequential.Viridis
    colors = px.colors.sequential.Blues  # 从浅蓝到深蓝
    color_scale = np.linspace(0, 1, num_curves)

    # 用于存储所有零等值线的点，用于自动调整坐标轴范围
    all_x_points = []
    all_y_points = []

    # 遍历每个时间点
    for i, t in enumerate(time_points):
        print(f"处理时间点 {i + 1}/{num_curves}: t = {t:.3f}")

        # 计算SDF网格
        grid_data = extractor.compute_sdf_grid(t)
        sdf_grid = grid_data['sdf_grid']
        x_coords = grid_data['x_coords']
        y_coords = grid_data['y_coords']
        z_coords = grid_data['z_coords']

        # 根据切片平面提取截面
        if slice_plane == 'x':
            # 找到最接近slice_value的x索引
            slice_idx = np.argmin(np.abs(x_coords - slice_value))
            sdf_slice = sdf_grid[slice_idx, :, :]
            x_plot, y_plot = y_coords, z_coords
            xlabel, ylabel = 'Y', 'Z'
            slice_title = f'X = {slice_value:.2f}'

        elif slice_plane == 'y':
            # 找到最接近slice_value的y索引
            slice_idx = np.argmin(np.abs(y_coords - slice_value))
            sdf_slice = sdf_grid[:, slice_idx, :]
            x_plot, y_plot = x_coords, z_coords
            xlabel, ylabel = 'X', 'Z'
            slice_title = f'Y = {slice_value:.2f}'

        else:  # slice_plane == 'z'
            # 找到最接近slice_value的z索引
            slice_idx = np.argmin(np.abs(z_coords - slice_value))
            sdf_slice = sdf_grid[:, :, slice_idx]
            x_plot, y_plot = x_coords, y_coords
            xlabel, ylabel = 'X', 'Y'
            slice_title = f'Z = {slice_value:.2f}'

        # 提取零等值线
        from skimage import measure
        try:
            # 寻找零等高线（SDF=0）
            zero_contours = measure.find_contours(sdf_slice.T, 0)

            # 获取当前时间点对应的颜色
            color_idx = int(color_scale[i] * (len(colors) - 1))
            line_color = colors[color_idx]

            # 绘制每条等值线
            for contour_idx, contour in enumerate(zero_contours):
                # 将轮廓索引转换为实际坐标
                x_contour = x_plot[0] + (x_plot[-1] - x_plot[0]) * contour[:, 1] / (len(x_plot) - 1)
                y_contour = y_plot[0] + (y_plot[-1] - y_plot[0]) * contour[:, 0] / (len(y_plot) - 1)

                # 收集点用于坐标轴范围调整
                all_x_points.extend(x_contour)
                all_y_points.extend(y_contour)

                # 添加曲线到图形
                show_legend = contour_idx == 0

                fig.add_trace(go.Scatter(
                    x=x_contour,
                    y=y_contour,
                    mode='lines',
                    line=dict(color=line_color, width=2.5),
                    name=f't = {t:.2f}',
                    showlegend=show_legend,
                    hovertemplate=f'{xlabel}: %{{x:.3f}}<br>{ylabel}: %{{y:.3f}}<br>t = {t:.2f}<extra></extra>',
                    opacity=0.9
                ))

        except Exception as e:
            print(f"时间点 t={t} 提取等值线时出错: {e}")

        # =============== 添加掩膜轮廓 ===============
        # 只在 slice_plane == 'x' 或 slice_plane == 'z' 时添加折线段
        if slice_plane in ['x', 'z']:
            # 获取当前时间点对应的颜色（与等值线相同）
            polyline_color = colors[int(color_scale[i] * (len(colors) - 1))]

            # 定义折线段的点
            # 固定点: (0.25, -0.7) 在截面坐标系中的表示
            # 可动点: (0.25+0.01*t, -1.0+0.1*t)

            # 注意：这里需要根据切片平面调整坐标
            if slice_plane == 'x':
                # 对于x切片，截面坐标是(Y, Z)
                # 固定点: y=-0.7, z=0.25（注意这里交换了）
                # 可动点: y=-1.0+0.1*t, z=0.25+0.01*t
                #h = 1.2
                P1 = (-0.7, 0.26)  # 固定点
                P2 = (-1.5 + 0.3 * t, 0.25 + 0.02 * t)  # 可动点
                P3 = (P2[0], P2[1] + 0.35)  # 延伸方向 (1,0) -> 在YZ平面上是(0,1)

                # 对称折线段（关于x=0对称）
                P1_sym = (-0.7, -0.26)  # x坐标取负
                P2_sym = (-1.5 + 0.3 * t, -(0.25 + 0.02 * t))
                P3_sym = (P2_sym[0], P2_sym[1] - 0.35)




            else:  # slice_plane == 'z'

                # 对于z切片，截面坐标是(X, Y)

                # 固定点: x=0.25, y=-0.7

                # 可动点: x=0.25+0.01*t, y=-1.0+0.1*t

                r = 0.25 + 0.02

                h = 1.2

                y_0 = -0.7

                P1 = (r, y_0)  # 固定点

                P2 = (r + 0.03 + 0.01 * t, y_0 - h + 0.27 * t)  # 可动点

                P3 = (r * 2, P2[1])  # 延伸方向 (1,0)

                # 对称折线段（关于z=0对称）

                P1_sym = (-r, y_0)  # x坐标取负

                P2_sym = (-(r + 0.03 + 0.01 * t), y_0 - h + 0.27 * t)

                P3_sym = (-r * 2, P2_sym[1])

                # 创建折线段的坐标列表

            x_polyline = [P1[0], P2[0], P3[0]]

            y_polyline = [P1[1], P2[1], P3[1]]

            x_polyline_sym = [P1_sym[0], P2_sym[0], P3_sym[0]]

            y_polyline_sym = [P1_sym[1], P2_sym[1], P3_sym[1]]

            # 将折线段的点也加入到范围计算中

            all_x_points.extend(x_polyline)

            all_x_points.extend(x_polyline_sym)

            all_y_points.extend(y_polyline)

            all_y_points.extend(y_polyline_sym)

            # 添加第一段折线段（只在第一个时间点显示图例）

            show_polyline_legend = i == 0

            fig.add_trace(go.Scatter(

                x=x_polyline,

                y=y_polyline,

                mode='lines+markers',

                line=dict(color=polyline_color, width=2, dash='dash'),

                marker=dict(size=6, symbol='circle'),

                name='right mask' if show_polyline_legend else '',

                showlegend=show_polyline_legend,

                hoverinfo='skip',

                opacity=0.7

            ))

            # 添加第二段对称折线段（不显示图例）

            fig.add_trace(go.Scatter(

                x=x_polyline_sym,

                y=y_polyline_sym,

                mode='lines+markers',

                line=dict(color=polyline_color, width=2, dash='dash'),

                marker=dict(size=6, symbol='circle'),

                name='left mask' if show_polyline_legend else '',

                showlegend=show_polyline_legend,

                hoverinfo='skip',

                opacity=0.7

            ))

    # 创建时间颜色条
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(
            colorscale='Blues',
            showscale=True,
            cmin=t_start,
            cmax=t_end,
            colorbar=dict(
                title="Time (t)",
                thickness=20,
                len=0.75,
                x=1.02,
                xpad=10
            )
        ),
        hoverinfo='none',
        showlegend=False
    ))

    # 设置坐标轴范围（稍微扩展一点边界）
    if all_x_points and all_y_points:
        x_min, x_max = min(all_x_points), max(all_x_points)
        y_min, y_max = min(all_y_points), max(all_y_points)

        # 扩展10%的边界
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_min -= x_range * 0.01
        x_max += x_range * 0.01
        y_min -= y_range * 0.15
        y_max += y_range * 0.4
    else:
        # 如果没有点，使用bbox
        x_min, x_max = bbox[0], bbox[1]
        y_min, y_max = bbox[0], bbox[1]

    # 更新布局
    fig.update_layout(
        title=dict(
            text=f'{slice_title} Profile Evolution Curves (with mask)',
            font=dict(size=16)
        ),
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        width=800,
        height=700,
        showlegend=True,
        # legend=dict(
        #     title="Time points",
        #     yanchor="top",
        #     y=0.99,
        #     xanchor="left",
        #     x=0.01,
        #     bgcolor='rgba(255, 255, 255, 0.8)',
        #     bordercolor='rgba(0, 0, 0, 0.2)',
        #     borderwidth=1
        # ),
        hovermode='closest',
        plot_bgcolor='white'
    )

    # 设置坐标轴范围和等比例
    fig.update_xaxes(
        range=[x_min, x_max],
        scaleanchor="y",
        scaleratio=1,
        constrain='domain',
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='gray'
    )

    fig.update_yaxes(
        range=[y_min, y_max],
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='gray',
        autorange="reversed"  # 根据原始代码翻转y轴
    )

    # 添加说明文本
    # annotation_text = f"Slices of : {slice_plane}={slice_value:.2f}<br>Amount of curves: {num_curves}"
    # if slice_plane in ['x', 'z']:
    #     annotation_text += "<br>Including hard mask"
    #
    # fig.add_annotation(
    #     text=annotation_text,
    #     xref="paper", yref="paper",
    #     x=0.02, y=0.02,
    #     showarrow=False,
    #     font=dict(size=12, color="black"),
    #     align="left",
    #     bgcolor="rgba(255, 255, 255, 0.7)",
    #     bordercolor="rgba(0, 0, 0, 0.2)",
    #     borderwidth=1,
    #     borderpad=4
    # )

    # 保存图像
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_html(save_path)
        print(f"✓ 演化曲线图（含折线段）已保存到: {save_path}")

    return fig

def create_comparison_plot(extractor, time_steps):
    """创建时间演化对比图"""

    fig = go.Figure()

    colors = ['blue', 'green', 'red', 'orange', 'purple']

    for i, t in enumerate(time_steps):
        mesh = extractor.extract_isosurface(t)
        if mesh is None:
            continue

        vertices = mesh.vertices

        fig.add_trace(go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=mesh.faces[:, 0],
            j=mesh.faces[:, 1],
            k=mesh.faces[:, 2],
            opacity=0.7,
            color=colors[i % len(colors)],
            name=f't = {t}',
            showlegend=True
        ))

    fig.update_layout(
        title='刻蚀演化时间序列对比',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        width=900,
        height=700
    )

    fig.write_html('visualizations/test_results/time_evolution_comparison.html')
    print("✓ 时间演化对比图已保存")



def create_animated_evolution(extractor, time_points=30, time_end=1.0, save_path='visualizations/test_results/animated_evolution.html'):
    """创建带滑块的动态演化图"""

    # 生成时间序列
    times = np.linspace(0, time_end, time_points)

    # 预先计算所有时间步的网格
    meshes = []
    valid_indices = []

    print("正在计算时间序列...")
    for i, t in enumerate(times):
        mesh = extractor.extract_isosurface(t)
        if mesh is not None:
            meshes.append(mesh)
            valid_indices.append(i)

    if not meshes:
        print("错误：未能提取到任何等值面")
        return None

    # 创建图形
    fig = go.Figure()

    # 添加第一个时间步的初始网格
    initial_mesh = meshes[0]
    fig.add_trace(go.Mesh3d(
        x=initial_mesh.vertices[:, 0],
        y=initial_mesh.vertices[:, 1],
        z=initial_mesh.vertices[:, 2],
        i=initial_mesh.faces[:, 0],
        j=initial_mesh.faces[:, 1],
        k=initial_mesh.faces[:, 2],
        opacity=0.8,
        color='lightblue',
        name=f't = {times[valid_indices[0]]:.2f}'
    ))

    # 为每个时间步创建帧
    frames = []
    for idx, mesh_idx in enumerate(valid_indices):
        mesh = meshes[idx]
        frames.append(go.Frame(
            data=[go.Mesh3d(
                x=mesh.vertices[:, 0],
                y=mesh.vertices[:, 1],
                z=mesh.vertices[:, 2],
                i=mesh.faces[:, 0],
                j=mesh.faces[:, 1],
                k=mesh.faces[:, 2],
                opacity=0.8,
                color='lightblue'
            )],
            name=f'frame_{mesh_idx}',
            layout=go.Layout(title_text=f'刻蚀演化 (t = {times[mesh_idx]:.3f})')
        ))

    # 添加滑块
    steps = []
    for i, mesh_idx in enumerate(valid_indices):
        step = dict(
            method='animate',
            args=[
                [f'frame_{mesh_idx}'],
                dict(
                    mode='immediate',
                    frame=dict(duration=100, redraw=True),
                    transition=dict(duration=50)
                )
            ],
            label=f'{times[mesh_idx]:.2f}'
        )
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue=dict(prefix="时间 t = "),
        pad=dict(t=50),
        steps=steps
    )]

    # 添加播放按钮
    play_button = dict(
        type='buttons',
        showactive=False,
        buttons=[
            dict(
                label='播放',
                method='animate',
                args=[None, dict(
                    frame=dict(duration=100, redraw=True),
                    fromcurrent=True,
                    transition=dict(duration=50)
                )]
            ),
            dict(
                label='暂停',
                method='animate',
                args=[[None], dict(
                    mode='immediate',
                    frame=dict(duration=0, redraw=False),
                    transition=dict(duration=0)
                )]
            )
        ]
    )

    # 更新布局
    fig.update_layout(
        title='刻蚀动态演化 (时间 t: 0 → 1)',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        updatemenus=[play_button],
        sliders=sliders,
        width=1000,
        height=700
    )

    fig.frames = frames

    if save_path:
        fig.write_html(save_path)
        print(f"✓ 动态演化图已保存到: {save_path}")

    return fig


def calculate_highest_point(extractor, t=2.0, y_0=-0.7):
    """计算特定时刻界面在y方向的最高点"""

    # 提取等值面
    mesh = extractor.extract_isosurface(t)

    if mesh is None:
        print(f"无法在t={t}时提取等值面")
        return None

    # 获取所有顶点
    vertices = mesh.vertices

    if len(vertices) == 0:
        print(f"t={t}时等值面无顶点")
        return None

    # 找到y坐标最大的点（最高点）
    highest_point_idx = np.argmax(vertices[:, 1])  # y方向是第二个坐标轴
    highest_point = vertices[highest_point_idx]

    # 获取y坐标的最大值
    max_y = np.max(vertices[:, 1])-y_0 # 减去底面计算深度度

    # 如果有多个点具有相同的最大y值，可以获取所有最高点
    tolerance = 1e-6
    all_highest_points = vertices[abs(vertices[:, 1] - max_y) < tolerance]

    print(f"时间 t={t}:")
    print(f"  最高点坐标: ({highest_point[0]:.4f}, {highest_point[1]:.4f}, {highest_point[2]:.4f})")
    print(f"  深度: {max_y:.4f}")
    print(f"  顶点总数: {len(vertices)}")
    print(f"  具有最大y值的点数: {len(all_highest_points)}")

    return {
        'max_y': max_y,
        'highest_point': highest_point,
        'all_highest_points': all_highest_points,
        'mesh': mesh,
        'num_vertices': len(vertices)
    }

def test_different_resolutions():
    """测试不同分辨率下的表面提取"""

    print("\n测试不同分辨率...")

    try:
        model = SpaceTimeSIREN.load('models/model_final.pth')
    except:
        print("需要先训练模型")
        return

    resolutions = [32, 64, 128]
    extraction_times = []

    for res in resolutions:
        extractor = SurfaceExtractor(model, resolution=res)

        import time
        start_time = time.time()
        mesh = extractor.extract_isosurface(0.5)
        end_time = time.time()

        extraction_time = end_time - start_time
        extraction_times.append(extraction_time)

        if mesh is not None:
            print(f"  分辨率 {res}×{res}×{res}: {extraction_time:.2f}秒, "
                  f"顶点数: {len(mesh.vertices)}")
        else:
            print(f"  分辨率 {res}×{res}×{res}: 提取失败")

    return extraction_times


if __name__ == "__main__":
    # 检查模型文件是否存在
    # if not os.path.exists('models/model_final.pth'):
    #     print("错误: 未找到训练好的模型文件")
    #     print("请先训练模型")
    # else:
    #     test_loaded_model()
    #     test_different_resolutions()
    test_loaded_model()
    test_different_resolutions()
