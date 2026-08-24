#!/usr/bin/env python3
"""
主训练脚本 - 水平集神经网络刻蚀模拟
"""

import os
import torch

from models.neural_siren import SpaceTimeSIREN
from models.etching_models import EtchingRateModel
from training.loss_functions import LevelSetLoss
from training.trainer import EtchingTrainer
from visualization.surface_visualizer import SurfaceExtractor, plot_training_loss
from utils.geometry import InitialPlane2
from config.default_config import TrainingConfig

def setup_directories():
    """创建必要的目录"""
    directories = ['models', 'visualizations', 'checkpoints']
    for dir_name in directories:
        os.makedirs(dir_name, exist_ok=True)


def main():
    """主训练函数"""
    print("=== 水平集神经网络刻蚀模拟 ===")

    # 设置目录
    setup_directories()

    # 配置
    config = TrainingConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 预训练文件路径
    model_path = 'models/final_model.pth'

    # 检查文件是否存在
    if os.path.exists(model_path):
        print(f"加载已有模型: {model_path}")
        model = SpaceTimeSIREN.load(model_path).to(device)
    else:
        print("创建新模型")
        model = SpaceTimeSIREN(
            hidden_layers=config.hidden_layers,
            hidden_dim=config.hidden_dim,
            r = config.radius, alpha =config.alpha
        ).to(device)

    etching_model = EtchingRateModel()
    loss_fn = LevelSetLoss(
        lambda_data=config.lambda_data,

        lambda_pde=config.lambda_pde,
        lambda_eikonal=config.lambda_eikonal
    )

    trainer = EtchingTrainer(model, loss_fn, etching_model, r=config.radius , alpha = 2.0) # 时序数据调用 temporal_data_path='levelset_temporal_data.h5'

    # 初始表面
    initial_surface = InitialPlane2( y_interface=-0.7)

    # 训练
    print("开始训练...")
    losses = trainer.train(
        initial_surface,
        num_epochs=config.num_epochs,
        batch_size=config.batch_size,
        save_interval=config.save_interval
    )

    # 可视化结果
    print("生成可视化结果...")
    extractor = SurfaceExtractor(model, resolution=config.visualization_resolution)

    # 绘制训练损失
    plot_training_loss(losses, 'visualizations/training_loss')

    # 提取演化序列
    time_steps = [0.0, 0.2,0.5, 1.0, 1.5]
    surfaces = []

    for t in time_steps:
        surface = extractor.extract_isosurface(t)
        if surface is not None:
            surfaces.append(surface)
            surface.export(f'visualizations/etching_surface_t_{t:.1f}.obj')

    # 创建3D可视化
    figures = extractor.plot_evolution_sequence(time_steps)

    # 保存模型
    model.save('models/final_model.pth')

    print("训练完成！")
    print(f"最终损失: {losses[-1]:.6f}")
    print(f"生成的表面数量: {len(surfaces)}")

    return model, surfaces, losses


if __name__ == "__main__":
    model, surfaces, losses = main()