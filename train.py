import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import logging
import argparse

from dataloader import VideoSkeletonDataset
from pose_predict_model import PosePredictModel
from loss import MEDELoss

from memory_profiler import profile


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def reshape_input(input_vect):
    batch, frames, _, keypoints, _ = input_vect.size()
    return input_vect[:, :, 0, :, :].view(batch, frames, keypoints * 2).float()


def create_parser():
    parser = argparse.ArgumentParser(description="Pose Prediction Model Training")
    parser.add_argument('--input_len', type=int, default=10, help='Length of input sequence (default: 10)')
    parser.add_argument('--output_len', type=int, default=5, help='Length of output sequence (default: 5)')
    parser.add_argument('--input_step', type=int, default=1, help='Step size for input sequence (default: 1)')
    parser.add_argument('--output_step', type=int, default=1, help='Step size for output sequence (default: 1)')
    return parser


if __name__ == "__main__":
    # @profile
    def train(args):
        image_path = './tennis/imageFiles'
        keypoints_path = './output/keypoints'

        # 使用命令行参数
        input_len = args.input_len
        output_len = args.output_len
        input_step = args.input_step
        output_step = args.output_step

        # 创建数据集和数据加载器
        batch_size = 16
        learning_rate = 0.0001
        dataset = VideoSkeletonDataset(image_path, keypoints_path, input_len, output_len, input_step, output_step)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = PosePredictModel(input_len, output_len)
        criterion = MEDELoss()
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        print('create model:')
        print(model)

        print(f"Total Parameters: {count_parameters(model):,}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        num_epochs = 200
        avgloss_list = []

        model_info = f'epo{num_epochs}_len_{input_len}_{output_len}_step_{input_step}_{output_step}'
        log_filename = f'{model_info}.log'

        # 配置日志记录
        logging.basicConfig(
            filename=log_filename,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        logging.info(f"Model Info: {model_info}")
        logging.info(f"Batch Size: {batch_size}, Learning Rate: {learning_rate}")

        with tqdm(total=num_epochs, desc="Total Progress") as epoch_bar:
            for epoch in range(num_epochs):

                with tqdm(total=len(data_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False) as batch_bar:
                    loss_list = []
                    for input, labels in data_loader:
                        optimizer.zero_grad()  # 清空梯度
                        input = reshape_input(input).to(device)  # 调整输入形状
                        labels = reshape_input(labels).to(device)
                        output = model(input, labels)  # 前向传播
                        loss = criterion(output, labels)  # 计算损失
                        loss_list.append(loss.item())
                        loss.backward()  # 反向传播
                        optimizer.step()  # 更新参数

                        batch_bar.update(1)

                avg_loss = torch.mean(torch.tensor(loss_list)).item()
                avgloss_list.append(avg_loss)
                epoch_bar.set_postfix(loss=f"{avg_loss:.4f}")
                epoch_bar.update(1)

                logging.info(f"Epoch {epoch + 1}/{num_epochs} finished with average loss: {avg_loss:.4f}")

        print(avgloss_list)

        # 假设你正在训练模型并希望保存模型和优化器状态
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, f'./output/{model_info}.pth')


    parser = create_parser()
    args = parser.parse_args()
    train(args)
