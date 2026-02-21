import torch
import torch.nn as nn


class MEDELoss(nn.Module):
    def __init__(self):
        super(MEDELoss, self).__init__()

    def forward(self, pred, target):
        """
        计算平均误差: mean_t,j(sqrt((x_pred-x_target)^2 + (y_pred-y_target)^2)).

        Args:
            pred (torch.Tensor): shape: (n, t, f). where:
                'n' is batch size;
                't' is number of frames;
                'f' is total number of features, example: joint=(17, 2) -> features=34,
                ((x_1, y_1), ..., (x_j, y_j)) -> (x_1, y_1, ..., x_j, y_j)
            target (torch.Tensor): same as pred
        Returns:
            torch.Tensor: {l_1, l_2, ..., l_n}
        """
        # 计算欧几里得距离
        diff = (pred - target).view(pred.size(0), pred.size(1), -1, 2)  # [n, t, f] -> [n, t, j, 2]
        distance = torch.sqrt(torch.sum(diff ** 2, dim=-1))  # [n, t, j]

        # 在时间帧和关节维度上取均值
        mean_per_sample = torch.mean(distance, dim=(-1, -2))  # [n]

        # 对批量维度取均值
        mean_error = torch.mean(mean_per_sample)  # 标量
        return mean_error
