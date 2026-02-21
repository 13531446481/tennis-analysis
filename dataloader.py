import os

import numpy as np
from torch.utils.data import Dataset


# define dataset
class VideoSkeletonDataset(Dataset):
    def __init__(self, image_path, keypoints_path, input_len, output_len, input_step=1, output_step=1, transform=None):
        self.image_path = image_path
        self.keypoints_path = keypoints_path
        self.input_len = input_len
        self.output_len = output_len
        self.input_step = input_step
        self.output_step = output_step
        self.input_seq_len = (input_len-1) * input_step + 1
        self.output_seq_len = output_len * output_step
        self.seq_len = self.input_seq_len + self.output_seq_len
        self.transform = transform

        self.data_list = self.load_path(keypoints_path)

        self.frame_sum_list = [0]
        for idx, data in enumerate(self.data_list):
            self.frame_sum_list.append(
                self.frame_sum_list[idx] + data.shape[0] - self.seq_len
            )
        self.data_num = self.frame_sum_list[-1]

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        input, output = [], []
        for frame_sum in self.frame_sum_list:
            if idx < frame_sum:
                list_idx = self.frame_sum_list.index(frame_sum) - 1
                start_idx = idx - self.frame_sum_list[list_idx]
                mid_idx = start_idx + self.input_seq_len
                end_idx = start_idx + self.seq_len
                input = self.data_list[list_idx][start_idx: mid_idx: self.input_step]
                output = self.data_list[list_idx][mid_idx+self.output_step: end_idx+1: self.output_step]
                break
        return input, output

    @staticmethod
    def load_path(keypoints_path):
        data_list = []
        for file_name in sorted(os.listdir(keypoints_path)):
            file_path = os.path.join(keypoints_path, file_name)
            data = np.load(file_path)
            data_list.append(data)
        return data_list
