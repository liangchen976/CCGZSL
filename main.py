import torch
from train import *
import torch.backends.cudnn as cudnn
cudnn.benchmark = True


if __name__ == '__main__':
    for i in range(opt.task_num):
        train_one_task()