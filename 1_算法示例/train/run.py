# coding: UTF-8
import sys
sys.path.append('..')
import time
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from train_eval import train, init_network
from importlib import import_module
from utils import build_dataset, build_iterator, get_time_dif

if __name__ == '__main__':
    # 数据集
    dataset = 'train/model'  
    # 导入bert模块
    x = import_module('model.bert')
    # 导入配置信息
    config = x.Config(dataset)
    # 设置随机数种子
    np.random.seed(1)
    # 设置随机数种子
    torch.manual_seed(1)
    # 使每次执行得到相同的随机数
    torch.cuda.manual_seed_all(1)
    # 保证每次结果一样
    torch.backends.cudnn.deterministic = True

    start_time = time.time()
    print("Loading data...")
    # 加载数据
    train_data, dev_data, test_data = build_dataset(config)
    # 使用 build_iterator() 函数将数据转换为迭代器，方便模型的训练。
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # 在cpu上运行
    model = x.Model(config).to(config.device)
    # 调用train()函数来训练模型
    train(config, model, train_iter, dev_iter, test_iter)
