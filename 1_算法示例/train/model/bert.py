# coding: UTF-8

import sys
sys.path.append('bert_pretrain')
sys.path.append('model')

import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel, BertTokenizer


class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        # 使用bert模型
        self.model_name = 'bert'
        # 训练集
        self.train_path = dataset + '/data/train.txt'
        # 验证集
        self.dev_path = dataset + '/data/dev.txt'
        # 测试集
        self.test_path = dataset + '/data/test.txt'
        # 类别名单
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt').readlines()]
        # 模型训练结果
        self.save_path = dataset + '/saved_model/' + self.model_name + '.ckpt'
        # 在cpu设备上运行
        self.device = torch.device('cpu')
        # 若超过1000batch效果还没提升，则提前结束训练
        self.require_improvement = 1000
        # 类别数
        self.num_classes = len(self.class_list)
        # epoch数
        self.num_epochs = 10
        # mini-batch大小
        self.batch_size = 16
        # 每句话处理成的长度(短填长切)，文本不足128个字符自动填充，超过就切掉后面的部分
        self.pad_size = 128
        # 学习率
        self.learning_rate = 5e-5
        # bert预处理模型路径
        self.bert_path = dataset + '/saved_model'
        # 加载bert模型
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        # 定义隐层维数
        self.hidden_size = 768


class Model(nn.Module):

    """模型参数"""
    def __init__(self, config):
        # 初始化
        super(Model, self).__init__()
        # 加载bert模型
        self.bert = BertModel.from_pretrained(config.bert_path)
        # bert模型参数提取
        for param in self.bert.parameters():
            param.requires_grad = True
        # 外接全连接层
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        # 输入的句子
        context = x[0]
        # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        mask = x[2]
        # 控制是否输出所有encoder层的结果
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        # 得到 1 0 分类的结果
        out = self.fc(pooled)
        # 返回结果
        return out
