# coding: UTF-8
import torch
from tqdm import tqdm
import time
from datetime import timedelta

# padding符号, bert中综合信息符号
PAD, CLS = '[PAD]', '[CLS]'

# build_dataset 函数从文件中加载数据，处理它，并将其作为训练、开发和测试数据集的元组返回
def build_dataset(config):

    def load_dataset(path, pad_size=32):
        # 定义一个内部函数，用于加载指定路径的文本数据集
        contents = []
        # 读取文件
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                # 对于每行，去除两边的空白字符，并判断是否为空行
                lin = line.strip()
                if not lin:
                    continue
                # 按照制表符(\t)分割每行，得到内容和标签
                content, label = lin.split('\t')
                # 对内容进行分词
                token = config.tokenizer.tokenize(content)
                # 在分词列表的开头添加[CLS]标记
                token = [CLS] + token
                # 拿到文本长度
                seq_len = len(token)
                # mask
                mask = []
                # 将分词列表转化为对应的词编号列表
                token_ids = config.tokenizer.convert_tokens_to_ids(token)
                # 判断是否需要对序列进行补全
                if pad_size:
                    if len(token) < pad_size:
                        # 对序列进行补全，并生成mask标记列表
                        mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                        token_ids += ([0] * (pad_size - len(token)))
                    else:
                        mask = [1] * pad_size
                        token_ids = token_ids[:pad_size]
                        seq_len = pad_size
                contents.append((token_ids, int(label), seq_len, mask))
        return contents
    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return train, dev, test

# DatasetIterater类是数据集的迭代器。它有一个构造函数，接受数据批次列表，批量大小和用于张量计算的设备
class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        # 每个 batch 中包含的数据量
        self.batch_size = batch_size
        # 原始的数据集，由多个batch组成
        self.batches = batches
        # 计算数据集中batch的数量
        self.n_batches = len(batches) // batch_size
        # 记录batch数量是否为整数
        self.residue = False
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        # 表示数据将在哪个设备上进行处理
        self.device = device

    # 将输入的数据转换为 PyTorch 的 tensor
    def _to_tensor(self, datas):
        # 输入的序列的开始位置
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        # 输入的序列的结束位置
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        # mask
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        return (x, seq_len, mask), y

    def __next__(self):
        # 如果有剩余的批次，并且当前索引指向的是最后一个批次，则返回剩余的批次
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches
        # 如果当前索引已经大于等于总批次数，则重置索引并抛出 StopIteration 异常
        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        # 否则返回当前索引指向的批次
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches

# build_iterator 函数为给定的数据集和配置创建并返回 DatasetIterater 类的实例
def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter

# get_time_dif 函数接受开始时间并返回当前时间的时间差。它可能用于测量执行某个操作所用的时间
def get_time_dif(start_time):
    # 获取已使用时间
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
