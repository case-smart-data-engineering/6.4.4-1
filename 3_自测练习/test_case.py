# encoding:utf-8

import torch
from importlib import import_module

# 定义了一个字典，将文本情感分类结果映射到字符串形式，即：中性、积极和消极
key = {
    0: '中性',
    1: '积极',
    2: '消极'
}

# 定义bert
model_name = 'bert'
# 使用 import_module 函数从model包内加载bert.py文件
x = import_module('1_算法示例.train.model.' + model_name)
# 实例化bert.py文件里的模型
config = x.Config('../1_算法示例/train/model')
# 实例化bert.py文件里的配置对象 并使用cpu运行
model = x.Model(config).to(config.device)
# 完成模型调用
model.load_state_dict(torch.load(config.save_path, map_location='cpu'))


# 文本构造
def build_predict_text(text):
    # 使用tokenizer.tokenize()方法对文本分词
    token = config.tokenizer.tokenize(text)
    # 手动拼接[CLS] [CLS]标志表明句子开始
    token =
    # 获取文本长度
    seq_len =
    # 创建一个空的 mask 列表，该列表用于存储文本的掩码信息
    mask =
    # 使用tokenizer将文本的每个词转换为对应的 ID，并将结果存储到 token_ids 列表中。
    token_ids =
    # 每句话处理成的长度(短填长切)
    pad_size = config.pad_size
    # 如果有长度
    if pad_size:
        # 如果文本长度小于 config.pad_size
        if len(token) < :
            # 则在 token_ids 和 mask 列表的末尾添加 0，以便使文本长度满足模型的要求
            mask = [1] * len(token_ids) + ([0] * (pad_size - len(token)))
            token_ids += ([0] * (pad_size - len(token)))
        # 否则将文本截断为指定长度，并将 mask 列表填充为指定长度。
        else:
            # mask操作
            mask = [1] * pad_size
            token_ids = token_ids[:pad_size]
            seq_len = pad_size
    # 转换ids,seq_len,mask为 PyTorch 的张量并返回
    ids = torch.LongTensor([token_ids]).to(device="cpu")
    seq_len = torch.LongTensor([seq_len]).to(device="cpu")
    mask = torch.LongTensor([mask]).to(device="cpu")

    return


def predict(text):
    """
    单个文本预测
    :param text:
    :return:
    """
    # 文本构造
    data = build_predict_text(text)
    # 开始网络训练
    with torch.no_grad():
        outputs = model(data)
        # print('outputs:', outputs)
        # 在维度上变成最大值的索引
        num =
        # 输出对应的那个索引
        print('num:', num)

    # 返回索引对应的值 即情感
    return key[int(num)]


if __name__ == '__main__':
    # 自定义预测结果
    print(predict('客户：每个月七十八G流量怎么用得完，一直在外面的话才能用完，快给我把套餐改了'))
