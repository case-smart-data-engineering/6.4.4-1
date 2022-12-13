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
x = import_module('train.model.' + model_name)
# 实例化bert.py文件里的模型
config = x.Config('train/model')
# 实例化bert.py文件里的配置对象 并使用cpu运行
model = x.Model(config).to(config.device)
# 完成模型调用
model.load_state_dict(torch.load(config.save_path, map_location='cpu'))


# 文本构造
def build_predict_text(text):
    # 使用tokenizer.tokenize()方法对文本分词
    token = config.tokenizer.tokenize(text)
    # 手动拼接[CLS] [CLS]标志表明句子开始
    token = ['[CLS]'] + token
    # 获取文本长度
    seq_len = len(token)
    # 创建一个空的 mask 列表，该列表用于存储文本的掩码信息
    mask = []
    # 使用tokenizer将文本的每个词转换为对应的 ID，并将结果存储到 token_ids 列表中。
    token_ids = config.tokenizer.convert_tokens_to_ids(token)
    # 每句话处理成的长度(短填长切)
    pad_size = config.pad_size
    # 如果有长度
    if pad_size:
        # 如果文本长度小于 config.pad_size
        if len(token) < pad_size:
            # 则在 token_ids 和 mask 列表的末尾添加 0，以便使文本长度满足模型的要求
            mask = [1] * len(token_ids) + ([0] * (pad_size - len(token)))
            token_ids += ([0] * (pad_size - len(token)))
        # 否则将文本截断为指定长度，并将 mask 列表填充为指定长度。
        else:
            # mask操作
            mask = [1] * pad_size
            token_ids = token_ids[:pad_size]
            seq_len = pad_size
    # 将ids seq_len mask转换为 PyTorch 的张量
    ids = torch.LongTensor([token_ids]).to(device="cpu")
    seq_len = torch.LongTensor([seq_len]).to(device="cpu")
    mask = torch.LongTensor([mask]).to(device="cpu")
    # 返回PyTorch张量
    return ids, seq_len, mask


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
        num = torch.argmax(outputs)
        # 输出对应的那个索引
        print('num:', num)

    # 返回索引对应的值 即情感
    return key[int(num)]


if __name__ == '__main__':
    # print(predict('客户：嗯能帮我把套餐改一下嗯。客户：啊，帮我改低一点。客户：百五五八。客户：啊。客户：三八用超了一g多少。'))
    # print(predict('客户：只要那个四g。客户：高速流量是什么意思啊？客户：哦怎么会像这样子整了嘛你们原来不是原来四g就用不用高速低速度，都一样的块怎么现在又搞了什么套餐这个东西是？客户：啊我天你们这个移动怎么搞些管理古董的东西，受理这种四g吗就？就怎么用嘛怎么用嘛是套餐反正都是在这里的你，每个月说实在话的也用不完呢我们这个套餐是。客户：问题想想想嘛每个月七十八怎么用得完呢怎么，一直在外面外面的话才能用，都在家里面啊你看回来都在家里面都用，大家的这种那种，自己的那个wifi呀本来每个月都是，我问那这太难用了现在每天都是用了。'))
    # print(predict('客户：哎。客户：限速，你好，你。客户：点，申请，对。客户：务员。客户：嗯，没收到这个也要，不要。'))
    # print(predict('客户：呃。客户：哦，九十多分，嗯问。客户：那个是。客户：那个手机。客户：就没有什么补贴啊之类的我觉得我好像每个月用的流话费都超过一百五了？。'))
    # print(predict('客户：哦那个不要了。客户：还有。客户：嗯嗯。客户：哦，嗯好的都关掉了嘎。客户：我查，不客气。'))  # 1
    # print(predict('客户：啊哦，你们有些什么套餐呢我想换个便宜一点的？客户：也就没有其他套餐吗？客户：回来了吗？客户：有。客户：哦十八的那种这些什么都没有的流量？'))
    # print(predict('客户：每个月七十八G流量怎么用得完，一直在外面的话才能用完，快给我把套餐改了'))
    print(predict('您好！很高兴为您服务。'))
