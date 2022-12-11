# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
from pytorch_pretrained_bert.optimization import BertAdam


# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    # 遍历网络中的所有参数
    for name, w in model.named_parameters():
        # 通过 exclude 参数来指定是否排除某些参数不进行初始化
        if exclude not in name:
            if len(w.size()) < 2:
                continue
            # 如果遍历到的参数名称中包含 weight 则根据 method 参数指定的方法使用不同的初始化方法
            if 'weight' in name:
                # 如 xavier 初始化这些参数
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                # 或 kaiming 初始化这些参数
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            # 如果遍历到的参数名称中包含 bias
            elif 'bias' in name:
                # 使用常数初始化（将值设为 0）
                nn.init.constant_(w, 0)
            # 最后，如果遍历到的参数既不是权重也不是偏差
            else:
                # 则不做任何操作
                pass

# 定义了模型训练的主函数
def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    model.train()
    # 创建优化器
    param_optimizer = list(model.named_parameters())
    # 附加到模型的参数上
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # 指定不同参数的学习率和正则化项
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=config.learning_rate,
                         warmup=0.05,
                         t_total=len(train_iter) * config.num_epochs)
    # 记录进行到多少batch
    total_batch = 0
    # dev_best_loss
    dev_best_loss = float('inf')
    # 记录上次验证集loss下降的batch数
    last_improve = 0
    # 记录是否很久没有效果提升
    flag = False
    model.train()
    # 进入主循环，并在每个 epoch 内对训练数据进行遍历
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            # 梯度初始化为零
            model.zero_grad()
            # 求交叉熵
            loss = F.cross_entropy(outputs, labels)
            # 反向传播
            loss.backward()
            # 更新模型参数
            optimizer.step()
            # 每一百个batch
            if total_batch % 100 == 0:
                # 是否转移到了CPU上
                true = labels.data.cpu()
                # 从输出中选择概率最大的作为预测类别，然后转移到CPU
                predic = torch.max(outputs.data, 1)[1].cpu()
                # 在训练集上计算准确率
                train_acc = metrics.accuracy_score(true, predic)
                # 在验证集上计算准确率和损失，评估模型。
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                # 如果验证集上的损失有所下降
                if dev_loss < dev_best_loss:
                    # 记录
                    dev_best_loss = dev_loss
                    # 保存模型
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    # 记录下改进的batch数
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                # 会输出当前的迭代次数、训练集损失、训练集准确率、验证集损失、验证集准确率和训练时间
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    test(config, model, test_iter)


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    # 调用评估参数
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    # predict_all
    predict_all = np.array([], dtype=int)
    # labels_all
    labels_all = np.array([], dtype=int)
    # 计算的结果在计算图当中，可以进行梯度反传等操作
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)
