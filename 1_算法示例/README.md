# 算法示例

## 使用指南

1. 按 `CTRL + P` 打开命令行面板，输入 "terminal: Create New Terminal" 打开一个命令行终端.
2. 在命令行里输入 `pip install -r requirements.txt` 按 `ENTER` 安装示例程序所需依赖库。
> 如果安装报错 可以尝试以下命令升级pip
```
pip install --upgrade pip
pip install --upgrade setuptools
pip install ez_setup
```
3. 在命令行里输入 `cd 1_算法示例` 并按 `ENTER` 进入"算法示例"目录。
4. 通过 https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz 下载bert预处理模型放置于1_算法示例/train/model/bert_pretrain下
5. 在命令行里输入 `python train/run.py` 按 `ENTER` 运行训练模型程序。
6. 模型训练完毕后，在命令行里输入 `python predict.py` 按 `ENTER` 运行示例程序。
