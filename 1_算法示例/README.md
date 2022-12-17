# 算法示例

## 使用指南

1. 按 `CTRL + P` 打开命令行面板，输入 `terminal: Create New Terminal` 打开一个命令行终端 或者按` CTRL + `直接打开命令行终端。
```
如果快捷键没有反应，请手动点击最下面那行状态栏的第四个图标，然后在弹出的tab里点击TERMINAL打开命令行终端。
```
2. 在命令行里输入 `pip install -r requirements.txt` 按 `ENTER` 安装示例程序所需依赖库。
> 如果安装报错，可以尝试以下命令升级pip：
```
pip install --upgrade pip
pip install --upgrade setuptools
pip install ez_setup
```
3. 在命令行里输入 `cd 1_算法示例` 并按 `ENTER` 进入"算法示例"目录。
4. 已经通过git lfs 上传模型文件至github上，gitpod无法访问，显示的是指针型文件，需要手动从github上下载放置于指定路径中或者将本项目克隆到本地运行。
```
# 如果是在gitpod上运行，按以下步骤操作：
# 下载预训练模型pytorch_model.bin放置于1_算法示例/train/model/saved_model下,下载链接：
https://github.com/case-smart-data-engineering/6.4.4-1/blob/main/1_%E7%AE%97%E6%B3%95%E7%A4%BA%E4%BE%8B/train/model/bert_pretrain/pytorch_model.bin
# 如果是克隆到本地运行，按以下步骤操作：
# 直接把1_算法示例/train/model/bert_pretrain/pytorch_model.bin文件复制粘贴于1_算法示例/train/model/saved_model下。
```
5. 在命令行里输入 `python train/run.py` 按 `ENTER` 运行训练模型程序，或者跟4一样下载我们已经训练好的bert模型。
```
# 如果是运行run.py文件训练的话，会自动在该目录下生成模型文件。
# 也可以不用手动训练模型，直接用我们训练好的模型。
# 下载bert模型bert.ckpt放置于1_算法示例/train/model/saved_model下,下载链接：
https://github.com/case-smart-data-engineering/6.4.4-1/blob/main/1_%E7%AE%97%E6%B3%95%E7%A4%BA%E4%BE%8B/train/model/saved_dict/bert.ckpt
# 如果是克隆到本地运行，按以下步骤操作：
# 直接把1_算法示例/train/model/saved_dict/bert.ckpt文件复制粘贴于1_算法示例/train/model/saved_model下。
```
6. 模型训练完毕后，在命令行里输入 `python predict.py` 按 `ENTER` 运行示例程序。

## 运行结果
正确运行代码后，会给文本给出一个情感预测结果。

## 备注
如果是在命令行上运行代码，需要按照使用指南的顺序进入`正确的目录`才可运行成功。
