# Transformer 英法翻译作业
1. 以 Transformer 为框架，实现英文到法文的翻译任务；
2. 将原始 Transformer 中的点积注意力机制改为加性注意力机制，并比较两者效果。
小组成员：谢敏 熊丽 田思雨 孟笑然
---

## 一、项目结构

```text
03_Transformer/
├── README.md
├── requirements.txt
├── main-task1.py
├── task2.py
├── data/
│   ├── eng-fra.txt
│   ├── eng-fra_train_data.txt
│   └── eng-fra_test_data.txt
├── results/
│   ├── task1_result.txt
│   ├── task2_result.txt
│   ├── task1_terminal_result.png
│   ├── task2_terminal_result.png
│   └── task2_loss_compare.png
└── models/
    ├── task1_best_dot_model.pt
    ├── task2_best_dot_model.pt
    └── task2_best_additive_model.pt
```
## 二、运行环境
Python 3.x
PyTorch
NumPy
Matplotlib
NLTK

安装依赖：
pip install -r requirements.txt


## 三、数据集说明

本实验使用英法平行语料数据集，包含以下文件：

eng-fra.txt
eng-fra_train_data.txt
eng-fra_test_data.txt

其中：

eng-fra_train_data.txt 用于训练
eng-fra_test_data.txt 用于测试
eng-fra.txt 用于保存完整语料

## 四、Task 1：Transformer 点积注意力翻译任务
1、测试集结果
BLEU-1：65.10
BLEU-4：37.41

2、结果分析
本实验使用标准 Transformer 架构完成英文到法文的翻译任务，并采用点积注意力机制作为原始注意力形式。从训练过程看，训练损失和验证损失均随 epoch 增加而持续下降，说明模型能够稳定收敛并逐步学习英法句对之间的映射关系。
最终在测试集上，模型取得了 BLEU-1 为 65.10、BLEU-4 为 37.41 的结果。结果表明，该模型不仅能够较好地完成单词级别的翻译匹配，同时也具备一定的短语和句子结构生成能力，能够较为有效地完成英法翻译任务。

3、翻译示例
英文：I gave you my word.
真实法文：Je t'ai donné ma parole.
模型法文：je vous ai donné ma parole .

英文：I liked your idea and adopted it.
真实法文：J'ai aimé ton idée et l'ai adopté.
模型法文：je l ' ai apprécié et l ' ai adopté .

英文：Let me help you put on your coat.
真实法文：Laissez-moi vous aider à mettre votre manteau.
模型法文：laissez - moi vous aider à votre manteau .

英文：How stupid he is!
真实法文：Quel imbécile il fait !
模型法文：c ' est stupide !

英文：I think that's highly unlikely.
真实法文：Je pense que c'est hautement improbable.
模型法文：je pense qu ' il est improbable que c ' est improbable .

英文：If you do not have this book, you can buy it.
真实法文：Si vous ne disposez pas de ce livre, vous pouvez en faire l'acquisition.
模型法文：si tu n ' as pas ce livre , tu peux l ' acheter .

英文：Are you seriously thinking about selling this on eBay?
真实法文：Pensez-vous sérieusement à vendre ça sur eBay ?
模型法文：pensez - vous sérieusement à vendre cela sur cette tokyo ?

英文：Money is welcome everywhere.
真实法文：L'argent est bienvenu partout.
模型法文：l ' argent est la bienvenue partout .



## 五、Task 2：点积注意力与加性注意力对比实验
1. 运行方式
python task2.py
2. 最终结果
| 注意力类型 | BLEU-1(%) | BLEU-4(%) | 平均 epoch 耗时(s) | 最优验证损失 |
| ----- | --------- | --------- | -------------- | ------ |
| 点积注意力 | 64.01     | 36.42     | 1698.06        | 2.6547 |
| 加性注意力 | 64.16     | 35.98     | 1491.31        | 2.6553 |
3. 结果分析
实验结果表明，两种注意力机制在本实验中的整体翻译效果非常接近。
从单词级指标看，加性注意力在 BLEU-1 上略高于点积注意力，说明其在局部词语匹配上略有优势；但从句子级指标 BLEU-4 以及最优验证损失来看，点积注意力略优于加性注意力，说明其在句子整体结构建模和最终翻译质量上略占优势。
总体而言，在本实验设置下，将点积注意力替换为加性注意力并未带来显著性能提升，说明两种注意力机制在该英法翻译任务中的表现差异较小。

4. 样例说明
从点积注意力与加性注意力的翻译样例看，两种模型在多数句子上的输出较为相似，均能够生成较合理的法语句子，但在复杂句式和细节搭配上仍存在一定误差。这也进一步说明，在本实验设置下，两种注意力机制在实际翻译效果上的差异不明显。


## 六、结果文件说明
results/task1_result.txt：Task 1 结果总结
results/task2_result.txt：Task 2 结果总结
results/task1_terminal_result.png：Task 1 终端运行截图
results/task2_terminal_result.png：Task 2 终端运行截图
results/task2_loss_compare.png：Task 2 损失曲线对比图


## 七、模型文件说明
models/task1_best_dot_model.pt：Task 1 中点积注意力最优模型
models/task2_best_dot_model.pt：Task 2 中点积注意力最优模型
models/task2_best_additive_model.pt：Task 2 中加性注意力最优模型