# 20 Newsgroups 二分类任务（1层双向GRU）
小组成员：谢敏 熊丽 田思雨 孟笑然
## 一、运行环境

本项目使用 Python 3.9 和 TensorFlow / Keras 实现，建议运行环境如下：

- Python 3.9
- numpy==1.26.4
- tensorflow==2.17.1
- scikit-learn==1.5.2

安装依赖命令：

```bash
pip install -r requirements.txt
```

---

## 二、项目结构

本项目主要包含以下文件：

```text
第二次作业
├─ 20_news_data.py
├─ requirements.txt
├─ README.md
```

各文件说明如下：

- `20_news_data.py`：主程序文件，包含数据加载、文本预处理、词表构建、模型训练、验证与测试全过程
- `requirements.txt`：项目依赖文件
- `README.md`：项目说明文档

---

## 三、步骤运行

### 1. 安装依赖
在项目根目录下打开终端，运行：

```bash
pip install -r requirements.txt
```

### 2. 运行程序
安装完成后，运行以下命令：

```bash
python 20_news_data.py
```

### 3. 程序主要执行流程
程序运行后，将自动完成以下步骤：

1. 加载 20 Newsgroups 数据集中的两个类别：
   - `alt.atheism`
   - `soc.religion.christian`

2. 对文本进行预处理，包括：
   - 转小写
   - 去除 HTML 标签
   - 去除 URL
   - 去除邮箱地址
   - 去除标点符号
   - 去除数字
   - 去除多余空格

3. 将原始训练集划分为训练集和验证集

4. 基于训练集构建词汇表，并将文本转换为定长序列

5. 构建 1 层双向 GRU 模型进行训练

6. 在验证集上搜索最佳分类阈值

7. 在测试集上评估模型准确率

---

## 四、运行结果

本项目在当前实验设置下，测试集准确率约为：

**0.7601**


程序运行结束后，终端中会输出以下几类信息：

- 词汇表大小
- 训练集、验证集、测试集样本数量
- 模型结构信息
- 每轮训练的损失和准确率
- 验证集最佳阈值
- 测试集最终准确率

---

## 五、说明

### 1. 数据说明
本项目使用 `sklearn.datasets.fetch_20newsgroups` 接口加载数据。  
本次实验选择的类别为：

```python
categories = ['alt.atheism', 'soc.religion.christian']
```

在数据读取时，设置为去除：

- headers
- footers
- quotes

因此该版本实验主要基于**正文内容**完成分类任务。

### 2. 模型说明
本实验使用的是 **1层双向GRU模型**，模型主要结构如下：

1. Embedding 层
2. SpatialDropout1D
3. 1层 Bidirectional GRU
4. GlobalMaxPooling1D
5. 全连接层 + Dropout
6. Sigmoid 输出层

### 3. 优化说明
相较于基础版本，本项目主要做了以下优化：

1. 词汇表仅由训练集构建，避免测试集信息泄漏
2. 增强文本清洗，减少噪声干扰
3. 将最大序列长度设置为 400，保留更多正文内容
4. 使用 `return_sequences=True + GlobalMaxPooling1D`，提升文本特征提取效果
5. 引入 `EarlyStopping` 和 `ReduceLROnPlateau`，提高训练稳定性
6. 在验证集上搜索最佳分类阈值，而不是固定使用 0.5

### 4. 结果说明
本版本属于较为规范的正文分类版本。  
如果保留新闻头部信息（headers），模型准确率通常还能进一步提高；但本项目正式提交版本仍以正文分类为主，更符合一般文本分类任务的实验设置。