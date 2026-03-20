# Word2Vec课程作业

本项目为课程作业第六题，实现了一个基于 Word2Vec 的词向量训练与二维可视化小项目。

##小组信息
- 小组成员：谢敏、田思雨、熊丽、孟笑然

## 运行环境

- Python 3.11

## 项目结构

```text
word2vec_course_project/
├─ data/
│  └─ corpus.txt
├─ models/
├─ output/
├─ main.py
├─ requirements.txt
└─ README.md
```

## 运行步骤

### 1. 创建 conda 环境

```bash
conda create -n word2vec_py311 python=3.11 -y
conda activate word2vec_py311
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 运行主程序

```bash
python main.py
```

## 运行结果

程序运行后会读取 data/corpus.txt 中的语料，训练 Word2Vec 模型，并在 output 文件夹中生成词向量可视化图像。
## 说明

- 程序会先读取 data/corpus.txt 中的本地语料，然后训练 Word2Vec 模型，并将模型保存到 models/word2vec.model。
- 之后，程序会选取 10 个词向量，利用 PCA 将其降到二维，并将可视化结果保存到 output/word_vectors.png。

