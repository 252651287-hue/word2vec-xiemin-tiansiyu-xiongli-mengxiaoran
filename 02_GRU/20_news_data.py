import re
import string
import random
import numpy as np
import tensorflow as tf

from collections import Counter
from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    GRU,
    Dense,
    Dropout,
    Bidirectional,
    SpatialDropout1D,
    GlobalMaxPooling1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


def set_seed(seed=42):
    """固定随机种子，保证结果尽量可复现"""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def preprocess_text(text):
    """文本预处理"""
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)                 # 去HTML标签
    text = re.sub(r'http\\S+|www\\.\\S+', ' ', text)     # 去URL
    text = re.sub(r'\\S+@\\S+', ' ', text)              # 去邮箱
    text = text.translate(str.maketrans('', '', string.punctuation))  # 去标点
    text = re.sub(r'\\d+', ' ', text)                   # 去数字
    text = ' '.join(text.split())                      # 去多余空格
    return text


def build_vocab(texts, min_freq=3, max_vocab_size=20000):
    """构建词汇表：只保留训练集中的高频词"""
    word_freq = Counter()
    for text in texts:
        words = text.split()
        word_freq.update(words)

    word_to_idx = {'<PAD>': 0, '<UNK>': 1}

    # 先按词频过滤，再按频率降序截断词表大小
    valid_words = [(word, freq) for word, freq in word_freq.items() if freq >= min_freq]
    valid_words = sorted(valid_words, key=lambda x: x[1], reverse=True)

    if max_vocab_size is not None:
        valid_words = valid_words[:max_vocab_size - 2]

    for word, _ in valid_words:
        word_to_idx[word] = len(word_to_idx)

    return word_to_idx


def text_to_sequences(texts, word_to_idx, max_seq_len):
    """文本转整数序列并padding"""
    sequences = []
    for text in texts:
        words = text.split()
        seq = [word_to_idx.get(word, 1) for word in words]   # 1对应<UNK>
        sequences.append(seq)

    padded_sequences = pad_sequences(
        sequences,
        maxlen=max_seq_len,
        padding='post',
        truncating='post'
    )
    return padded_sequences


def load_and_preprocess_data():
    """加载数据、划分训练/验证/测试集，并完成序列化"""
    categories = ['alt.atheism', 'soc.religion.christian']

    newsgroups_train = fetch_20newsgroups(
        subset='train',
        categories=categories,
        remove=('headers', 'footers', 'quotes')
    )
    newsgroups_test = fetch_20newsgroups(
        subset='test',
        categories=categories,
        remove=('headers', 'footers', 'quotes')
    )

    X_train_raw = [preprocess_text(doc) for doc in newsgroups_train.data]
    X_test_raw = [preprocess_text(doc) for doc in newsgroups_test.data]
    y_train_raw = newsgroups_train.target
    y_test_raw = newsgroups_test.target

    label_encoder = LabelEncoder()
    y_train_raw = label_encoder.fit_transform(y_train_raw)
    y_test_raw = label_encoder.transform(y_test_raw)

    # 训练集内部再拆一个验证集
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_raw,
        y_train_raw,
        test_size=0.2,
        random_state=42,
        stratify=y_train_raw
    )

    # 只用训练集建词表
    word_to_idx = build_vocab(
        X_train_split,
        min_freq=3,
        max_vocab_size=20000
    )
    vocab_size = len(word_to_idx)

    # 增大最大长度，尽量保留更多正文信息
    max_seq_len = 400

    X_train = text_to_sequences(X_train_split, word_to_idx, max_seq_len)
    X_val = text_to_sequences(X_val_split, word_to_idx, max_seq_len)
    X_test = text_to_sequences(X_test_raw, word_to_idx, max_seq_len)

    y_train = np.array(y_train_split)
    y_val = np.array(y_val_split)
    y_test = np.array(y_test_raw)

    print(f"词汇表大小: {vocab_size}")
    print(f"训练样本: {len(X_train)}")
    print(f"验证样本: {len(X_val)}")
    print(f"测试样本: {len(X_test)}")
    print(f"最大序列长度: {max_seq_len}")

    return X_train, X_val, X_test, y_train, y_val, y_test, word_to_idx, vocab_size, max_seq_len


def build_gru_model(vocab_size, embedding_dim, gru_units, learning_rate=0.001):
    """
    1层双向GRU模型
    说明：
    - 这里只有1层GRU，只是外面加了Bidirectional
    - return_sequences=True 后接 GlobalMaxPooling1D，更适合长文本分类
    """
    model = Sequential()

    model.add(Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        mask_zero=True
    ))

    model.add(SpatialDropout1D(0.2))

    model.add(Bidirectional(
        GRU(
            units=gru_units,
            dropout=0.2,
            recurrent_dropout=0.0,
            return_sequences=True
        )
    ))

    model.add(GlobalMaxPooling1D())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def find_best_threshold(y_true, y_prob):
    """在验证集上寻找最佳分类阈值"""
    best_thr = 0.5
    best_acc = 0.0

    for thr in np.arange(0.40, 0.61, 0.01):
        y_pred = (y_prob >= thr).astype(int)
        acc = accuracy_score(y_true, y_pred)
        if acc > best_acc:
            best_acc = acc
            best_thr = thr

    return best_thr, best_acc


if __name__ == "__main__":
    set_seed(42)

    X_train, X_val, X_test, y_train, y_val, y_test, word_to_idx, vocab_size, max_seq_len = load_and_preprocess_data()

    # 实验参数
    embedding_dim = 128
    gru_units = 96
    learning_rate = 0.001
    batch_size = 32
    epochs = 12

    model = build_gru_model(vocab_size, embedding_dim, gru_units, learning_rate)
    print(model.summary())

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=1,
        min_lr=1e-5,
        verbose=1
    )

    print("\n开始训练模型...")
    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    print("\n先在验证集上寻找最佳阈值...")
    y_val_prob = model.predict(X_val).ravel()
    best_thr, best_val_acc = find_best_threshold(y_val, y_val_prob)
    print(f"验证集最佳阈值: {best_thr:.2f}")
    print(f"验证集最佳准确率: {best_val_acc:.4f}")

    print("\n评估测试集效果...")
    y_test_prob = model.predict(X_test).ravel()
    y_test_pred = (y_test_prob >= best_thr).astype(int)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print(f"测试集准确率: {test_accuracy:.4f}")