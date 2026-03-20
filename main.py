from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.decomposition import PCA


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / 'data' / 'corpus.txt'
MODEL_DIR = BASE_DIR / 'models'
OUTPUT_DIR = BASE_DIR / 'output'
MODEL_PATH = MODEL_DIR / 'word2vec.model'
FIGURE_PATH = OUTPUT_DIR / 'word_vectors.png'


def load_corpus(file_path: Path):
    if not file_path.exists():
        raise FileNotFoundError(f'没有找到语料文件: {file_path}')

    sentences = []
    with file_path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tokens = line.split()
            if tokens:
                sentences.append(tokens)

    if not sentences:
        raise ValueError('语料文件为空，无法训练 Word2Vec 模型。')
    return sentences


def train_model(sentences):
    model = Word2Vec(
        sentences=sentences,
        vector_size=50,
        window=4,
        min_count=1,
        workers=1,
        sg=1,
        epochs=300,
        negative=5,
        seed=42,
    )
    return model


def choose_words(model):
    candidate_words = [
        'python', 'github', 'project', 'data', 'model',
        'word', 'vector', 'city', 'fruit', 'animal'
    ]

    words = [w for w in candidate_words if w in model.wv]
    if len(words) < 10:
        for w in model.wv.index_to_key:
            if w not in words:
                words.append(w)
            if len(words) == 10:
                break
    return words


def visualize_words(model, words, figure_path: Path):
    vectors = [model.wv[word] for word in words]
    pca = PCA(n_components=2, random_state=42)
    reduced = pca.fit_transform(vectors)

    plt.figure(figsize=(10, 8))
    for i, word in enumerate(words):
        x, y = reduced[i, 0], reduced[i, 1]
        plt.scatter(x, y)
        plt.text(x + 0.01, y + 0.01, word, fontsize=10)

    plt.title('Word2Vec Visualization of 10 Word Vectors')
    plt.xlabel('PCA Dimension 1')
    plt.ylabel('PCA Dimension 2')
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300)
    plt.close()


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print('1. 读取语料...')
    sentences = load_corpus(DATA_PATH)
    print(f'   语句数量: {len(sentences)}')

    print('2. 训练 Word2Vec 模型...')
    model = train_model(sentences)
    model.save(str(MODEL_PATH))
    print(f'   模型已保存到: {MODEL_PATH}')

    print('3. 选择 10 个词向量并做二维可视化...')
    words = choose_words(model)
    visualize_words(model, words, FIGURE_PATH)
    print('   选中的词:')
    print('   ' + ', '.join(words))
    print(f'   图像已保存到: {FIGURE_PATH}')

    print('4. 输出每个词向量的前 5 维，便于检查模型结果:')
    for word in words:
        preview = model.wv[word][:5]
        print(f'   {word}: {preview}')

    print('\n运行完成。你现在可以打开 output/word_vectors.png 查看 10 个词向量的分布图。')


if __name__ == '__main__':
    main()
