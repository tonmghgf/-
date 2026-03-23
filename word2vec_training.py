import re
import jieba
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from collections import Counter
from torch.utils.data import Dataset, DataLoader

# 配置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# -------------------------- 新增：文档核心关键词列表（从教程中提取）--------------------------
CORE_KEYWORDS = [
    # 语言基础
    "Python", "解释型语言", "交互式语言", "面向对象", "语法结构", "关键字", "标识符", "注释", "缩进", "变量",
    "数据类型", "Number", "int", "float", "bool", "complex", "String", "列表", "List", "元组", "Tuple",
    "集合", "Set", "字典", "Dictionary", "bytes", "运算符", "算术运算符", "比较运算符", "赋值运算符",
    # 核心语法
    "条件语句", "if-elif-else", "match-case", "循环语句", "for", "while", "break", "continue", "pass",
    "推导式", "函数", "def", "lambda", "return", "装饰器", "模块", "import", "包", "输入输出", "print",
    "input", "文件操作", "with",
    # 进阶特性
    "迭代器", "生成器", "yield", "异常处理", "try-except", "面向对象", "类", "实例", "继承", "多继承",
    "方法重写", "命名空间", "作用域", "虚拟环境", "类型注解",
    # 工具与应用
    "标准库", "os", "sys", "第三方库", "PyCharm", "VS Code", "Web开发", "数据分析", "算法"
]

# 数据加载与预处理（保持原逻辑，适配文档路径）
def load_and_preprocess_data(file_path):
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            content = re.sub(r'^\d+\s*', '', line.strip())
            if len(content) > 5:
                sentences.append(content)
    return sentences

# 中文分词处理（保持原逻辑）
def chinese_tokenize(sentences):
    tokenized_sentences = []
    for i, sentence in enumerate(sentences):
        words = jieba.lcut(sentence)
        words = [word.strip() for word in words if len(word.strip()) > 1]
        tokenized_sentences.append(words)
        if i % 200 == 0:
            print(f"已处理 {i}/{len(sentences)} 条数据")
    return tokenized_sentences

# 词汇统计分析（保持原逻辑）
def analyze_vocabulary(tokenized_corpus):
    all_words = [word for sentence in tokenized_corpus for word in sentence]
    word_freq = Counter(all_words)
    print("词汇统计信息：")
    print(f"总词汇量: {len(all_words)}")
    print(f"唯一词汇数: {len(word_freq)}")
    print(f"平均句子长度: {np.mean([len(sentence) for sentence in tokenized_corpus]):.2f}")
    print(f"最长句子长度: {max([len(sentence) for sentence in tokenized_corpus])}")
    print(f"最短句子长度: {min([len(sentence) for sentence in tokenized_corpus])}")

    print("\n前20个最高频词汇：")
    for word, freq in word_freq.most_common(20):
        print(f"{word}: {freq}次")
    return word_freq

# 构建词汇表（保持原逻辑）
def build_vocab(tokenized_corpus, min_count=5):
    word_counts = Counter([word for sentence in tokenized_corpus for word in sentence])
    vocab = {word: count for word, count in word_counts.items() if count >= min_count}
    idx_to_word = ['<PAD>', '<UNK>'] + list(vocab.keys())
    word_to_idx = {word: idx for idx, word in enumerate(idx_to_word)}
    print(f"词汇表大小: {len(word_to_idx)} (包含 {len(word_counts) - len(vocab)} 个低频词被过滤)")
    return word_to_idx, idx_to_word, vocab

# 创建训练数据（保持原逻辑）
def create_training_data(tokenized_corpus, word_to_idx, window_size=5, num_negatives=5):
    training_data = []
    vocab_size = len(word_to_idx)
    unk_idx = word_to_idx.get('<UNK>', 0)

    word_counts = np.zeros(vocab_size)
    for word, idx in word_to_idx.items():
        if word in vocab:
            word_counts[idx] = vocab[word]

    word_distribution = np.power(word_counts, 0.75)
    word_distribution = word_distribution / word_distribution.sum()

    for sentence in tokenized_corpus:
        sentence_indices = [word_to_idx.get(word, unk_idx) for word in sentence]
        for i, target_word_idx in enumerate(sentence_indices):
            start = max(0, i - window_size)
            end = min(len(sentence_indices), i + window_size + 1)
            for j in range(start, end):
                if j != i:
                    context_word_idx = sentence_indices[j]
                    training_data.append((target_word_idx, context_word_idx))

    print(f"创建了 {len(training_data)} 个训练样本")
    return training_data, word_distribution

# 自定义Dataset（保持原逻辑）
class Word2VecDataset(Dataset):
    def __init__(self, training_data, word_distribution, num_negatives=5):
        self.training_data = training_data
        self.word_distribution = word_distribution
        self.num_negatives = num_negatives
        self.vocab_size = len(word_distribution)

    def __len__(self):
        return len(self.training_data)

    def __getitem__(self, idx):
        target, context = self.training_data[idx]
        negative_samples = []
        while len(negative_samples) < self.num_negatives:
            negative = np.random.choice(self.vocab_size, p=self.word_distribution)
            if negative != target and negative != context:
                negative_samples.append(negative)

        return {
            'target': torch.tensor(target, dtype=torch.long),
            'context': torch.tensor(context, dtype=torch.long),
            'negatives': torch.tensor(negative_samples, dtype=torch.long)
        }

# Word2Vec模型（保持原逻辑）
class Word2VecModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=50):
        super(Word2VecModel, self).__init__()
        self.target_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)

        init_range = 0.5 / embedding_dim
        self.target_embeddings.weight.data.uniform_(-init_range, init_range)
        self.context_embeddings.weight.data.uniform_(-init_range, init_range)

    def forward(self, target_word, context_word, negative_words):
        target_embed = self.target_embeddings(target_word)
        context_embed = self.context_embeddings(context_word)
        negative_embed = self.context_embeddings(negative_words)

        positive_score = torch.sum(target_embed * context_embed, dim=1)
        positive_score = torch.clamp(positive_score, max=10, min=-10)

        target_embed_expanded = target_embed.unsqueeze(1)
        negative_score = torch.bmm(negative_embed, target_embed_expanded.transpose(1, 2))
        negative_score = torch.clamp(negative_score.squeeze(2), max=10, min=-10)

        return positive_score, negative_score

# 损失函数（保持原逻辑）
def skipgram_loss(positive_score, negative_score):
    positive_loss = -torch.log(torch.sigmoid(positive_score))
    negative_loss = -torch.sum(torch.log(torch.sigmoid(-negative_score)), dim=1)
    return (positive_loss + negative_loss).mean()

# 训练函数（保持原逻辑）
def train_word2vec_gpu(model, dataset, batch_size=1024, epochs=1, learning_rate=0.025):
    model = model.to(device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\n开始训练...")
    print(f"批量大小: {batch_size}")
    print(f"训练轮数: {epochs}")
    print(f"优化器: Adam, 学习率: {learning_rate}")

    losses = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        start_time = time.time()

        for batch_idx, batch in enumerate(dataloader):
            target_words = batch['target'].to(device)
            context_words = batch['context'].to(device)
            negative_words = batch['negatives'].to(device)

            optimizer.zero_grad()
            positive_score, negative_score = model(target_words, context_words, negative_words)
            loss = skipgram_loss(positive_score, negative_score)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")

        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs} 完成 | 平均损失: {avg_loss:.4f} | 时间: {epoch_time:.2f}秒")

        if (epoch + 1) % 10 == 0:
            print(f"保存第 {epoch+1} 轮的词向量...")

    print("\n训练完成！")
    return model, losses

# 提取词向量（保持原逻辑）
def get_word_vectors(model, word_to_idx):
    model.eval()
    with torch.no_grad():
        all_indices = torch.arange(len(word_to_idx)).to(device)
        word_vectors = model.target_embeddings(all_indices).detach().cpu()

    word_vectors_dict = {}
    for word, idx in word_to_idx.items():
        word_vectors_dict[word] = word_vectors[idx]
    return word_vectors_dict, word_vectors

# 保存/加载词向量（适配自定义模型路径）
def save_word_vectors_pt(word_vectors_dict, output_path):
    torch.save(word_vectors_dict, output_path)
    print(f"词向量已保存到: {output_path}")

def load_word_vectors_pt(input_path):
    word_vectors_dict = torch.load(input_path, map_location=device)
    print(f"词向量已从 {input_path} 加载")
    return word_vectors_dict

# Gensim兼容包装器（修复GPU张量问题，保持原逻辑）
class PyTorchWord2VecWrapper:
    def __init__(self, word_vectors_dict, word_to_idx, idx_to_word):
        self.wv = self.WordVectors(word_vectors_dict, word_to_idx, idx_to_word)
        self.vector_size = list(word_vectors_dict.values())[0].shape[0]

    class WordVectors:
        def __init__(self, word_vectors_dict, word_to_idx, idx_to_word):
            self.vectors_dict = word_vectors_dict
            self.word_to_idx = word_to_idx
            self.idx_to_word = idx_to_word
            self.key_to_index = word_to_idx
            vectors_cpu = [vec.cpu().numpy() for vec in word_vectors_dict.values()]
            self.vectors = np.stack(vectors_cpu)

        def __getitem__(self, word):
            return self.vectors_dict.get(word, None)

        def __contains__(self, word):
            return word in self.vectors_dict

        def similarity(self, word1, word2):
            if word1 not in self.vectors_dict or word2 not in self.vectors_dict:
                raise KeyError(f"词语不在词汇表中: {word1} 或 {word2}")
            vec1 = self.vectors_dict[word1].cpu().numpy()
            vec2 = self.vectors_dict[word2].cpu().numpy()
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return np.dot(vec1, vec2) / (norm1 * norm2)

        def most_similar(self, word, topn=10):
            if word not in self.vectors_dict:
                raise KeyError(f"词语不在词汇表中: {word}")
            target_vec = self.vectors_dict[word].cpu().numpy()
            similarities = []
            for w, vec in self.vectors_dict.items():
                if w == word:
                    continue
                norm_target = np.linalg.norm(target_vec)
                norm_vec = np.linalg.norm(vec.cpu().numpy())
                if norm_target == 0 or norm_vec == 0:
                    sim = 0.0
                else:
                    sim = np.dot(target_vec, vec.cpu().numpy()) / (norm_target * norm_vec)
                similarities.append((w, sim))
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:topn]

# -------------------------- 新增：关键词查找功能（基于词向量相似度）--------------------------
def find_related_keywords(w2v_model, query_keywords, topn=5):
    """
    基于词向量相似度查找文档中相关的核心关键词
    :param w2v_model: 训练好的词向量模型
    :param query_keywords: 待查询的关键词列表
    :param topn: 返回每个关键词的topN相关词
    :return: 相关关键词字典
    """
    related_keywords = {}
    for query in query_keywords:
        if query in w2v_model.wv:
            similar_words = w2v_model.wv.most_similar(query, topn=topn)
            # 过滤出在文档核心关键词列表中的相关词
            filtered_similar = [(word, score) for word, score in similar_words if word in CORE_KEYWORDS]
            related_keywords[query] = filtered_similar
    return related_keywords

# ===================== 主运行流程（适配自定义模型路径和关键词查找）=====================
if __name__ == "__main__":
    # 1. 加载数据（适配教程文档路径）
    file_path = r"E:\big model use\anli2\python_tutorial_runoob.txt"
    sentences = load_and_preprocess_data(file_path)
    print(f"总共加载了 {len(sentences)} 条文本数据")
    print("前5条数据示例：")
    for i in range(min(5, len(sentences))):
        print(f"{i+1}: {sentences[i][:50]}...")

    # 2. 中文分词
    tokenized_corpus = chinese_tokenize(sentences)
    print("分词完成！")
    print("\n分词后的数据示例：")
    for i in range(min(3, len(tokenized_corpus))):
        print(f"原文: {sentences[i][:30]}...")
        print(f"分词: {tokenized_corpus[i][:10]}...")

    # 3. 词汇分析
    word_frequency = analyze_vocabulary(tokenized_corpus)

    # 4. 构建词汇表
    word_to_idx, idx_to_word, vocab = build_vocab(tokenized_corpus, min_count=5)

    # 5. 创建训练数据
    training_data, word_distribution = create_training_data(
        tokenized_corpus, word_to_idx, window_size=5, num_negatives=5
    )

    # 6. 创建数据集与模型
    dataset = Word2VecDataset(training_data, word_distribution, num_negatives=5)
    model = Word2VecModel(vocab_size=len(word_to_idx), embedding_dim=50)

    # 7. 训练模型（可跳过训练，直接加载已有模型）
    trained_model, losses = train_word2vec_gpu(
        model, dataset, batch_size=1024, epochs=5, learning_rate=0.025
    )

    # 8. 提取并保存词向量（保存到自定义路径）
    word_vectors_dict, all_vectors = get_word_vectors(trained_model, word_to_idx)
    custom_model_path = r"E:\big model use\anli2\word_vectors.pt"
    save_word_vectors_pt(word_vectors_dict, custom_model_path)

    # 9. 加载自定义路径的词向量
    loaded_word_vectors_dict = load_word_vectors_pt(custom_model_path)

    # 10. 创建包装器并测试
    w2v_model = PyTorchWord2VecWrapper(loaded_word_vectors_dict, word_to_idx, idx_to_word)

    print("\n" + "="*50)
    print("词向量模型测试")
    print("="*50)

    # -------------------------- 新增：文档关键词查找测试 --------------------------
    print("\n" + "-"*30)
    print("文档核心关键词查找（基于词向量相似度）")
    print("-"*30)
    # 选择部分核心关键词作为查询词
    query_keywords = ["Python", "函数", "面向对象", "列表", "异常处理", "模块"]
    related_keywords = find_related_keywords(w2v_model, query_keywords, topn=3)
    for query, related in related_keywords.items():
        if related:
            print(f"\n与'{query}'相关的核心关键词：")
            for word, score in related:
                print(f"  {word}: 相似度 {score:.3f}")
        else:
            print(f"\n'{query}' 未找到相关核心关键词")

    # 原有相似词和相似度测试（保持不变）
    print("\n" + "-"*30)
    print("基础相似词测试")
    print("-"*30)
    test_words = ['Python', '函数', '列表', '字典']
    for word in test_words:
        if word in w2v_model.wv:
            similar_words = w2v_model.wv.most_similar(word, topn=5)
            print(f"\n与'{word}'最相似的词：")
            for similar, score in similar_words:
                print(f"  {similar}: {score:.3f}")
        else:
            print(f"'{word}'不在词汇表中")

    print("\n" + "-"*30)
    print("词汇相似度计算")
    print("-"*30)
    word_pairs = [('函数', 'lambda'), ('列表', '元组'), ('字典', '集合'), ('模块', '包')]
    for word1, word2 in word_pairs:
        if word1 in w2v_model.wv and word2 in w2v_model.wv:
            similarity = w2v_model.wv.similarity(word1, word2)
            print(f"'{word1}' 和 '{word2}' 的相似度: {similarity:.3f}")
        else:
            print(f"词汇对 ({word1}, {word2}) 中有词不在词汇表中")