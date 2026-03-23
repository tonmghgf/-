# 中文 Word2Vec 词向量训练项目

基于 PyTorch 实现的中文 Word2Vec 词向量训练工具，使用 Skip-gram 模型从文本数据中学习词嵌入。

## 项目简介

本项目用于训练中文词向量模型，通过 Word2Vec 算法从 Python 教程文本中学习词语的向量表示。支持词语相似度计算、关键词查找等功能，可用于自然语言处理任务。

## 项目结构

```
.
├── word2vec_training.py    # 主训练脚本
├── word_vectors.pt         # 训练好的词向量模型文件
├── python_tutorial_runoob.txt  # 训练数据：Python 教程文本
└── README.md               # 项目说明文档
```

## 功能特性

- **Skip-gram 模型**：使用负采样（Negative Sampling）优化训练
- **GPU 加速**：自动检测并使用 CUDA 进行训练加速
- **中文分词**：基于 jieba 的中文分词处理
- **词向量相似度计算**：支持词语相似度查询和最近邻搜索
- **关键词查找**：基于词向量相似度查找相关核心关键词

## 环境依赖

```bash
pip install torch numpy jieba
```

## 安装步骤

1. 克隆或下载项目文件
2. 安装依赖包：
   ```bash
   pip install torch numpy jieba
   ```

## 使用方法

### 1. 训练新模型

运行训练脚本：

```bash
python word2vec_training.py
```

训练流程：
1. 加载并预处理文本数据
2. 中文分词处理
3. 构建词汇表
4. 创建训练数据（Skip-gram 格式）
5. 训练 Word2Vec 模型
6. 保存词向量到 `word_vectors.pt`

### 2. 使用已有模型

在代码中加载已训练的模型：

```python
# 加载词向量
loaded_word_vectors_dict = load_word_vectors_pt("word_vectors.pt")
w2v_model = PyTorchWord2VecWrapper(loaded_word_vectors_dict, word_to_idx, idx_to_word)
```

### 3. 查询相似词

```python
# 查找与"Python"最相似的5个词
similar_words = w2v_model.wv.most_similar("Python", topn=5)

# 计算两个词的相似度
similarity = w2v_model.wv.similarity("函数", "lambda")
```

## 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `embedding_dim` | 50 | 词向量维度 |
| `window_size` | 5 | 上下文窗口大小 |
| `num_negatives` | 5 | 负采样数量 |
| `min_count` | 5 | 最小词频阈值 |
| `batch_size` | 1024 | 训练批次大小 |
| `epochs` | 5 | 训练轮数 |
| `learning_rate` | 0.025 | 学习率 |

## 核心关键词列表

项目预定义了 Python 相关的核心关键词，用于关键词查找功能：

- **语言基础**：Python、数据类型、变量、运算符等
- **核心语法**：条件语句、循环语句、函数、模块等
- **进阶特性**：迭代器、生成器、异常处理、面向对象等
- **工具与应用**：标准库、第三方库、Web开发、数据分析等

## 模型架构

```
Word2VecModel(
  (target_embeddings): Embedding(vocab_size, embedding_dim)
  (context_embeddings): Embedding(vocab_size, embedding_dim)
)
```

## 示例输出

```
文档核心关键词查找（基于词向量相似度）
------------------------------

与'Python'相关的核心关键词：
  解释型语言: 相似度 0.823
  面向对象: 相似度 0.756
  语法结构: 相似度 0.712

与'函数'最相似的词：
  lambda: 0.891
  def: 0.856
  return: 0.834
  装饰器: 0.812
  参数: 0.798
```

## 文件说明

- **word2vec_training.py**：包含完整的数据处理、模型定义、训练和测试代码
- **word_vectors.pt**：PyTorch 保存的词向量字典，可直接加载使用
- **python_tutorial_runoob.txt**：用于训练的原始文本数据

## 注意事项

- 确保安装了所有依赖包
- 训练数据文件路径需要正确设置
- GPU 训练需要 CUDA 支持

## 许可证

MIT License
