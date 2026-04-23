# 机器学习实验：基于 Word2Vec 的情感预测

## 1. 学生信息
- **姓名**：叶嘉乐
- **学号**：112304260119
- **班级**：数据1231

> 注意：姓名和学号必须填写，否则本次实验提交无效。

---

## 2. 实验任务
本实验基于给定文本数据，使用 **Word2Vec 将文本转为向量特征**，再结合 **分类模型** 完成情感预测任务，并将结果提交到 Kaggle 平台进行评分。

本实验重点包括：
- 文本预处理
- Word2Vec 词向量训练或加载
- 句子向量表示
- 分类模型训练
- Kaggle 结果提交与分析

---

## 3. 比赛与提交信息
- **比赛名称**：Bag of Words Meets Bags of Popcorn
- **比赛链接**：https://www.kaggle.com/competitions/word2vec-nlp-tutorial/overview
- **提交日期**：2026.4.23

- **GitHub 仓库地址**：https://github.com/kskbl747/112304260119yejiale
- **GitHub README 地址**：https://github.com/kskbl747/112304260119yejiale/blob/main/README.md

> 注意：GitHub 仓库首页或 README 页面中，必须能看到“姓名 + 学号”，否则无效。

---

## 4. Kaggle 成绩
请填写你最终提交到 Kaggle 的结果：

- **Public Score**：0.96434
- **Private Score**（如有）：0.96434
- **排名**（如能看到可填写）：

---

## 5. Kaggle 截图
请在下方插入 Kaggle 提交结果截图，要求能清楚看到分数信息。
![alt text]({3EADE71D-65D1-4542-8734-E7C0432EBDF2}.png)
![Kaggle截图](./images/kaggle_score.png)

> 建议将截图保存在 `images` 文件夹中。  
> 截图文件名示例：`2023123456_张三_kaggle_score.png`

---

## 6. 实验方法说明

### （1）文本预处理
请说明你对文本做了哪些处理，例如：
- 分词
- 去停用词
- 去除标点或特殊符号
- 转小写

**我的做法：**  
去除 HTML 标签：使用 BeautifulSoup 解析文本并提取纯文本内容，移除所有 HTML 标签；
大小写归一化：将所有文本转换为小写，避免大小写带来的特征差异；
字符级文本处理：仅保留文本中的小写字母、单引号和空格，移除其他特殊符号、数字等非字母字符；
空白符归一化：将多个连续的空格替换为单个空格，并去除文本首尾的空格；
分别构建了「单词级清洗文本」和「字符级清洗文本」：单词级文本侧重语义层面的清洗，字符级文本保留更多字符粒度信息，用于后续不同维度的特征提取。
---

### （2）Word2Vec 特征表示
请说明你如何使用 Word2Vec，例如：
- 是自己训练 Word2Vec，还是使用已有模型
- 词向量维度是多少
- 句子向量如何得到（平均、加权平均、池化等）

**我的做法：**  
本次实验未直接使用 Word2Vec 训练词向量，而是采用 TF-IDF 进行特征提取（补充方案）：
特征提取维度：
单词级 TF-IDF：分析器为单词（word），采用 1-gram 和 2-gram 组合，最大特征数 50000；
字符级 TF-IDF：分析器为字符窗口（char_wb），采用 3-5 gram 组合，最大特征数 50000；
训练数据：合并标注数据、测试数据、无标注数据的文本进行 TF-IDF 拟合，保证特征空间的完整性；
特征拼接：分别生成单词级和字符级的 TF-IDF 特征矩阵，用于后续分类模型的训练。

---

### （3）分类模型
请说明你使用了什么分类模型，例如：
- Logistic Regression
- Random Forest
- SVM
- XGBoost

并说明最终采用了哪一个模型。

**我的做法：**  
采用「模型融合」策略，结合 Logistic Regression 和 LinearSVC 两个模型的预测结果：
逻辑回归（Logistic Regression）：
基于字符级 TF-IDF 特征训练，参数 C=3，求解器为 liblinear；
输出概率值作为预测结果；
线性支持向量机（LinearSVC）：
基于单词级 TF-IDF 特征训练，参数 C=0.5，最大迭代次数 1000；
输出决策函数值作为预测结果；
模型融合方式：
采用 5 折分层交叉验证（StratifiedKFold）训练两个模型，得到折内验证集预测值和测试集预测值；
对两个模型的预测结果分别进行「排名归一化」（scaled ranks），再按 0.5:0.5 的权重加权融合，最终得到情感预测分数。

---

## 7. 实验流程
请简要说明你的实验流程。

示例：
1. 读取训练集和测试集  
2. 对文本进行预处理  
3. 训练或加载 Word2Vec 模型  
4. 将每条文本表示为句向量  
5. 用训练集训练分类器  
6. 在测试集上预测结果  
7. 生成 submission 文件并提交 Kaggle  

**我的实验流程：**  
数据加载：读取标注数据（labeledTrainData）、测试数据（testData）、无标注数据（unlabeledTrainData）的压缩包，解析 TSV 格式数据；
文本清洗：对所有数据的评论文本进行 HTML 标签去除、大小写归一化、特殊字符过滤、空白符归一化，生成单词级和字符级清洗文本；
TF-IDF 特征拟合：合并所有文本数据，分别训练单词级和字符级 TF-IDF 向量器，生成特征矩阵；
交叉验证训练：
采用 5 折分层交叉验证划分训练集和验证集；
分别训练 Logistic Regression（字符级特征）和 LinearSVC（单词级特征），记录每折验证集预测值和测试集预测值；
模型融合：对两个模型的预测结果进行排名归一化，加权融合后得到最终预测分数；
结果输出：生成包含测试集 ID 和情感分数的 submission.csv 文件，提交至 Kaggle 平台；
模型评估：计算验证集的 OOF AUC 分数，评估模型性能。

---

## 8. 文件说明
请说明仓库中各文件或文件夹的作用。

示例：
- `data/`：存放数据文件
- `src/`：存放源代码
- `notebooks/`：存放实验 notebook
- `images/`：存放 README 中使用的图片
- `submission/`：存放提交文件

**我的项目结构：**
project/
├─ data/                    # 数据文件夹（存放实验所需压缩包）
│  ├─ labeledTrainData.tsv.zip  # 标注训练数据
│  ├─ testData.tsv.zip          # 测试数据
│  └─ unlabeledTrainData.tsv.zip # 无标注训练数据
├─ src/                     # 源代码文件夹
│  └─ main.py               # 主程序（数据加载、清洗、训练、预测）
├─ images/                  # 图片文件夹
│  └─ kaggle_score.png      # Kaggle 提交结果截图
├─ submission/              # 提交文件文件夹
│  └─ submission.csv        # 生成的 Kaggle 提交文件
└─ README.md                # 实验说明文档

