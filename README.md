以下是基于您的论文内容修改后的中文README文档：

---

# KFC-Net (Knowledge Fusion & Compression Net)

本仓库为河海大学本科毕业论文《基于知识蒸馏的多任务生物医学命名实体识别方法研究》的代码实现。论文提出了一种融合知识蒸馏与多任务学习的轻量化框架 **KFC-Net**，通过概率空间聚合策略与混合损失优化，解决了生物医学命名实体识别中的标签异构性与模型冗余问题，在主流生物医学数据集上实现了优异的性能与高效的部署能力。

## 🔧 环境配置

通过以下命令安装依赖：
```bash
pip install -r requirements.txt
```

## 💻 使用指南

### 预训练模型
在 **NCBI-Disease**（疾病）、**BC5CDR-Chem**（化学物质）、**BC2GM**（基因）数据集上训练的KFC-Net学生模型已发布于[HuggingFace Hub](https://huggingface.co/Resfir/KFC-Net-bio)。

### 完整训练流程

1. **训练单任务教师模型**  
   以NCBI-Disease数据集（疾病实体）为例：
   ```bash
   python train_teacher.py \
   --data_dir data/NCBI-disease \
   --output_dir models/Teachers/NCBI-disease \
   --logging_dir logs/Teachers/NCBI-disease
   ```
   
2. **生成多任务统一数据集**  
   聚合多数据集并转换标签至统一格式：
   
   ```bash
   python generate_global_datasets.py \
   --input_dir data \
   --output_dir data/GLOBAL
   ```
   
3. **聚合教师模型概率分布**  
   基于独立性假设融合多教师预测分布（核心创新）：
   ```bash
   python generate_teachers_distributions.py \
   --data_dir data/GLOBAL \
   --teachers_dir models/Teachers \
   --model_name_or_path roberta-base \
   ```
   
4. **训练KFC-Net学生模型**  
   使用混合损失（KL散度 + 真实标签）训练轻量化学生：
   ```bash
   python train_student.py \
   --data_dir data/GLOBAL/Student \
   --model_name_or_path distilbert-base \
   --output_dir models/Student \
   ```

### 快速推理
加载训练好的模型进行预测：
```python
from transformers import pipeline
ner_pipeline = pipeline("ner", model="Resfir/KFC-Net-bio")
text = "EGFR gene mutations are closely related to lung cancer and can be targeted for treatment with gefitinib."
results = ner_pipeline(text)
```

## 📊 性能对比

KFC-Net在生物医学实体识别任务中达到SOTA性能：

| 数据集       | F1值   | 精确率 | 召回率 |
| ------------ | ------ | ------ | ------ |
| NCBI-Disease | 88.34% | 86.87% | 89.86% |
| BC5CDR-Chem  | 93.62% | 94.48% | 92.77% |
| BC2GM        | 83.84% | 83.29% | 84.40% |

框架支持 **RoBERTa-base**（473MB）、**DistilBERT**（253MB）、**TinyBERT**（54MB）等多种轻量化学生模型，在边缘设备上推理速度最高可达7200样本/秒。

## 📍 核心创新

- **概率空间聚合策略**  
  通过数学建模解决多教师预测冲突，公式：  
  $$\mathcal{A}(B\text{-}e_i|x) = p_B^i \cdot \prod_{j \neq i}(p_I^j + p_O^j)$$  
  确保多任务标签的互斥性与一致性。

- **混合损失优化**  
  联合优化知识蒸馏损失（$\mathcal{L}_{KD}$）与真实标签损失（$\mathcal{L}_{GT}$）：  
  $$\mathcal{L} = \lambda \mathcal{L}_{KD} + (1-\lambda)\mathcal{L}_{GT}$$

- **轻量化部署支持**  
  兼容多种Transformer架构，支持云端至边缘设备的无缝迁移。

## 🛠 未来工作

- 动态权重分配：根据任务表现自适应调整教师模型贡献。
- 嵌套实体检测：引入指针网络优化边界模糊问题。
- 跨语言/领域迁移：验证模型在通用生物医学NLP任务中的泛化能力。
