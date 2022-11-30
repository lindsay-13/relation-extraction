# 工业制造故障案例文本关系抽取



### 环境

bert4keras 0.11.3

numpy 1.23.3

scikit_learn 0.24.0

torch 1.12.1+cu116

transformers 4.21.0



### 预训练模型下载

使用huggingface预训练模型[luhua/chinese_pretrain_mrc_roberta_wwm_ext_large](https://huggingface.co/luhua/chinese_pretrain_mrc_roberta_wwm_ext_large/tree/main). 训练及测试前需将pytorch_model.bin, config.json, vocab.txt下载并保存在pretrain_models/roberta_wwm_large.



### 数据集

**实体及关系类型：**

| 关系     | 主体     | 客体     |
| -------- | -------- | -------- |
| 部件故障 | 部件单元 | 故障状态 |
| 性能故障 | 性能表征 | 故障状态 |
| 检测工具 | 检测工具 | 性能表征 |
| 组成     | 部件单元 | 部件单元 |

**数据格式：**

训练数据（data/bdci/train_bdci.json）：文件每行为一个样本。

- ID：样本编号
- text：输入文本
- spo_list：列表每项为一个关系。
  - h：主体
  - t：客体
  - relation：关系类别

训练数据样例：

```json

{
	"ID": "AT0001",
	"text":"故障现象:车速到100迈以上发动机盖后部随着车速抖动。故障原因简要分析:经技术人员试车；怀疑发动机盖锁或发动机盖铰链松旷。",
	"spo_list":[
		{"h": {"name": "发动机盖", "pos": [14, 18]},
		"t": {"name": "抖动", "pos": [24, 26]},
		"relation": "部件故障"},
		{"h": {"name": "发动机盖锁", "pos": [46, 51]},
		"t": {"name": "松旷", "pos": [58, 60]},
		"relation": "部件故障"},
		{"h": {"name": "发动机盖铰链", "pos": [52, 58]},
		"t": {"name": "松旷", "pos": [58, 60]},
		"relation":"部件故障"}
	]
}
```

测试数据（data/bdci/evalA.json）：文件每行为一个样本

- ID：样本编号
- text：输入文本

测试数据样例：

```json
{
	"ID": "AE0002", 
	"text": "套管渗油、油位异常现象：套管表面渗漏有油渍。套管油位异常下降或者升高。处理原则：套管严重渗漏或者外绝缘破裂，需要更换时，向值班调控人员申请停运处理。套管油位异常时，应利用红外测温装置等方法检测油位，确认套管发生内漏需要处理时，向值班调控人员申请停运处理。"
}
```

输出数据格式：与训练数据相同



### 数据预处理

```
python data_generator.py
```

在data/bdci中生成训练数据文件train.json及测试数据文件test.json.



### 模型训练

运行train.py:

```
python train.py
```

参数说明：

--over_sampling_ratio：过采样比例，输入4元list，分别对应部件故障、性能故障、检测工具、组成四类关系的过采样比例。默认[0, 0, 0, 0]，不执行过采样。

--negative_sampling_ratio：负采样比例，即对于训练集中的每个正例样本采样多少个不存在关系的样本作为负例。默认0，不执行负采样。

--aug_ratio：数据增强比例，即通过数据增强获取的新样例数为原数据集样例数aug_ratio倍。默认0，不执行数据增强。

--aug_type：数据增强方式。选项：swap，replace。其他输入均为不执行数据增强。

- swap：随机选择文本中两个字并交换，重复执行次数为文本长度/20. 不会改动关系主体及客体的文字和位置。([EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks](https://arxiv.org/abs/1901.11196))
- replace：随机选择文本中的一个字，将其用[MASK]代替，使用bert_base_chinese预训练模型预测[MASK]处的字，在概率前3的预测中随机选择一个。再随机在这一字符的前或后插入[MASK]并用bert_base_chinese预测，选择概率最高的字插入文本。重复执行次数为文本长度/20. ([BAE: BERT-based Adversarial Examples for Text Classification](https://arxiv.org/abs/2004.01970v1))

--adversarial_method：对抗训练方式。选项：fgm，pgd。其他输入均为不执行对抗训练。

- fgm: Fast Gradient Method. 对embedding层$x$做如下扰动，对抗样本为$x+r$。([Adversarial Training Methods for Semi-Supervised Text Classification](https://arxiv.org/abs/1605.07725))

$$
r=\epsilon\cdot\frac{g}{||g||_2} \\
g = \nabla_xL(\theta,x,y)
$$

- pgd: Projected Gradient Descent方法。对embedding层$x$做如下扰动，对抗样本为$x+r_k$。([Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083))

$$
r_{t+1}=\alpha \cdot \frac{g_t}{||g_t||_2} \\
||r||_2\le\epsilon
$$

--alpha, --epsilon, --pgd_k：对抗训练中的超参数。

--k_num：k折交叉验证中k的取值。



### 模型预测

运行predict.py:

```
python predict.py
```

参数说明：

--k_num: k折交叉验证中k的取值，需要和训练时k_num参数取值相同。

--thresh: 合并k_num个模型的结果时的阈值。若相同的关系出现在至少thresh个模型的预测结果中，将其加入最终结果。



### 模型结果分析

运行result_analysis.ipynb：

- 错误结果分析：将错误样例分类保存在result的四个文件（inc_spans.json, inc_relation.json, inc_s.json, inc_o.json）并分类输出错误样例数。类别含义：
  - incorrect spans：主体及客体识别正确，具体位置预测错误。
  - incorrect relation：主体及客体识别正确，关系类别预测错误。
  - incorrect subject：关系类别及客体正确，主体预测错误。
  - incorrect object：关系类别及主体正确，客体预测错误。
- 按文本长度分类计算预测效果
- 按关系类别分类计算预测效果
- 统计真实结果和预测结果中各类关系数目