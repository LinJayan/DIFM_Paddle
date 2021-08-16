## **基于Paddle的DIFM模型复现**
### **论文名称**<a href="https://www.ijcai.org/Proceedings/2020/0434.pdf">A Dual Input-aware Factorization Machine for CTR Prediction</a>

### **一、简介**
### 因子分解机(FMs)是指一类使用实值特征向量的一般预测器，它们以能够在显著稀疏性下估计模型参数而闻名，并在点击率(CTR)等预测领域找到了成功的应用。然而，标准的FMs只为不同输入实例中的每个特征产生一个单一的固定表示，这可能会限制CTR模型的表达和预测能力。受输入感知因子分解机(IFMs)成功的启发，旨在根据不同的输入实例学习更灵活和信息丰富的表示，我们提出了一种名为双输入软件因子分解机(DIFMs)的新模型，它可以同时自适应地在位和向量水平重新加权原始特征表示。此外，DIFMs战略性地将包括多头自检、剩余网络和dnn在内的各种组件集成到一个统一的端到端模型中。
### 模型结构如下图所示：
![](https://ai-studio-static-online.cdn.bcebos.com/ccf9a4714a3e47f4b40faf6f3a7ea2e05f1abe7109164c10ab34c18334689455)
### 在IFM模型的基础上增加了多头注意力用来对特征的向量级做交互。

### **二、复现精度**
### 原论文中DIFM的实验结果如下（PS:论文中的数据量相对比PaddleRec Criteo 数据集量会大点，在实现过程中采用与原论文一样的参数会造成过拟合严重！！！另一方面是DIFM模型在IFM模型基础上改进，增加多头注意力机制，模型过于复杂化了。。。）
![](https://ai-studio-static-online.cdn.bcebos.com/310f3e4992da4c38a79e301f7b747b63b825d02054eb4e5db9a74cc6d3da35e2)
### 本次复现要求AUC>0.799
### (本来很早写完模型跑实验，结果一直过拟合解决，精度一直在0.798左右徘徊，后来被大佬一晚上干出来了，感觉自己菜菜的了。不过通过复现最重要的是学习了（虚伪），虽然提交拿不到冠军，至少花时间学习收获了不少，记录一下)

### **通过减少点参数，今天终于能达到原论文的精度,最终复现结果如下：**
### **AUC=0.7996 (PaddleRec官方Criteo全量测试集)**

### **三、数据集**
训练及测试数据集选用[PaddleRec](https://github.com/PaddlePaddle/PaddleRec/blob/release/2.1.0/datasets/criteo/run.sh)提供的Criteo数据集。

train set: 4400, 0000 条

test set: 184, 0617 条

该数据集包括两部分：训练集和测试集。训练集包含一段时间内Criteo的部分流量，测试集则对应训练数据后一天的广告点击流量。
每一行数据格式如下所示：
```
<label> <integer feature 1> ... <integer feature 13> <categorical feature 1> ... <categorical feature 26>
```
其中```<label>```表示广告是否被点击，点击用1表示，未点击用0表示。```<integer feature>```代表数值特征（连续特征），共有13个连续特征。```<categorical feature>```代表分类特征（离散特征），共有26个离散特征。相邻两个特征用```\t```分隔，缺失特征用空格表示。测试集中```<label>```特征已被移除。  
在模型目录的data目录下为您准备了快速运行的示例数据，若需要使用全量数据可以参考下方[快速开始](#快速开始)部分。

### **四、环境依赖**
CPU、GPU均可，相应设置。

PaddlePaddle >= 2.1.2

Python >= 3.7

### **五、快速开始**
### 大周末的GPU资源耗尽，使用CPU训练2个epoch大约需要11个小时。。。
#### ============================== Step 1,git clone 代码库 ==============================
#### git clone https://github.com/LinJayan/DIFM_Paddle.git

#### ============================== Step 2 download data ==============================
### Download  data
#### cd workpath/DIFM_Paddle/data && wget https://paddlerec.bj.bcebos.com/datasets/criteo/slot_test_data_full.tar.gz
#### tar xzvf slot_test_data_full.tar.gz
    
#### cd workpath/DIFM_Paddle/data && wget https://paddlerec.bj.bcebos.com/datasets/criteo/slot_train_data_full.tar.gz
#### tar xzvf slot_train_data_full.tar.gz

#### ============================== Step 2, train model ==============================
### 启动训练、测试脚本 (需注意当前是否是 GPU 环境）
### !cd workpath/DIFM_Paddle && sh run.sh config_bigdata.yaml

#### 训练日志记录
![](https://ai-studio-static-online.cdn.bcebos.com/a568bd21ca624ff4ba6049bf4ec03cd6520eefab20d64bdc921cc3c31fb28937)

### criteo slot_test_data_full 测试集结果
#### **AUC**=0.7996
![](https://ai-studio-static-online.cdn.bcebos.com/1f54fd58299746dfb5066bec13c6c5004b5325bfa85b4aea9bb847fe8f9aabf7)

### **六、代码结构与详细说明**
```
├─models
   ├─ rank
        ├─difm # DIFM模型代码
        ├─ data #样例数据
        ├── __init__.py
        ├── config.yaml # sample数据配置
        ├── config_bigdata.yaml # 全量数据配置
        ├── net.py # 模型核心组网（动静统一）
        ├── criteo_reader.py #数据读取程序
        ├── dygraph_model.py # 构建动态图
├─tools
├─README.md #文档
├─LICENSE #项目LICENSE
├─run.sh
```

### **七、模型信息**




**原论文重要参数和本项目复现参数对比**

|模型 | num_head | DNN_Layes |Sparse_dim |QKV_dim |
| -------- | -------- | -------- | -------- | -------- |
| DIFM-original | 16 | [256,256,256] |20 |80 |
| DIFM-paddle | 16 | [256,256,39] |20 |80 |
