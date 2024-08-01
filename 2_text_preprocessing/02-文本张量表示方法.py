# -*- coding: utf-8 -*-
# @Author  : Chinesejun
# @Email   : itcast@163.com
# @File    : 02-文本张量表示方法.py
# @Software: PyCharm

# todo 1： onehot编码实现
# 导入用于对象保存与加载的joblib
from sklearn.externals import joblib
# 导入keras中的词汇映射器Tokenizer
from keras.preprocessing.text import Tokenizer
# 假定vocab为语料集所有不同词汇集合
# vocab = {"周杰伦", "陈奕迅", "王力宏", "李宗盛", "吴亦凡", "鹿晗"}
vocab = ["周杰伦", "陈奕迅", "王力宏", "李宗盛", "吴亦凡", "鹿晗"]
# 实例化一个词汇映射器对象
t = Tokenizer(num_words=None, char_level=False)
# 使用映射器拟合现有文本数据
t.fit_on_texts(vocab)

for token in vocab:
    zero_list = [0]*len(vocab)
    # 使用映射器转化现有文本数据, 每个词汇对应从1开始的自然数
    # 返回样式如: [[2]], 取出其中的数字需要使用[0][0]
    print('token===', t.texts_to_sequences([token]))
    token_index = t.texts_to_sequences([token])[0][0] - 1
    zero_list[token_index] = 1
    print(token, "的one-hot编码为:", zero_list)
    '''
    [[1]]
    鹿晗 的one-hot编码为: [1, 0, 0, 0, 0, 0]
    [[2]]
    周杰伦 的one-hot编码为: [0, 1, 0, 0, 0, 0]
    [[3]]
    李宗盛 的one-hot编码为: [0, 0, 1, 0, 0, 0]
    [[4]]
    陈奕迅 的one-hot编码为: [0, 0, 0, 1, 0, 0]
    [[5]]
    王力宏 的one-hot编码为: [0, 0, 0, 0, 1, 0]
    [[6]]
    吴亦凡 的one-hot编码为: [0, 0, 0, 0, 0, 1]
    '''

# 使用joblib工具保存映射器, 以便之后使用
tokenizer_path = "./Tokenizer"
joblib.dump(t, tokenizer_path)

print("*"*50)
# 调用验证
# 导入用于对象保存与加载的joblib
from sklearn.externals import joblib
# 加载之前保存的Tokenizer, 实例化一个t对象
t = joblib.load(tokenizer_path)

# 编码token为"李宗盛"
# 注意： 只有在词表中映射完成的才会有，  没有映射就会报错
token = "李宗盛"
# 使用t获得token_index
token_index = t.texts_to_sequences([token])[0][0] - 1
# 初始化一个zero_list
zero_list = [0]*len(vocab)
# 令zero_List的对应索引为1
zero_list[token_index] = 1
print(token, "的one-hot编码为:", zero_list)

# # todo 2：使用fasttext工具实现word2vec的训练和使用
# import fasttext
# model = fasttext.train_unsupervised('data/fil9')
# print(model.get_word_vector('the'))
# print('*'*50)
# print(model.get_word_vector('chinese'))
#
# model1 = fasttext.train_unsupervised('data/fil9', "cbow", dim=300, epoch=1, lr=0.1, thread=8)
# print(model1.get_nearest_neighbors('sports'))
# print("*"*50)
# print(model1.get_nearest_neighbors('music'))
# print("*"*50)
# print(model1.get_nearest_neighbors('dog'))
#
# # 模型保存和加载
# model.save_model('fil9.bin')
# model = fasttext.load_model('fil9.bin')
# print(model.get_word_vector('the'))
#
# todo 3: 词嵌入的生成过程和可视化
# 导入torch和tensorboard的摘要写入方法
import torch
import json
import fileinput
from torch.utils.tensorboard import SummaryWriter
# 实例化一个摘要写入对象
writer = SummaryWriter()

# 随机初始化一个100x50的矩阵, 认为它是我们已经得到的词嵌入矩阵
# 代表100个词汇, 每个词汇被表示成50维的向量 这里的100需要和100个词对应
embedded = torch.randn(100, 60)

# 导入事先准备好的100个中文词汇文件, 形成meta列表原始词汇
# print('====', len(list(fileinput.FileInput("./vocab100.csv"))))
meta = list(map(lambda x: x.strip(), fileinput.FileInput("./vocab100.csv")))
writer.add_embedding(embedded, metadata=meta)
writer.close()
print('close.....')





