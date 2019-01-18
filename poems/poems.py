# -*- coding: utf-8 -*-
# file: poems.py
# author: JinTian
# time: 08/03/2017 7:39 PM
# Copyright 2017 JinTian. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
import collections
import os
import sys
import numpy as np

start_token = 'B'
end_token = 'E'


def process_poems(file_name):
    # poems -> list of numbers
    poems = []
    with open(file_name, "r", encoding='utf-8', ) as f:
        for line in f.readlines():
            try:
                title, content = line.strip().split(':')
                content = content.replace(' ', '') # 去掉空格
                if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content or \
                        start_token in content or end_token in content:
                    continue
                if len(content) < 5 or len(content) > 79:
                    continue
                content = start_token + content + end_token
                poems.append(content)
            except ValueError as e:
                pass
    # poems = sorted(poems, key=len)

    all_words = [word for poem in poems for word in poem]
    counter = collections.Counter(all_words) # 将诗句中出现的字符按照出现的频次进行排序
    count_pairs = sorted(counter.items(), key=lambda x: x[1], reverse=True) # reverse=True表示的是对x[1]使用降序的方法进行的排序
    words, _ = zip(*count_pairs)
    # zip()与zip(*)之间的关系：zip是将矩阵打包为元祖的形式，而zip(*)与zip相反，可理解为解压，将元组数据进行解压为原始的格式
    words = words + (' ',) # 加入一个空格
    word_int_map = dict(zip(words, range(len(words))))
    poems_vector = [list(map(lambda word: word_int_map.get(word, len(words)), poem)) for poem in poems]
    # 上一句代码操作的是poem，从poems(诗集)中提取出poem(一首诗)，word表示的是每首诗中的每个字,对每个字进行
    # 梳理，获取每个字的频次，将诗句转换为数字进行存储。相当于每首诗的存储从汉字变为数字。
    #
    # for poem in poems:
    #   aa = list(map(lambda wo: word_int_map.get(wo, len(words)), poem))
    # return aa
    return poems_vector, word_int_map, words


def generate_batch(batch_size, poems_vec, word_to_int):
    n_chunk = len(poems_vec) // batch_size # //表示的是取整除，总共有60178首诗
    x_batches = []
    y_batches = []
    for i in range(n_chunk):
        start_index = i * batch_size
        end_index = start_index + batch_size

        batches = poems_vec[start_index:end_index]
        length = max(map(len, batches)) # 将batches中的每个变量求解len()
        # np.full(参数1：shape-数组的大小，参数2：数组填充的常数值，参数3：数值类型)
        x_data = np.full((batch_size, length), word_to_int[' '], np.int32)
        for row, batch in enumerate(batches):# enumerate表示的是将一个可遍历的数据对象组合为一个索引序列，同时列出下标
            x_data[row, :len(batch)] = batch
        y_data = np.copy(x_data) # 创建一个给定的array副本
        y_data[:, :-1] = x_data[:, 1:] # 大小为64 * 50
        """
        x_data             y_data
        [6,2,4,6,9]       [2,4,6,9,9]
        [1,4,2,8,5]       [4,2,8,5,5]
        """
        x_batches.append(x_data)
        y_batches.append(y_data)
    return x_batches, y_batches
