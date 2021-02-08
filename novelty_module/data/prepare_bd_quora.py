# coding=utf-8
# Copyright (C) 2019 Alibaba Group Holding Limited
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


import os
from tqdm import tqdm


print('processing BD quora')
os.makedirs('bd_quora', exist_ok=True)
# use the partition on https://zhiguowang.github.io
for split in ('train_quora_bd', 'test_quora_bd'):
    with open('orig/BD_Quora/{}.tsv'.format(split)) as f, \
            open('bd_quora/{}.txt'.format(split), 'w') as fout:
        n_lines = 0
        for _ in f:
            n_lines += 1
        f.seek(0)
        for line in tqdm(f, total=n_lines, leave=False):
            elements = line.rstrip().split('\t')
            #print(elements, len(elements))
            # TO skip empty lines
            if len(elements) == 1:
            	continue
            fout.write('{}\t{}\t{}\n'.format(elements[0], elements[1], int(elements[2])))
