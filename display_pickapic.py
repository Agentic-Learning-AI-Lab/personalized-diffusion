#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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

from accelerate.logging import get_logger
from datasets import load_dataset
from diffusers.utils import check_min_version
from matplotlib import pyplot as plt
from collections import Counter
import math


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.20.0")
logger = get_logger(__name__, log_level="INFO")

def main():
    split = 'train'
    dataset = load_dataset('yuvalkirstain/pickapic_v2', cache_dir='./pick_a_pic_v2/')

    # eliminate no-decisions (0.5-0.5 labels)
    orig_len = dataset[split].num_rows
    not_split_idx = [i for i, label_0 in enumerate(dataset[split]['label_0']) if label_0 in (0, 1)]
    dataset[split] = dataset[split].select(not_split_idx)
    new_len = dataset[split].num_rows
    print(f"Eliminated {orig_len - new_len}/{orig_len} split decisions for Pick-a-pic")

    dataset_split = dataset[split]
    user_id_counts = Counter(dataset_split['user_id'])
    counts = list(user_id_counts.values())
    counts = [math.log(x, 10) for x in counts]
    plt.figure()
    plt.title('user id frequency')
    plt.xlabel('log10(#labels)')
    plt.ylabel('#users')
    plt.hist(counts, bins=20)
    plt.savefig('user_id_distribution.jpg')
    plt.close()



if __name__ == "__main__":
    main()
