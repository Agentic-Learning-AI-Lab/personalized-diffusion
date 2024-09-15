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

import datasets
from accelerate.logging import get_logger
from datasets import load_dataset
from diffusers.utils import check_min_version
from matplotlib import pyplot as plt
from collections import Counter
import math
from tqdm import tqdm


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.20.0")
logger = get_logger(__name__, log_level="INFO")


class HashableDict(dict):
    def __hash__(self):
        return hash(tuple(sorted(self.items())))


def main():
    # remove images to analyze other columns
    dataset = load_dataset('yuvalkirstain/pickapic_v2', cache_dir='./pick_a_pic_v2/')
    print(dataset)
    dataset = dataset.remove_columns(['jpg_0', 'jpg_1', 'created_at'])
    print()

    # number of samples in each split
    for split, subset in dataset.items():
        print(f'{split} has {len(subset)} samples')

    # assert all entries are unique
    # for split, subset in dataset.items():
    #     subset_dicts = [HashableDict(x) for x in subset]
    #     assert len(subset_dicts) == len(set(subset_dicts))

    # check if unique sets are subsets
    val_split_dicts = set([HashableDict(x) for x in dataset['validation']])
    val_unique_split_dicts = set([HashableDict(x) for x in dataset['validation']])
    test_split_dicts = set([HashableDict(x) for x in dataset['test']])
    test_unique_split_dicts = set([HashableDict(x) for x in dataset['test']])
    assert all(x in val_split_dicts for x in val_unique_split_dicts)
    assert all(x in test_split_dicts for x in test_unique_split_dicts)

    # check no overlap between valid and test
    assert all(x not in test_split_dicts for x in val_split_dicts)

    # check if val and test uids are unique, they are NOT
    # val_unique_uids = dataset['validation_unique']['user_id']
    # test_unique_uids = dataset['test_unique']['user_id']
    # assert len(val_unique_uids) == len(set(val_unique_uids))
    # assert len(test_unique_uids) == len(set(test_unique_uids))

    # check if val and test prompts are unique, they ARE
    val_unique_captions = dataset['validation_unique']['caption']
    test_unique_captions = dataset['test_unique']['caption']
    assert len(val_unique_captions) == len(set(val_unique_captions))
    assert len(test_unique_captions) == len(set(test_unique_captions))

    # check if val and test unique have same captions as val and test
    val_captions = dataset['validation']['caption']
    test_captions = dataset['test']['caption']
    val_unique_captions = dataset['validation_unique']['caption']
    test_unique_captions = dataset['test_unique']['caption']
    assert len(set(val_captions)) == len(val_unique_captions)
    assert len(set(test_captions)) == len(test_unique_captions)

    # remove validation_unique and test_unique
    del dataset['validation_unique'], dataset['test_unique']

    # merge train, validation, test
    combined_dataset = datasets.concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])

    # collect some stuff to check
    index_levels, ranking_ids, captions, num_example_per_prompts = [], [], [], []
    image_0_uids, image_1_uids = [], []
    for data in tqdm(combined_dataset, total=len(combined_dataset)):
        index_levels.append(data['__index_level_0__'])
        ranking_ids.append(data['ranking_id'])
        captions.append(data['caption'])
        num_example_per_prompts.append(data['num_example_per_prompt'])
        image_0_uids.append(data['image_0_uid'])
        image_1_uids.append(data['image_1_uid'])

    # check index level is just range
    index_level = combined_dataset['__index_level_0__']
    index_level = sorted(index_level)
    assert len(index_level) == len(set(index_level))
    breakpoint()
    assert index_level == list(range(len(index_level)))

    # check ranking_id is unique
    ranking_id = combined_dataset['ranking_id']
    ranking_id = sorted(ranking_id)
    assert len(ranking_id) == len(set(ranking_id))
    # assert ranking_id == list(range(len(ranking_id))) # not true

    # check if image ids are unique
    breakpoint()

    # check num_example_per_prompt is caption count
    caption = combined_dataset['caption']
    num_example_per_prompt = combined_dataset['num_example_per_prompt']





    # eliminate no-decisions (0.5-0.5 labels)
    orig_len = combined_dataset.num_rows
    not_split_idx = [i for i, label_0 in enumerate(combined_dataset['label_0']) if label_0 in (0, 1)]
    combined_dataset = combined_dataset.select(not_split_idx)
    new_len = combined_dataset.num_rows
    print(f"Eliminated {orig_len - new_len}/{orig_len} split decisions for Pick-a-pic subset")




    # plot histogram of user id
    breakpoint()
    user_id_counts = Counter(combined_dataset['user_id'])
    counts = list(user_id_counts.values())
    counts = [math.log(x, 10) for x in counts]
    plt.figure()
    plt.title('user id frequency')
    plt.xlabel('log10(#labels)')
    plt.ylabel('#users')
    plt.hist(counts, bins=20)
    plt.savefig('user_id_distribution.jpg')
    plt.close()

    # plot histogram of caption
    caption_counts = Counter(combined_dataset['caption'])
    counts = list(caption_counts.values())
    counts = [math.log(x, 10) for x in counts]
    plt.figure()
    plt.title('user id frequency')
    plt.xlabel('log10(#labels)')
    plt.ylabel('#captions')
    plt.hist(counts, bins=20)
    plt.savefig('caption_distribution.jpg')
    plt.close()




if __name__ == "__main__":
    main()
