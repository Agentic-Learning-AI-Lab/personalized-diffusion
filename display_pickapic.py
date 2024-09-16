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

from collections import defaultdict
import datasets
from accelerate.logging import get_logger
from datasets import load_dataset
from diffusers.utils import check_min_version
from matplotlib import pyplot as plt
from collections import Counter
import math
from tqdm import tqdm
import numpy as np

import torch


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.20.0")
logger = get_logger(__name__, log_level="INFO")


class HashableDict(dict):
    def __hash__(self):
        return hash(tuple(sorted(self.items())))


def get_stuff(combined_dataset):
    combined_dataloader = torch.utils.data.DataLoader(
        combined_dataset,
        shuffle=False,
        batch_size=256,
        num_workers=16,
        drop_last=False
    )

    # collect some stuff to check
    index_levels, ranking_ids, num_example_per_prompts, user_ids = [], [], [], []
    captions, image_0_uids, image_1_uids = [], [], []
    for data in tqdm(combined_dataloader):
        index_levels.append(data['__index_level_0__'])
        ranking_ids.append(data['ranking_id'])
        num_example_per_prompts.append(data['num_example_per_prompt'])
        user_ids.append(data['user_id'])
        for x in data['caption']: captions.append(x)
        for x in data['image_0_uid']: image_0_uids.append(x)
        for x in data['image_1_uid']: image_1_uids.append(x)
    index_levels = torch.concat(index_levels).tolist()
    ranking_ids = torch.concat(ranking_ids).tolist()
    num_example_per_prompts = torch.concat(num_example_per_prompts).tolist()
    user_ids = torch.concat(user_ids).tolist()

    return index_levels, ranking_ids, num_example_per_prompts, user_ids, captions, image_0_uids, image_1_uids


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

    # get stuff
    combined_dataset = datasets.concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])
    index_levels, ranking_ids, num_example_per_prompts, user_ids, captions, image_0_uids, image_1_uids = get_stuff(combined_dataset)

    # check index level is just range
    index_levels = sorted(index_levels)
    assert index_levels == list(range(len(index_levels)))

    # check ranking_id is unique
    ranking_ids = sorted(ranking_ids)
    assert len(ranking_ids) == len(set(ranking_ids))
    # assert ranking_ids == list(range(len(ranking_ids))) # not true

    # check if image ids are unique (they are not)
    # assert len(image_0_uids) == len(set(image_0_uids)) # 417214
    # assert len(image_1_uids) == len(set(image_1_uids)) # 704142
    # image_pair_uids = [(x, y) for x, y in zip(image_0_uids, image_1_uids)]
    # assert len(image_pair_uids) == len(set(image_pair_uids)) # 991317

    # check num_example_per_prompt is caption count
    caption_to_count = {}
    for caption, num_example_per_prompt in zip(captions, num_example_per_prompts):
        if caption in caption_to_count:
            assert caption_to_count[caption] == num_example_per_prompt
        else:
            caption_to_count[caption] = num_example_per_prompt
    caption_to_count_2 = Counter(captions)
    assert caption_to_count == caption_to_count_2

    # eliminate no-decisions (0.5-0.5 labels)
    orig_len = combined_dataset.num_rows
    not_split_idx = [i for i, label_0 in enumerate(combined_dataset['label_0']) if label_0 in (0, 1)]
    filtered_dataset = combined_dataset.select(not_split_idx)
    new_len = filtered_dataset.num_rows
    print(f"Eliminated {orig_len - new_len}/{orig_len} split decisions for Pick-a-pic subset")

    # get stuff
    _, _, _, user_ids, captions, _, _ = get_stuff(filtered_dataset)
    print(len(set(user_ids)), 'unique users')
    print(len(set(captions)), 'unique captions')
    # 5871 unique users
    # 54840 unique captions

    # plot histogram of user id
    user_id_counts = Counter(user_ids)
    counts = list(user_id_counts.values())
    counts = [math.log(x, 10) for x in counts]
    plt.figure()
    plt.title('user id frequency')
    plt.xlabel('log10(#labels)')
    plt.ylabel('#users')
    plt.hist(counts, bins=20)
    plt.savefig('pickapic_eda/user_id_distribution.jpg')
    plt.close()

    # plot histogram of caption
    caption_counts = Counter(captions)
    counts = list(caption_counts.values())
    counts = [math.log(x, 10) for x in counts]
    plt.figure()
    plt.title('caption frequency')
    plt.xlabel('log10(#labels)')
    plt.ylabel('#captions')
    plt.hist(counts, bins=20)
    plt.savefig('pickapic_eda/caption_distribution.jpg')
    plt.close()

    # plot per user prompt variety, are users just using the same prompts?
    user_id_to_caption = defaultdict(list)
    for user_id, caption in zip(user_ids, captions):
        user_id_to_caption[user_id].append(caption)
    unique_caption_rate_per_user = [len(set(v)) / len(v) for _, v in user_id_to_caption.items()]
    plt.figure()
    plt.title('unique caption rate per user')
    plt.xlabel('unique caption rate')
    plt.hist(unique_caption_rate_per_user, bins=20)
    plt.savefig('pickapic_eda/unique_caption_rate_per_user_distribution.jpg')
    plt.close()

    # are the high contributers repeating more prompts? (yes)
    user_id_counts = Counter(user_ids)
    user_id_set = list(set(user_ids))
    user_id_set.sort(key=lambda x: user_id_counts[x])
    user_counts_sorted = [user_id_counts[x] for x in user_id_set]
    unique_caption_rate_per_user_sorted = [len(set(user_id_to_caption[x])) / len(user_id_to_caption[x]) for x in user_id_set]
    plt.figure()
    plt.scatter(user_counts_sorted, unique_caption_rate_per_user_sorted)
    plt.xlabel('# contribution')
    plt.ylabel('unique caption rate')
    plt.savefig('pickapic_eda/correlate_contribution_with_unique_caption_rate.jpg')
    plt.close()

    # figure out the proportion of top contributors such that left over ones are still significant
    # plot number of sample vs. number of contributors (sorted by contribution hi to lo)
    user_id_set.sort(key=lambda x: -user_id_counts[x])
    sample_counts = np.cumsum([user_id_counts[x] for x in user_id_set])
    plt.figure()
    plt.scatter(range(len(user_id_set)), sample_counts)
    plt.xlabel('user count (hi to lo contribution)')
    plt.ylabel('sample count')
    plt.savefig('pickapic_eda/overall_contribution_user_count.jpg')
    plt.close()

    # set a user threshold, put all users with <200 labels into validation subset
    for thr in [5, 10, 50, 100, 150, 200]: # (0.002, 0.008, 0.067, 0.123, 0.166, 0.199)
        n_user = len([user_id_counts[x] for x in user_id_set if user_id_counts[x] < thr])
        counts = sum(user_id_counts[x] for x in user_id_set if user_id_counts[x] < thr) / len(user_ids)
        print(f'{counts} total labels for {n_user} users with less than {thr} labels')
    # 0.0023341098686795784 total labels for 950 users with less than 5 labels
    # 0.008102390069955738 total labels for 1732 users with less than 10 labels
    # 0.06699897425273016 total labels for 3910 users with less than 50 labels
    # 0.12336553196424409 total labels for 4614 users with less than 100 labels
    # 0.16624874877974358 total labels for 4926 users with less than 150 labels
    # 0.19893867247360475 total labels for 5094 users with less than 200 labels


if __name__ == "__main__":
    main()
