#! -*- coding:utf-8 -*-
import numpy as np
import random
import copy
import os
import pickle
import torch
import random
from data_utils import DataGenerator
from util import *
from transformers import pipeline
import json

fill = pipeline('fill-mask', model='pretrain_models/bert_base_chinese', tokenizer='pretrain_models/bert_base_chinese')

rel2id = {
    "部件故障": 0,
    "性能故障": 1,
    "检测工具": 2,
    "组成": 3
}
id2rel = {
    0: "部件故障",
    1: "性能故障",
    2: "检测工具",
    3: "组成"
}

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_all_data(orig_data, over_sample=[1, 10, 20, 10], negative_sample=0, augment_ratio=0, augment_method=None):
    orig_data = np.array(orig_data)
    all_data = list(orig_data)
    for idx, sample in enumerate(orig_data):
        tot = 0
        cnt = 0
        h_list = []
        t_list = []
        num_spos = len(sample['spos'])
        for spo in sample['spos']:
            h, rel, t = spo
            tot += over_sample[rel2id[rel]]
            cnt += 1
            h_list.append(h)
            t_list.append(t)
        for i in range(num_spos*negative_sample):
            sampled_h_idx = random.randint(0, num_spos-1)
            sampled_t_idx = random.randint(0, num_spos-1)
            if sampled_h_idx == sampled_t_idx:
                if sampled_t_idx == num_spos - 1:
                    sampled_t_idx = 0
                else:
                    sampled_t_idx += 1
            rel_id = np.random.choice([0, 1, 2, 3], p=[0.75, 0.1, 0.1, 0.05])
            neg_spo = [h_list[sampled_h_idx], rel_id, t_list[sampled_t_idx]]
            orig_data[idx]['spos'].append(neg_spo)
        if cnt > 0:
            ratio = (int) (tot/cnt)
        else:
            ratio = 0
        for i in range(ratio):
            all_data.append(orig_data[idx])
        for iter in range(augment_ratio):
            if augment_method is 'swap':
                text = sample['text']
                text_len = len(text)
                num_swap = int(text_len / 20) + 1
                new_sample = sample
                for _ in range(num_swap):
                    idx1 = random.randint(0, text_len-1)
                    idx2 = random.randint(0, text_len-1)
                    for spo in sample['spos']:
                        h, rel, t = spo
                        if idx1 in range(h[0], h[1]):
                            try:
                                idx1 = h[1]
                            except:
                                idx1 = h[0] - 1
                        if idx1 in range(t[0], t[1]):
                            try:
                                idx1 = t[0] - 1
                            except:
                                idx1 = t[1]
                        if idx2 in range(h[0], h[1]):
                            try:
                                idx2 = h[1]
                            except:
                                idx2 = h[0] - 1
                        if idx2 in range(t[0], t[1]):
                            try:
                                idx2 = t[0] - 1
                            except:
                                idx2 = t[1]
                    new_text = ''
                    for idx, char in enumerate(text):
                        if idx == idx1:
                            new_text += text[idx2]
                        elif idx == idx2:
                            new_text += text[idx1]
                        else:
                            new_text += text[idx]
                    new_sample['text'] = new_text
                    text = new_text
                all_data.append(new_sample)
            elif augment_method is 'replace':
                text = sample['text']
                text_len = len(text)
                num_replace = int(text_len / 20) + 1
                new_sample = sample
                for _ in range(num_replace):
                    text_len = len(text)
                    try:
                        idx = random.randint(0, text_len-1)
                    except:
                        continue
                    new_text = ''
                    for cur_idx, char in enumerate(text):
                        if cur_idx == idx:
                            new_text += '[MASK]'
                        else:
                            new_text += text[cur_idx]
                    print(new_text)
                    result = fill(new_text)
                    print(result)
                    res_idx = random.randint(0, 3)
                    new_text = result[res_idx]['sequence'].replace(" ", "")
                    ins_pos = random.randint(0, 2)
                    if ins_pos == 0:
                        idx = max(idx-1, 0)
                    new_text = text[:idx+1] + '[MASK]' + text[idx+1:]
                    result = fill(new_text)
                    print(new_text)
                    print(result)
                    new_text = result[0]['sequence'].replace(" ", "")
                    new_sample['text'] = new_text
                    text = new_text
                    new_spos = []
                    for spo in new_sample['spos']:
                        h, rel, t = spo
                        if h[0] > idx:
                            h[0] += 1
                        if h[1] > idx:
                            h[1] += 1
                        if t[0] > idx:
                            t[0] += 1
                        if t[1] > idx:
                            t[1] += 1
                        new_spo = [h, rel, t]
                        new_spos.append(new_spo)
                    new_sample['spos'] = new_spos
                    print(new_sample)
                all_data.append(new_sample)
                print(len(all_data))
            
    fout = open('data/bdci/train_aug.json', 'w', encoding='utf8')
    fout.writelines(json.dumps(all_data, ensure_ascii=False, indent=2))


class data_generator_sampling(DataGenerator):
    def __init__(self, args, train_data, tokenizer, predicate_map, label_map, batch_size, random=False, is_train=True):
        super(data_generator_sampling, self).__init__(train_data, batch_size)
        self.max_len = args.max_len
        self.tokenizer = tokenizer
        self.predicate2id, self.id2predicate = predicate_map
        self.label2id, self.id2label = label_map
        self.random = random
        self.is_train = is_train

    def __iter__(self):
        batch_token_ids, batch_mask = [], []
        batch_label, batch_neg_label = [], []
        batch_mask_label = []
        batch_ex = []
        for is_end, d in self.sample(self.random):
            if self.is_train:
                if judge(d) == False:
                    continue
            token_ids, mask = self.tokenizer.encode(
                d['text'], maxlen=self.max_len
            )
            if self.is_train:
                entities = []
                for spo in d['spos']:
                    entities.append(tuple(spo[0]))
                    entities.append(tuple(spo[2]))
                entities = sorted(list(set(entities)))
                one_info = get_token_idx(d['text'], entities, self.tokenizer)
                spoes = {}
                for ss, pp, oo in d['spos']:
                    s_key = (ss[0], ss[1])
                    if pp in [0, 1, 2, 3]:
                        p = pp
                        NA = True
                    else:
                        p = rel2id[pp]
                        NA = False
                    o_key = (oo[0], oo[1])
                    s = tuple(one_info[s_key])
                    o = copy.deepcopy(one_info[o_key])
                    o.append(p)
                    o.append(NA)
                    o = tuple(o)
                    if s not in spoes:
                        spoes[s] = []
                    spoes[s].append(o)

                if spoes:
                    label = np.zeros([len(token_ids), len(token_ids), len(self.id2predicate)])
                    is_neg = label
                    for s in spoes:
                        s1, s2 = s
                        try:
                            for o1, o2, p, is_NA in spoes[s]:
                                
                                if is_NA:
                                    try:
                                        label[s1, o1, p] = self.label2id["N/A"]
                                        is_neg[s1, o1, p] = 1
                                    except:
                                        print(d, spoes)
                                else:
                                    try:
                                        if s1 == s2 and o1 == o2:
                                            label[s1, o1, p] = self.label2id["SS"]
                                        elif s1 != s2 and o1 == o2:
                                            label[s1, o1, p] = self.label2id["MSH"]
                                            label[s2, o1, p] = self.label2id["MST"]
                                        elif s1 == s2 and o1 != o2:
                                            label[s1, o1, p] = self.label2id["SMH"]
                                            label[s1, o2, p] = self.label2id["SMT"]
                                        elif s1 != s2 and o1 != o2:
                                            label[s1, o1, p] = self.label2id["MMH"]
                                            label[s2, o2, p] = self.label2id["MMT"]
                                    except:
                                        print(d, spoes)
                        except Exception as e:
                            print(one_info, d['text'])
                            assert 0

                    mask_label = np.ones(label.shape)
                    mask_label[0, :, :] = 0
                    mask_label[-1, :, :] = 0
                    mask_label[:, 0, :] = 0
                    mask_label[:, -1, :] = 0

                    for a, b in zip([batch_token_ids, batch_mask, batch_label, batch_neg_label, batch_mask_label, batch_ex],
                                    [token_ids, mask, label, is_neg, mask_label, d]):
                        a.append(b)

                    if len(batch_token_ids) == self.batch_size or is_end:
                        batch_token_ids, batch_mask = [sequence_padding(i) for i in [batch_token_ids, batch_mask]]
                        batch_label = mat_padding(batch_label)
                        batch_neg_label = mat_padding(batch_neg_label)
                        batch_mask_label = mat_padding(batch_mask_label)
                        yield [
                            batch_token_ids, batch_mask,
                            batch_label, batch_neg_label,
                            batch_mask_label, batch_ex
                        ]
                        batch_token_ids, batch_mask = [], []
                        batch_label, batch_neg_label = [], []
                        batch_mask_label = []
                        batch_ex = []
            else:
                for a, b in zip([batch_token_ids, batch_mask, batch_ex], [token_ids, mask, d]):
                    a.append(b)
                if len(batch_token_ids) == self.batch_size or is_end:
                    batch_token_ids, batch_mask = [sequence_padding(i) for i in [batch_token_ids, batch_mask]]
                    yield [
                        batch_token_ids, batch_mask, batch_ex
                    ]
                    batch_token_ids, batch_mask = [], []
                    batch_ex = []
