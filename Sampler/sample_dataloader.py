# -- coding: utf-8 --
import collections
import torch
import numpy as np
import json
import random
from transformers import BertTokenizer
from torch.utils.data import Dataset
import itertools


class sample_dataloader(object):

    def __init__(self, quadruple, memory, id2sent, config=None, seed=None, FUN_CODE=3, rel_rep=None, task_sample=True,
                 use_mem_data=False, picture=False):
        # When FUN_CODE is 0, quadruple is sentence of current task.
        self.quadruple = quadruple
        self.id2sent = id2sent
        self.memory = memory
        self.config = config
        self.FUN_CODE = FUN_CODE
        self.use_mem_data = use_mem_data
        self.picture=picture
        if FUN_CODE == 0:
            self.use_task_sample = task_sample
            self.data_idx = self.get_mem_enhanced_data(self.config.batch_size,
                                                       use_task_sample=self.use_task_sample)  # actually is data.
        if FUN_CODE == 1:
            self._ini_before_get_data()
            self.data_idx = self.get_data(self.config.batch_size)
        if FUN_CODE == 2:
            self.data_idx = self.get_contrastive_data(self.config.batch_size)
        if FUN_CODE == 3:
            self.rel_rep = rel_rep
            self.use_task_sample = task_sample
            self.data_idx = self.get_fix_proto_data(self.config.batch_size,
                                                    use_task_sample=self.use_task_sample)  # actually is data.
        self._ix = 0

        self.seed = seed
        if self.seed != None:
            self.set_seed(self.seed)

    def __len__(self):
        return len(self.data_idx)

    def __iter__(self):
        return self

    def __next__(self):
        if self.FUN_CODE == 2:
            try:
                l_idx, is_p_data = self.data_idx[self._ix]
            except IndexError:
                self.data_idx = self.get_contrastive_data(self.config.batch_size)
                self._ix = 0
                raise StopIteration

            batch_all = [self.quadruple[i] for i in l_idx]
            sent_inp, emb_inp, preds, trues = [], [], [], []
            for (sent_id, emb, pred, true) in batch_all:
                sent_inp.append(torch.tensor(self.id2sent[sent_id]).int())
                emb_inp.append(emb)
                preds.append(pred)
                trues.append(true)
            labels = np.zeros(shape=(len(trues), len(trues)))
            if is_p_data:
                comparison = torch.ones((len(trues), len(trues)))
                for i in range(1, len(trues)):
                    for j in range(0, i):
                        labels[i][j] = (trues[i] == trues[j])
            else:
                comparison = torch.ones((len(trues), len(trues)))
                for i in range(1, len(trues)):
                    for j in range(0, i):
                        labels[i][j] = (preds[i] == trues[i]) * (preds[j] == trues[j]) * (trues[i] == trues[j])
                        # AB+AB^C+A^BC
                        comparison[i][j] = (preds[i] == trues[i]) * (preds[j] == trues[j]) or (
                                (preds[i] == trues[i]) * (preds[j] != trues[j]) + (preds[i] != trues[i]) * (
                                preds[j] == trues[j])) * (trues[i] == trues[j])
            labels += labels.T
            for i in range(len(trues)):
                labels[i][i] = (preds[i] == trues[i])
            labels = torch.IntTensor(labels)

            self.verify_metrix(labels, comparison)
            sent_inp = torch.stack(sent_inp)
            emb_inp = torch.stack(emb_inp)
            self._ix += 1
            return sent_inp, emb_inp, labels, comparison
        elif self.FUN_CODE == 1:
            try:
                batch_sent, batch_emb = self.data_idx[self._ix]
            except IndexError:
                self.data_idx = self.get_data(self.config.batch_size)
                self._ix = 0
                raise StopIteration

            sent_batch = [self.quadruple[i] for i in batch_sent]
            emb_batch = [self.quadruple[i] for i in batch_emb]
            sent_inp, emb_inp, preds, trues = [], [], [], []
            for (sent_id, _, _, true) in sent_batch:
                sent_inp.append(torch.tensor(self.id2sent[sent_id]).int())
                trues.append(true)  # B1
            for (_, emb, _, true) in emb_batch:
                emb_inp.append(emb)
                preds.append(true)  # B2
            trues = torch.tensor(trues)
            preds = torch.tensor(preds)
            comparison = torch.ones((trues.shape[0], preds.shape[0]))
            trues = trues.expand((preds.shape[0], trues.shape[0])).transpose(-1, -2)
            preds = preds.expand((trues.shape[0], preds.shape[0]))
            labels = (trues == preds).int()

            # self.verify_metrix(labels, comparison)
            sent_inp = torch.stack(sent_inp)
            emb_inp = torch.stack(emb_inp)
            self._ix += 1
            return sent_inp, emb_inp, labels, comparison
        elif self.FUN_CODE == 0:
            try:
                batch_change, batch_fix = self.data_idx[self._ix]
            except IndexError:
                self.data_idx = self.get_mem_enhanced_data(self.config.batch_size, use_task_sample=self.use_task_sample)
                self._ix = 0
                raise StopIteration

            trues = torch.tensor([item['relation'] for item in batch_change])
            tokens = [torch.tensor(item['tokens']) for item in batch_change]

            left_inp = torch.stack(tokens, dim=0)
            preds = torch.tensor([item['relation'] for item in batch_fix])
            tokens = [torch.tensor(item['tokens']) for item in batch_fix]
            right_inp = torch.stack(tokens, dim=0)

            comparison = torch.ones((trues.shape[0], preds.shape[0]))
            trues = trues.expand((preds.shape[0], trues.shape[0])).transpose(-1, -2)
            preds = preds.expand((trues.shape[0], preds.shape[0]))
            labels = (trues == preds).int()

            self.verify_metrix(labels, comparison)

            self._ix += 1
            return left_inp, right_inp, labels, comparison
        elif self.FUN_CODE == 3:
            try:
                batch_change, batch_fix = self.data_idx[self._ix]
            except IndexError:
                self.data_idx = self.get_fix_proto_data(self.config.batch_size,
                                                        use_task_sample=self.use_task_sample)
                self._ix = 0
                raise StopIteration

            trues = torch.tensor([item['relation'] for item in batch_change])
            tokens = [torch.tensor(item['tokens']) for item in batch_change]
            left_inp = torch.stack(tokens, dim=0)

            random.shuffle(batch_fix)
            preds = torch.tensor(batch_fix)
            right_inp = torch.cat([self.rel_rep[i] for i in batch_fix], dim=0)

            comparison = torch.ones((trues.shape[0], preds.shape[0]))
            trues = trues.expand((preds.shape[0], trues.shape[0])).transpose(-1, -2)
            preds = preds.expand((trues.shape[0], preds.shape[0]))
            labels = (trues == preds).int()

            self.verify_metrix(labels, comparison)

            self._ix += 1
            if not self.picture:
                return left_inp, right_inp, labels, comparison
            else:
                return left_inp, right_inp, labels, comparison, preds, trues

    def get_fix_proto_data(self, batch_size, use_task_sample=True):
        # kmeans结果
        left, right = [], []
        batch_d = []
        C = len(self.memory)
        right = list(self.rel_rep.keys())

        for i in self.memory.values():
            random.shuffle(i)

        if use_task_sample:
            if self.use_mem_data:
                for l in self.choice_object_no_replace(self.quadruple + list(itertools.chain(*self.memory.values())),
                                                       batch_size, shuffle=True):
                    batch_d.append((l, right))
            else:
                for l in self.choice_object_no_replace(self.quadruple, batch_size, shuffle=True):
                    batch_d.append((l, right))
        else:
            data_pool = list(itertools.chain(*self.memory.values()))
            for l in self.choice_object_no_replace(data_pool, min(batch_size, C), shuffle=True):
                batch_d.append((l, right))
        return batch_d

    def get_mem_enhanced_data(self, batch_size, use_task_sample=True):
        # kmeans结果
        left, right = [], []
        batch_d = []
        C = len(self.memory)

        for i in self.memory.values():
            random.shuffle(i)
        for right_part in zip(*self.memory.values()):
            right.append(right_part)
        if use_task_sample:
            K = len(i)
            sample_ins = random.choices(self.quadruple, k=int(K * batch_size))
            for l, r in zip(self.choice_object_no_replace(sample_ins, batch_size, shuffle=True), right):
                batch_d.append((l, r))
        else:
            data_pool = list(itertools.chain(*self.memory.values()))
            for l, r in zip(self.choice_object_no_replace(data_pool, min(batch_size, C), shuffle=True), right):
                batch_d.append((l, r))
        return batch_d

    def verify_metrix(self, labels, comparison):
        if self.FUN_CODE == 2:
            if comparison[labels == 1].sum() < comparison[labels == 1].shape[0] or labels[comparison == 0].sum() != 0:
                raise Exception('labels and comparison not matched')
        elif self.FUN_CODE in [0, 1]:
            """
            检测labels每行是不是有且仅有一个1
            """
            if (torch.sum(labels, dim=-1) == 1).sum() != labels.shape[0]:
                print(labels)
                print(torch.sum(labels, dim=-1) == 1)
                print((torch.sum(labels, dim=-1) == 1).sum())
                raise Exception('labels and comparison not matched')

    def get_mem_data(self, mem_cluster2quad_p, num_sent):
        for i in mem_cluster2quad_p.values():
            random.shuffle(i)
        a = self._get_mem_data(mem_cluster2quad_p, num_sent)
        while 1:
            try:
                res = next(a)
            except StopIteration:
                for i in mem_cluster2quad_p.values():
                    random.shuffle(i)
                a = self._get_mem_data(mem_cluster2quad_p, num_sent)
                res = next(a)
            yield res

    def _get_mem_data(self, mem_cluster2quad_p, num_sent):
        for quad_id in zip(*mem_cluster2quad_p.values()):
            # yield (quad,quad[:num_sent])#emb and sent poisition in quadruple
            yield quad_id  # emb and sent poisition in quadruple

    def _ini_before_get_data(self):
        # statistic
        # 得到句子序号到quadruple序号的映射，i是quadruple序号
        sent_id2quad_id = collections.defaultdict(list)
        self.cluster2quad_p = collections.defaultdict(list)
        self.sent_cluster2quad_p = collections.defaultdict(list)
        self.mem_cluster2quad_p = collections.defaultdict(list)
        for i, quad in enumerate(self.quadruple):
            sent_id2quad_id[quad[0]].append(i)
            if quad[-1] == quad[-2]:
                self.cluster2quad_p[quad[-1]].append(i)

        for quad_ids in sent_id2quad_id.values():
            self.sent_cluster2quad_p[self.quadruple[quad_ids[0]][-1]].append(
                quad_ids[0])  # 建立句子类别到quadruple id的唯一映射，不会有重复句子
        # memory update
        start_index = len(self.quadruple)
        len_mem = 0

        for mem_cluster, all_items in self.memory.items():
            for _, quad_ins in all_items:
                self.quadruple.append(quad_ins)
                self.mem_cluster2quad_p[mem_cluster].append(start_index + len_mem)
                len_mem += 1

    def get_data(self, batch_size):
        # 为各个簇统计向量位置，选一个簇类，如果有memory，在memory中找一个正向量，此簇和memory此簇中所有句子形成池，池中采样句子
        num_sent = max(1, len(self.mem_cluster2quad_p))
        mem_gen = self.get_mem_data(self.mem_cluster2quad_p, num_sent=num_sent)  # 每次返回各个类的quad_id,结束时候自动shuffle
        task_cluster = list(self.cluster2quad_p.keys())
        random.shuffle(task_cluster)
        batch_data = []
        count = 0
        for cluster in task_cluster:
            # choose this cluster,get all sent and every emb
            cur_clu_emb_all = self.cluster2quad_p[cluster]
            cur_clu_sent_all = self.sent_cluster2quad_p[cluster]
            # if (batch_size - num_sent) > len(cur_clu_sent_all):
            if 2 * num_sent > len(cur_clu_sent_all):
                count += 1
                print(f"Sentence oversampling {count} times,sent id:{cluster},num:{len(cur_clu_sent_all)}.")
            cur_all_sent_batch = self.multi_sample_no_replace(cur_clu_sent_all, 2 * num_sent, shuffle=True)
            len_sent_batch = len(cur_all_sent_batch)
            i = 0

            for control, emb in enumerate(cur_clu_emb_all):
                sent_batch = []
                emb_batch = []
                if control & 1:
                    continue
                # if control>2:
                #     break
                if len(self.mem_cluster2quad_p):
                    mem_emb = next(mem_gen)
                    # 数据加入正句子，加入正向量，采样此簇所有句子
                    sent_batch.extend(mem_emb[:num_sent])  # 采样mem中句子，采样num_sent个
                    if i < len_sent_batch:  # 采样task中句子，并在采完一次后重复
                        sent_batch.extend(cur_all_sent_batch[i])
                    else:
                        cur_all_sent_batch = self.multi_sample_no_replace(cur_clu_sent_all, 2 * num_sent,
                                                                          shuffle=True)
                        len_sent_batch = len(cur_all_sent_batch)
                        i = 0
                        sent_batch.extend(cur_all_sent_batch[i])
                    # memory取正句子和向量，采样此簇所有句子
                    emb_batch.append(emb)
                    emb_batch.extend(mem_emb)
                    for ot_cluster in task_cluster:
                        if ot_cluster != cluster:
                            emb_batch.extend(random.choices(self.cluster2quad_p[ot_cluster], k=4))
                            sent_batch.extend(random.choices(self.sent_cluster2quad_p[ot_cluster], k=1))
                    i += 1
                    # 对于簇间向量进行采样，如果没有看到所有簇就重复采样
                    random.shuffle(sent_batch)
                    random.shuffle(emb_batch)
                    batch_data.append((sent_batch,
                                       emb_batch))  # 句子：num_sent个mem+正类个，向量：其他类*K+一个正类+mem规定采样数 batch_size:(),1+exist_C+(task_C-1)*2
                else:  # mem is empty
                    if i < len_sent_batch:  # 采样task中句子，并在采完一次后重复
                        sent_batch.extend(cur_all_sent_batch[i])
                    else:
                        cur_all_sent_batch = self.multi_sample_no_replace(cur_clu_sent_all, 2 * num_sent,
                                                                          shuffle=True)
                        len_sent_batch = len(cur_all_sent_batch)
                        i = 0
                        sent_batch.extend(cur_all_sent_batch[i])

                    emb_batch.append(emb)

                    for ot_cluster in task_cluster:
                        if ot_cluster != cluster:
                            emb_batch.extend(random.choices(self.cluster2quad_p[ot_cluster], k=4))
                            sent_batch.extend(random.choices(self.sent_cluster2quad_p[ot_cluster], k=1))
                    i += 1
                    # 对于簇间向量进行采样，如果没有看到所有簇就重复采样
                    random.shuffle(sent_batch)
                    random.shuffle(emb_batch)
                    batch_data.append((sent_batch, emb_batch))  # batch_size,1+(task_C-1)*4
        return batch_data

    def get_contrastive_data(self, batch_size):
        """
        From
        1. Positive and negative embeddings in one sentence.
        2. Between cluster
        3. In cluster different sentence positive embedding.
        4. From memory.
        """
        # 根据句子簇分类,得到分好类的变量假设是 cluster2sentences
        pn_batch_idx = []
        p_idx = []
        p_batch_idx = []  # get positive pool
        sent_id2quad_id = collections.defaultdict(list)
        count = 0

        # # memory update
        # for all_items in self.memory.values():
        #     for _, quad_ins in all_items:
        #         self.quadruple.append(quad_ins)
        # 得到句子序号到quadruple序号的映射，i是quadruple序号
        for i, quad in enumerate(self.quadruple):
            sent_id2quad_id[quad[0]].append(i)
            if quad[-1] == quad[-2]:
                p_idx.append(i)
        # memory update
        start_index = len(self.quadruple)
        len_mem = 0
        for all_items in self.memory.values():
            for _, quad_ins in all_items:
                self.quadruple.append(quad_ins)
                len_mem += 1

        mem_pool = list(range(start_index, start_index + len_mem))

        pool_num = batch_size // 2 + 1 if batch_size & 1 else batch_size // 2
        # 取簇间正向量形成训练数据
        for j in range(2):
            if len(mem_pool) > pool_num:
                p_batch_idx = [(i, 1) for i in self.multi_sample_no_replace(p_idx, batch_size // 2)]
                for i in self.multi_sample_no_replace(p_idx, batch_size // 2):
                    i.extend(np.random.choice(a=mem_pool, size=pool_num, replace=False).tolist())
                    random.shuffle(i)
                    p_batch_idx.append((i, 1))

        random.shuffle(p_batch_idx)

        return p_batch_idx

    def choice_object_no_replace(self, object_list, num, shuffle):
        idx_pool = list(range(len(object_list)))
        ans = []
        for ins_list in self.multi_sample_no_replace(idx_pool, num, shuffle=shuffle):
            ans.append([object_list[i] for i in ins_list])
        return ans

    def multi_sample_no_replace(self, list_collection, n, shuffle=True):
        if shuffle:
            random.shuffle(list_collection)
        return list(self.split_list_by_n(list_collection, n, last_to_n=True))

    def split_list_by_n(self, list_collection, n, last_to_n=False):
        """
        将list均分，每份n个元素
        :return:返回的结果为评分后的每份可迭代对象
        """

        for i in range(0, len(list_collection), n):
            if last_to_n:
                if (i + n) > len(list_collection):
                    yield list_collection[i:] + np.random.choice(a=list_collection, size=i + n - len(list_collection),
                                                                 replace=False).tolist()
                else:
                    yield list_collection[i: i + n]

            else:
                yield list_collection[i: i + n]

    def set_seed(self, seed):
        self.seed = seed
        if self.seed != None:
            random.seed(self.seed)
            np.random.seed(self.seed)

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class memory_fn(object):

    def __init__(self, id2sent):
        self.id2sent = id2sent

    def collect_fn(self, batch_data):
        labels = []
        tokens_id = []
        tokens = []
        for ins in batch_data:
            labels.append(torch.tensor(ins[-1]))
            tokens_id.append(torch.tensor(ins[0]))
            tokens.append(torch.tensor(self.id2sent[ins[0]]))
        labels = torch.stack(labels, dim=0)
        tokens = torch.stack(tokens, dim=0)
        tokens_id = torch.stack(tokens_id, dim=0)
        return (labels, tokens, tokens_id)

class data_sampler(object):

    def __init__(self, config=None, seed=None):

        self.config = config

        self.tokenizer = BertTokenizer.from_pretrained(self.config.bert_path,
                                                       additional_special_tokens=["[E11]", "[E12]", "[E21]", "[E22]"])

        # read relation data
        self.id2rel, self.rel2id = self._read_relations(config.relation_file)
        self.config.num_of_relation = len(self.rel2id)
        # random sampling
        self.seed = seed
        if self.seed != None:
            self.set_seed(self.seed)
        self.shuffle_index = list(range(len(self.id2rel)))  # rel id
        random.shuffle(self.shuffle_index)
        self.shuffle_index = np.argsort(self.shuffle_index)

        # regenerate data
        self.training_dataset, self.valid_dataset, self.test_dataset = self._read_data(self.config.data_file)

        # generate the task number
        self.batch = 0
        self.task_length = len(self.id2rel) // self.config.rel_per_task  # 每一轮任务进入几个新关系
        self.lb_id2train_id = list(range(len(self.shuffle_index)))

        # record relations
        self.seen_relations = []
        self.history_test_data = {}

    def set_seed(self, seed):
        self.seed = seed
        if self.seed != None:
            random.seed(self.seed)
        self.shuffle_index = list(range(len(self.id2rel)))
        random.shuffle(self.shuffle_index)
        self.shuffle_index = np.argsort(self.shuffle_index)

    def __iter__(self):
        return self

    def __len__(self):
        return self.task_length

    def __next__(self):

        if self.batch == self.task_length:
            self.batch == 0
            raise StopIteration()

        indexs = self.shuffle_index[
                 self.config.rel_per_task * self.batch: self.config.rel_per_task * (self.batch + 1)]  # 每个任务出现的id
        self.batch += 1

        current_relations = []
        cur_training_data = {}
        cur_valid_data = {}
        cur_test_data = {}

        for index in indexs:
            current_relations.append(self.id2rel[index])
            self.seen_relations.append(self.id2rel[index])

            cur_training_data[self.id2rel[index]] = self.training_dataset[index]
            cur_valid_data[self.id2rel[index]] = self.valid_dataset[index]
            cur_test_data[self.id2rel[index]] = self.test_dataset[index]
            self.history_test_data[self.id2rel[index]] = self.test_dataset[index]

        return cur_training_data, cur_valid_data, cur_test_data, current_relations, self.history_test_data, self.seen_relations

    def _read_data(self, file):
        '''
        :param file: the input sample file
        :return: samples for the model: [relation label, text]
        '''
        data = json.load(open(file, 'r', encoding='utf-8'))
        train_dataset = [[] for i in range(self.config.num_of_relation)]
        val_dataset = [[] for i in range(self.config.num_of_relation)]
        test_dataset = [[] for i in range(self.config.num_of_relation)]
        self.id2sent = {}
        for j, relation in enumerate(data.keys()):
            rel_samples = data[relation]
            if self.seed != None:
                random.seed(self.seed)
            random.shuffle(rel_samples)
            count = 0
            count1 = 0
            for i, sample in enumerate(rel_samples):
                tokenized_sample = {}
                tokenized_sample["tokens_id"] = j * len(rel_samples) + i
                tokenized_sample['relation'] = self.rel2id[sample['relation']]
                tokenized_sample['tokens'] = self.tokenizer.encode(' '.join(sample['tokens']),
                                                                   padding='max_length',
                                                                   truncation=True,
                                                                   max_length=self.config.max_length)
                self.id2sent[j * len(rel_samples) + i] = tokenized_sample['tokens']
                if self.config.task_name == 'FewRel':
                    if i < self.config.num_of_train:
                        train_dataset[self.rel2id[relation]].append(tokenized_sample)
                    elif i < self.config.num_of_train + self.config.num_of_val:
                        val_dataset[self.rel2id[relation]].append(tokenized_sample)
                    else:
                        test_dataset[self.rel2id[relation]].append(tokenized_sample)
                else:
                    if i < len(rel_samples) // 5 and count <= 40:
                        count += 1
                        test_dataset[self.rel2id[relation]].append(tokenized_sample)
                    else:
                        count1 += 1
                        train_dataset[self.rel2id[relation]].append(tokenized_sample)
                        if count1 >= 320:  # 一个关系最多320个样本
                            break
        return train_dataset, val_dataset, test_dataset

    def _read_relations(self, file):
        '''
        :param file: input relation file
        :return:  a list of relations, and a mapping from relations to their ids.
        '''
        id2rel = json.load(open(file, 'r', encoding='utf-8'))
        rel2id = {}
        for i, x in enumerate(id2rel):
            rel2id[x] = i
        return id2rel, rel2id

    def get_id2sent(self):
        return self.id2sent
