# import os
# import sys
# # 找到当前文件的决定路径,__file__ 表示当前文件,也就是test.py
# file_path = os.path.abspath(__file__)
# print(file_path)
# # 获取当前文件所在的目录
# cur_path = os.path.dirname(file_path)
# print(cur_path)
# # 获取项目所在路径
# project_path = os.path.dirname(cur_path)
# print(project_path)
# # 把项目路径加入python搜索路径
# sys.path.append(project_path)
import collections
import heapq
import time
import numpy as np
import torch
import random
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from argparse import ArgumentParser
from config import Config
import torch.nn.functional as F

from model.encoder.bert_encoder import Bert_Encoder
from model.classifier.dropout_layer import Dropout_Layer
from model.classifier.softmax_classifier import Softmax_Layer
from model.contrastive_network.contrastive_network import ContrastiveNetwork

from utils import batch2device

from Sampler.sample_dataloader import sample_dataloader, data_sampler
from data_loader import get_data_loader
from model.memory_network.attention_memory_simplified import Attention_Memory_Simplified


def train_contrastive(config, logger, model, optimizer, scheduler, loss_func, dataloader, evaluator,
                      memory_network=None, mem_data=None, epoch=None, FUNCODE=0):
    train_dataloader = dataloader
    epoch = epoch or config.contrast_epoch
    for ep in range(epoch):
        # train
        model.train()
        t_ep = time.time()
        # epoch parameter start
        batch_cum_loss, batch_cum_acc, total_head_rel_sample_acc, total_tail_rel_sample_acc = 0., 0., 0., 0.
        all_len = 0
        # epoch parameter end
        for batch_ind, batch_train_data in enumerate(train_dataloader):
            # batch parameter start
            t_batch = time.time()
            # batch parameter end
            # move to device
            batch_train_data = batch2device(batch_train_data, config.device)
            # zero the parameter gradients
            optimizer.zero_grad()
            sent_inp, emb_inp, labels, comparison = batch_train_data
            # model forward start
            if FUNCODE == 4:
                mem_for_batch = mem_data.clone()
                mem_for_batch.unsqueeze(0)
                mem_for_batch = mem_for_batch.expand(len(sent_inp) + len(emb_inp), -1, -1)
                inp_lst = [sent_inp, emb_inp, comparison, memory_network, mem_for_batch, 3]
            else:
                inp_lst = [sent_inp, emb_inp, comparison, None, None, 4]

            m_out = model(*inp_lst)
            # model forward end
            # grad operation start
            hidden = m_out
            loss = loss_func(hidden, labels)

            loss.backward()
            optimizer.step()

            # grad operation end

            # accuracy calculation start
            acc = (torch.argmax(hidden, dim=-1) == torch.argmax(labels, dim=-1)).sum() / hidden.shape[0]

            loss, acc = loss.item(), acc.item()

            batch_cum_loss += loss
            batch_cum_acc += acc

            batch_avg_loss = batch_cum_loss / (batch_ind + 1)
            batch_avg_acc = acc
            # accuracy calculation end
            batch_print_format = "\rContrastive Epoch: {}/{}, batch: {}/{}, train_loss: {}, " + "acc: {}, " + "lr: {}, batch_time: {}, total_time: {} -------------"
            # batch logger and print start
            print(batch_print_format.format(
                ep + 1,
                epoch,
                batch_ind + 1,
                len(train_dataloader),
                batch_avg_loss,
                batch_avg_acc,
                optimizer.param_groups[0]['lr'],
                time.time() - t_batch,
                time.time() - t_ep,
            ),
                end="")
            # batch logger and print end

            # change lr
            scheduler.step()

    # epoch logger and print start
    batch_avg_acc = batch_cum_acc / len(train_dataloader)
    return batch_avg_acc


def contrastive_loss(hidden, labels, FUNCODE=1):
    LARGE_NUM = 1e9
    if FUNCODE != 0:
        logsoftmax = nn.LogSoftmax(dim=-1)
        softmax = nn.Softmax(dim=-1)
        return -(logsoftmax(hidden) * labels).sum() / labels.sum()
    else:
        alpha = 0.25
        gamma = 2
        ce_loss = torch.nn.functional.cross_entropy(hidden, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        # mean over the batch
        return (alpha * (1 - pt) ** gamma * ce_loss).mean()


def compute_jsd_loss(m_input):
    # m_input: the result of m times dropout after the classifier.
    # size: m*B*C
    m = m_input.shape[0]
    mean = torch.mean(m_input, dim=0)
    jsd = 0
    for i in range(m):
        loss = F.kl_div(F.log_softmax(mean, dim=-1), F.softmax(m_input[i], dim=-1), reduction='none')
        loss = loss.sum()
        jsd += loss / m
    return jsd


def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def select_to_memory(config, encoder, dropout_layer, classifier, training_data, memory):
    data_loader = get_data_loader(config, training_data, batch_size=config.batch_size, shuffle=True)
    encoder.eval()
    dropout_layer.eval()
    classifier.eval()

    with torch.no_grad():
        for step, (labels, tokens, tokens_id) in enumerate(data_loader):
            # with torch.no_grad():
            tokens_id = torch.stack(tokens_id, dim=0)
            tokens = torch.stack([x.to(config.device) for x in tokens], dim=0)  # b,s
            labels = labels.to(config.device)  # b
            reps = encoder(tokens)
            output, output_embedding_true = dropout_layer(reps)  # B H
            logits = classifier(output)
            # model prediction
            out_prob = F.softmax(logits, dim=-1)
            max_idx_true = torch.argmax(out_prob, dim=-1)  # B
            enable_dropout(dropout_layer)
            out_prob = []
            logits_all = []
            for _ in range(config.f_pass):
                output, _ = dropout_layer(reps)
                logits = classifier(output)
                logits_all.append(logits)
                out_prob.append(F.softmax(logits, dim=-1))

            out_prob = torch.stack(out_prob)  # m b c
            out_std = torch.std(out_prob, dim=0)  # B,C

            labels_mask = torch.zeros_like(out_std).scatter_(-1, labels.view(
                labels.shape[0], 1), 1)  # B,C
            preds_mask = torch.zeros_like(out_std).scatter_(-1, max_idx_true.view(
                max_idx_true.shape[0], 1), 1)  # B,C
            # uncertainty
            slt_mask = (out_std < config.kappa_pos) * (labels_mask * preds_mask)  # 不满足被标志为1之后去除 #BC

            slt_idx = torch.sum(slt_mask, dim=-1) > 0  # B,C->B
            slt_tokens_ids, slt_embeddings, slt_preds, slt_labels = tokens_id[slt_idx], output_embedding_true[slt_idx], \
                                                                    max_idx_true[slt_idx], labels[slt_idx]
            slt_unct = torch.sum(out_std * slt_mask, dim=-1)[slt_idx]  #

            config.K = 10
            for i in range(len(slt_tokens_ids)):
                memory_list = memory[slt_labels[i].item()]
                k = len(memory_list)
                heapq.heappush(memory_list,
                               (-slt_unct[i].item(), (
                                   slt_tokens_ids[i].tolist(), slt_embeddings[i].cpu(), slt_preds[i].item(),
                                   slt_labels[i].item())))
                while k > config.K:
                    heapq.heappop(memory_list)
                    k -= 1
    return memory


def train_first(config, encoder, dropout_layer, classifier, training_data,  current_relations, rel2id, fix_label, FUNCODE=0):
    data_loader = get_data_loader(config, training_data, batch_size=config.batch_size, shuffle=True)
    epochs = config.train_epoch
    encoder.train()
    dropout_layer.train()
    classifier.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {'params': encoder.parameters(), 'lr': 0.00001},
        {'params': dropout_layer.parameters(), 'lr': 0.00001},
        {'params': classifier.parameters(), 'lr': 0.001}
    ])
    id2sent = []
    ret_d = []
    for epoch_i in range(epochs):
        losses = []
        batch_cum_loss, batch_cum_acc = 0., 0.
        t_ep = time.time()
        for step, (labels, tokens, tokens_id) in enumerate(data_loader):
            t_batch = time.time()
            optimizer.zero_grad()
            out_prob = []
            logits_all = []
            tokens = torch.stack([x.to(config.device) for x in tokens], dim=0)
            if not fix_label:
                labels=relation2id(current_relations,rel2id,labels)
            labels = labels.to(config.device)
            reps = encoder(tokens)
            output_embeddings = []
            for _ in range(config.f_pass):
                output, output_embedding = dropout_layer(reps)
                if epoch_i == epochs - 1:
                    output_embeddings.append(output_embedding)
                logits = classifier(output)
                logits_all.append(logits)
                out_prob.append(F.softmax(logits, dim=-1))
            out_prob = torch.stack(out_prob)  # m,B,C
            logits_all = torch.stack(logits_all)
            m_labels = labels.expand((config.f_pass, labels.shape[0]))  # m,B
            loss1 = criterion(logits_all.reshape(-1, logits_all.shape[-1]), m_labels.reshape(-1))
            loss2 = compute_jsd_loss(logits_all)
            loss = loss1 + loss2
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
            acc = (torch.argmax(logits_all, dim=-1) == m_labels).sum() / (config.f_pass * len(labels))
            batch_cum_loss += loss
            batch_cum_acc += acc
            batch_avg_loss = batch_cum_loss / (step + 1)
            batch_avg_acc = batch_cum_acc / (step + 1)
            batch_print_format = "\rFirst Epoch: {}/{}, batch: {}/{}, train_loss: {}, " + "acc: {}, " + "lr: {}, batch_time: {}, total_time: {} -------------"
            # batch logger and print start
            print(batch_print_format.format(
                epoch_i + 1,
                config.train_epoch,
                step + 1,
                len(data_loader),
                batch_avg_loss,
                batch_avg_acc,
                optimizer.param_groups[0]['lr'],
                time.time() - t_batch,
                time.time() - t_ep,
            ),
                end="")
        print(f"loss is {np.array(losses).mean()}")
    return ret_d


# Done
def get_proto(config, encoder, dropout_layer, mem_set):
    # aggregate the prototype set for further use.
    encoder.eval()
    dropout_layer.eval()
    data_loader = get_data_loader(config, mem_set, False, False, 1)

    features = []
    with torch.no_grad():
        for step, (labels, tokens, _) in enumerate(data_loader):
            tokens = torch.stack([x.to(config.device) for x in tokens], dim=0)
            with torch.no_grad():
                feature = dropout_layer(encoder(tokens))[1]
            features.append(feature)
        features = torch.cat(features, dim=0)
        proto = torch.mean(features, dim=0, keepdim=True).cpu()

    # return the averaged prototype
    return proto


# Use K-Means to select what samples to save, similar to at_least = 0
def select_data(config, encoder, dropout_layer, sample_set):
    data_loader = get_data_loader(config, sample_set, shuffle=False, drop_last=False, batch_size=1)
    features = []
    encoder.eval()
    dropout_layer.eval()
    for step, (labels, tokens, _) in enumerate(data_loader):
        with torch.no_grad():
            tokens = torch.stack([x.to(config.device) for x in tokens], dim=0)
            feature = dropout_layer(encoder(tokens))[1].cpu()
        features.append(feature)
    features = np.concatenate(features)
    num_clusters = min(config.num_protos, len(sample_set))
    distances = KMeans(n_clusters=num_clusters, random_state=0).fit_transform(features)
    mem_set = []
    for k in range(num_clusters):
        sel_index = np.argmin(distances[:, k])
        instance = sample_set[sel_index]
        mem_set.append(instance)
    return mem_set


def relation2id(current_relations,rel2id,relationslist):
    d={}
    for i in range(len(current_relations)):
        d[rel2id[current_relations[i]]]=i
    ans=[]
    relationslist=np.array(relationslist)
    for j in range(len(relationslist)):
        ans.append(d[relationslist[j]])
    return torch.LongTensor(torch.tensor(ans))


def train_simple_model(config, encoder, dropout_layer, classifier, training_data, epochs, current_relations, rel2id, fix_label, lb_id2train_id=None,
                       FUNCODE=2):
    data_loader = get_data_loader(config, training_data, shuffle=True)

    encoder.train()
    dropout_layer.train()
    classifier.train()

    criterion = nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.Adam([
        {'params': encoder.parameters(), 'lr': 0.00001},
        {'params': dropout_layer.parameters(), 'lr': 0.00001},
        {'params': classifier.parameters(), 'lr': 0.001}
    ])
    for epoch_i in range(epochs):
        losses = []
        for step, (labels, tokens, tokens_id) in enumerate(data_loader):
            optimizer.zero_grad()
            if not fix_label:
                labels=relation2id(current_relations, rel2id, labels)

            labels = labels.to(config.device)

            tokens = torch.stack([x.to(config.device) for x in tokens], dim=0)
            reps = encoder(tokens)
            reps, _ = dropout_layer(reps)
            logits = classifier(reps)
            loss = criterion(logits, labels)
            losses.append(loss.item())
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(encoder.parameters(), config.max_grad_norm)
            # torch.nn.utils.clip_grad_norm_(dropout_layer.parameters(), config.max_grad_norm)
            # torch.nn.utils.clip_grad_norm_(classifier.parameters(), config.max_grad_norm)
            optimizer.step()
        print(f"loss is {np.array(losses).mean()}")


def evaluate_first_model(config, encoder, dropout_layer, classifier, test_data, seen_relations):
    data_loader = get_data_loader(config, test_data, batch_size=1)
    encoder.eval()
    dropout_layer.eval()
    classifier.eval()
    n = len(test_data)
    correct = 0
    # cum_acc = 0
    for step, (labels, tokens, tokens_id) in enumerate(data_loader):
        labels = labels.to(config.device)
        tokens = torch.stack([x.to(config.device) for x in tokens], dim=0)
        reps = encoder(tokens)
        reps, _ = dropout_layer(reps)
        logits = classifier(reps)
        seen_relation_ids = [rel2id[relation] for relation in seen_relations]
        seen_sim = logits[:, seen_relation_ids].cpu().data.numpy()
        max_smi = np.max(seen_sim, axis=1)

        label_smi = logits[:, labels].cpu().data.numpy()

        if label_smi >= max_smi:
            correct += 1
    return correct / n


def evaluate_contrastive_model(config, contrastive_network, memory, test_data, use_mem_net=False, memory_network=None,
                               protos4eval=None, test_emb=None):
    batch_size = 2
    testdata_loader = get_data_loader(config, test_data, batch_size=batch_size)
    if use_mem_net:
        memory_network.eval()
    contrastive_network.eval()
    cum_right = 0
    cum_len = 0
    top_k_right = 0
    if test_emb is None:
        test_emb = {}
        for label_id, ins_list in memory.items():
            test_emb[label_id] = get_proto(config, encoder, dropout_layer, ins_list)
    mem_id2label = list(test_emb.keys())
    label2mem_id = {lb: i for i, lb in enumerate(mem_id2label)}

    with torch.no_grad():
        right = torch.cat(list(test_emb.values()), dim=0).to(config.device)  # B2*H
        for step, (labels, tokens, _) in enumerate(testdata_loader):
            labels = torch.stack([torch.tensor(label2mem_id[label.item()], device=config.device) for label in labels],
                                 dim=-1)
            tokens = torch.stack([x.to(config.device) for x in tokens], dim=0)  # B1*H
            if use_mem_net:
                mem_for_batch = protos4eval.clone()
                mem_for_batch.unsqueeze(0)
                mem_for_batch = mem_for_batch.expand(len(tokens) + len(right), -1, -1)
                logits_aa = contrastive_network(tokens, right, comparison=torch.ones(len(tokens), len(memory),
                                                                                     device=config.device),
                                                memory_network=memory_network, mem_for_batch=mem_for_batch,
                                                FUN_CODE=3)  # B1*B2
            else:
                logits_aa = contrastive_network(tokens, right, comparison=torch.ones(len(tokens), len(memory),
                                                                                     device=config.device),
                                                FUN_CODE=4)  # B1*B2
            results = logits_aa
            preds = torch.argmax(results, dim=-1)  # B
            _, topk_preds = torch.topk(results, k=2, dim=-1, largest=True)  # B K
            topk_mask = torch.zeros((preds.shape[0], len(label2mem_id)), device=config.device).scatter_(-1, topk_preds,
                                                                                                        1)
            labels_mask = torch.zeros((preds.shape[0], len(label2mem_id)), device=config.device).scatter_(-1,
                                                                                                          labels.view(
                                                                                                              -1, 1), 1)
            top_k_right += (topk_mask * labels_mask).sum().item()
            cum_right += (labels == preds).sum().item()
            cum_len += len(preds)

    print(f"Contrastive acc is {cum_right / cum_len}")
    return cum_right / cum_len, top_k_right / cum_len


def quads2origin_data(quads, id2sentence):
    ret_d = []
    for quad in quads:
        tokenized_sample = {}
        tokenized_sample["tokens_id"] = quad[0]
        tokenized_sample['relation'] = quad[-1]
        tokenized_sample['tokens'] = id2sentence[quad[0]]
        ret_d.append(tokenized_sample)
    return ret_d


if __name__ == '__main__':
    start_time=time.time()
    FUNCODE = 3  # Funcode==1为4类版本，为2为多类版本
    use_mem_network = False
    fix_labels = True
    parser = ArgumentParser(
        description="Config for lifelong relation extraction (classification)")
    parser.add_argument('--config', default='config.ini')
    args = parser.parse_args()
    config = Config(args.config)

    config.device = torch.device(config.device)
    config.n_gpu = torch.cuda.device_count()
    config.batch_size_per_step = int(config.batch_size / config.gradient_accumulation_steps)

    all_result_total=[]
    all_result_cur=[]
    # set training batch
    for i in range(config.total_round):
        test_cur = []
        test_total = []
        # set random seed
        random.seed(config.seed + i * 100)
        torch.manual_seed(config.seed+i*100)
        torch.cuda.manual_seed(config.seed+i*100)
        # sampler setup
        sampler = data_sampler(config=config, seed=config.seed + i * 100)
        id2sentence = sampler.get_id2sent()
        id2rel = sampler.id2rel
        rel2id = sampler.rel2id
        # encoder setup
        encoder = Bert_Encoder(config=config).to(config.device)
        # dropout setup
        dropout_layer = Dropout_Layer(config=config, input_size=encoder.output_size).to(config.device)
        # classifier setup
        if not fix_labels:
            num_class = config.rel_per_task
        else:
            num_class = config.num_of_relation

        classifier = Softmax_Layer(input_size=encoder.output_size, num_class=num_class).to(config.device)
        # no dropout_layer in encoder
        contrastive_network = ContrastiveNetwork(config=config, encoder=encoder, dropout_layer=dropout_layer,
                                                 hidden_size=config.encoder_output_size).to(
            config.device)
        # record testing results
        sequence_results = []
        result_whole_test = []

        decay_rate = config.decay_rate
        decay_steps = config.decay_steps

        # initialize memory and prototypes
        num_class = len(sampler.id2rel)

        memory = collections.defaultdict(list)
        cur_all_acc = []
        his_all_acc = []
        test_cur = []
        test_total = []
        test_top_cur = []
        test_top_total = []
        # load data and start computation
        for steps, (
                training_data, valid_data, test_data, current_relations, historic_test_data,
                seen_relations) in enumerate(
            sampler):

            print(current_relations)
            temp_protos = []

            train_data_for_initial = []
            for relation in current_relations:
                train_data_for_initial += training_data[relation]

            # First Training
            if steps == 0:
                tokens_task1 = train_data_for_initial
            if steps==1:
                tokens_task2 = train_data_for_initial


            train_simple_model(config, encoder, dropout_layer, classifier, train_data_for_initial,
                               config.step1_epochs, current_relations, rel2id, fix_labels)

            # first model
            train_first(config, encoder, dropout_layer, classifier, train_data_for_initial, current_relations, rel2id, fix_labels, FUNCODE=FUNCODE)

            # Second training
            # picture before contrastive learning


            for relation in current_relations:
                memory[rel2id[relation]] = select_data(config, encoder, dropout_layer,
                                                       training_data[relation])  # 训练以后会选择出个K个数据又记录下来
            rel_rep = {}
            for rel_id, ins_list in memory.items():
                rel_rep[rel_id] = get_proto(config, encoder, dropout_layer, ins_list)

            if steps == 0:
                memory1=memory
                rel_rep1=rel_rep
            if steps==1:
                memory2=memory
                rel_rep2=rel_rep

            task_sample = True
            ctst_dload = sample_dataloader(quadruple=train_data_for_initial, memory=memory, rel_rep=rel_rep,
                                           id2sent=id2sentence,
                                           config=config,
                                           seed=config.seed + steps * 100 + i * 100, FUN_CODE=FUNCODE,
                                           task_sample=task_sample)

            if use_mem_network:
                memory_network = Attention_Memory_Simplified(mem_slots=len(seen_relations),
                                                             input_size=encoder.output_size,
                                                             output_size=encoder.output_size,
                                                             key_size=config.key_size,
                                                             head_size=config.head_size
                                                             ).to(config.device)
                for ins_list in memory.values():
                    temp_protos.append(get_proto(config, encoder, dropout_layer, ins_list))
                temp_protos = torch.cat(temp_protos, dim=0).detach()  # 新和老关系都被选择到了
                optimizer = optim.Adam([
                    {'params': contrastive_network.parameters(), 'lr': 4e-5},
                    {'params': memory_network.parameters(), 'lr': 1e-4}
                ])
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                            step_size=decay_steps,
                                                            gamma=decay_rate)
                inp_dict = {
                    "config": config, "logger": None, "model": contrastive_network, "optimizer": optimizer,
                    "scheduler": scheduler, "loss_func": contrastive_loss, "dataloader": ctst_dload,
                    "evaluator": None,
                    "epoch": config.step2_epochs, "memory_network": memory_network, "mem_data": temp_protos,
                    "FUNCODE": FUNCODE,
                }
            else:
                optimizer = optim.Adam([
                    {'params': contrastive_network.parameters(), 'lr': 4e-5},
                ])
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                            step_size=decay_steps,
                                                            gamma=decay_rate)
                inp_dict = {
                    "config": config, "logger": None, "model": contrastive_network, "optimizer": optimizer,
                    "scheduler": scheduler, "loss_func": contrastive_loss, "dataloader": ctst_dload,
                    "evaluator": None,
                    "epoch": config.step2_epochs, "FUNCODE": 1,
                }


            train_contrastive(**inp_dict)


            torch.cuda.empty_cache()
            task_sample = False
            ctst_dload = sample_dataloader(quadruple=train_data_for_initial, rel_rep=rel_rep, memory=memory,
                                           id2sent=id2sentence,
                                           config=config,
                                           seed=config.seed + steps * 100 + i * 100, FUN_CODE=FUNCODE,
                                           task_sample=task_sample)
            if use_mem_network:

                optimizer = optim.Adam([
                    {'params': contrastive_network.parameters(), 'lr': 4e-5},
                    {'params': memory_network.parameters(), 'lr': 1e-4}
                ])
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                            step_size=decay_steps,
                                                            gamma=decay_rate)
                inp_dict = {
                    "config": config, "logger": None, "model": contrastive_network, "optimizer": optimizer,
                    "scheduler": scheduler, "loss_func": contrastive_loss, "dataloader": ctst_dload, "evaluator": None,
                    "epoch": config.step3_epochs, "memory_network": memory_network, "mem_data": temp_protos,
                    "FUNCODE": FUNCODE,
                }
            else:
                optimizer = optim.Adam([
                    {'params': contrastive_network.parameters(), 'lr': 4e-5},

                ])
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                            step_size=decay_steps,
                                                            gamma=decay_rate)
                inp_dict = {
                    "config": config, "logger": None, "model": contrastive_network, "optimizer": optimizer,
                    "scheduler": scheduler, "loss_func": contrastive_loss, "dataloader": ctst_dload, "evaluator": None,
                    "epoch": config.step3_epochs, "FUNCODE": 1,
                }
            train_contrastive(**inp_dict)




            test_data_1 = []
            for relation in current_relations:
                test_data_1 += test_data[relation]

            test_data_2 = []
            for relation in seen_relations:
                test_data_2 += historic_test_data[relation]
            # trash clean
            ctst_dload = None
            if not fix_labels:
                cur_acc = evaluate_first_model(config, encoder, dropout_layer, classifier, test_data_1, seen_relations)
                total_acc = evaluate_first_model(config, encoder, dropout_layer, classifier, test_data_2, seen_relations)
                cur_all_acc.append(cur_acc)
                his_all_acc.append(total_acc)
                print(f'\nFirst model current test acc:{cur_all_acc}')
                print(f'First model history test acc:{his_all_acc}')


            if use_mem_network:
                protos4eval = []
                for label_id,ins_list in memory.items():
                    protos4eval.append(get_proto(config, encoder, dropout_layer, ins_list))
                protos4eval = torch.cat(protos4eval, dim=0).detach()  # 新和老关系都被选择到了
                protos4eval=protos4eval.to(config.device)
            #current evaluation
            if not use_mem_network:
                memory_network = None
                protos4eval = None
            cont_cur_acc, topk_cur_acc = evaluate_contrastive_model(config, contrastive_network, memory, test_data_1,
                                                                    memory_network=memory_network,
                                                                    protos4eval=protos4eval,
                                                                    use_mem_net=use_mem_network)

            cont_total_acc, topk_total_acc = 0, 0

            cont_total_acc, topk_total_acc = evaluate_contrastive_model(config, contrastive_network, memory,
                                                                        test_data_2,
                                                                        memory_network=memory_network,
                                                                        protos4eval=protos4eval,
                                                                        use_mem_net=use_mem_network)

            print(f'\nContrastive model current test acc:{cont_cur_acc}')
            print(f'Contrastive model history test acc:{cont_total_acc}')
            test_cur.append(cont_cur_acc)
            test_total.append(cont_total_acc)

            test_top_cur.append(topk_cur_acc)
            test_top_total.append(topk_total_acc)
            print(f'\nContrastive model current topK test acc:{topk_cur_acc}')
            print(f'Contrastive model history topK acc:{topk_total_acc}')
            print("\ncontrastive all:")
            print(test_cur)
            print(test_total)
            print("\ncontrastive topk all:")
            print(test_top_cur)
            print(test_top_total)
            a, b = evaluate_contrastive_model(config, contrastive_network, memory, test_data_2,
                                              memory_network=memory_network,
                                              protos4eval=protos4eval, use_mem_net=use_mem_network, test_emb=rel_rep)
            print(
                f"Contrastive model use training embedding to test: history test acc:{a},Contrastive model history topK acc:{b}")
            if steps==9:
                all_result_total.append(test_total)
                all_result_cur.append(test_cur)

    print(f"all_result_cur{all_result_cur},all_result_total{all_result_total}")
    all_result=np.array(all_result_total)
    all_result_cur=np.array(all_result_cur)
    print(f"result_mean:{all_result.mean(axis=0)},result_std:{all_result.std(axis=0)}")
    print(f"result_cur_mean:{all_result_cur.mean(axis=0)},result_cur_std:{all_result_cur.std(axis=0)}")
    print(f"alltime:{time.time()-start_time}")
