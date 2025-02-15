# ---encoding:utf-8---
# @Author : yu
# @Email : 408423952@qq.com
# @IDE : PyCharm
# @File : main1.py

import sys
import os
import pickle
import csv
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from preprocess import data_loader, data_loader_kmer
from configuration import config as cf
from util import util_metric
from model import iDNA_ABT, TextCNN, Focal_Loss
from train.model_operation import save_model, adjust_model
from train.visualization import dimension_reduction, penultimate_feature_visulization

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
import pickle
import seaborn as sns
from BiLSTM_att import resnet_main, BiLSTM_main, Linear_full_main, performance
from model.Focal_Loss import FocalLoss
def load_data(config):
    if_multi_scaled = config.if_multi_scaled

    if if_multi_scaled:
        residue2idx = pickle.load(open('../data/kmer_residue2idx.pkl', 'rb'))
        config.vocab_size = len(residue2idx)
        config.token2index = residue2idx
        print('old config.vocab_size:', config.vocab_size)

        train_iter_orgin, test_iter = data_loader_kmer.load_data(config)
        config.max_len = 256
    else:
        residue2idx = pickle.load(open('../data/residue2idx.pkl', 'rb'))  # 太多了
        config.vocab_size = len(residue2idx)
        config.token2index = residue2idx

        train_iter_orgin, test_iter = data_loader.load_data(config)

    print('-' * 20, 'data construction over', '-' * 20)
    print('config.vocab_size', config.vocab_size)
    print('max_len_train', config.max_len_train)
    print('max_len_test', config.max_len_test)
    print('config.max_len', config.max_len)
    return train_iter_orgin, test_iter


def draw_figure_CV(config, fig_name):
    sns.set(style="darkgrid")
    plt.figure(22, figsize=(16, 12))
    plt.subplots_adjust(wspace=0.2, hspace=0.3)

    for i, e in enumerate(train_acc_record):
        train_acc_record[i] = e.cpu().detach()

    for i, e in enumerate(train_loss_record):
        train_loss_record[i] = e.cpu().detach()

    for i, e in enumerate(valid_acc_record):
        valid_acc_record[i] = e.cpu().detach()

    for i, e in enumerate(valid_loss_record):
        valid_loss_record[i] = e.cpu().detach()

    plt.subplot(2, 2, 1)
    plt.title("Train Acc Curve", fontsize=23)
    plt.xlabel("Step", fontsize=20)
    plt.ylabel("Accuracy", fontsize=20)
    plt.plot(step_log_interval, train_acc_record)
    plt.subplot(2, 2, 2)
    plt.title("Train Loss Curve", fontsize=23)
    plt.xlabel("Step", fontsize=20)
    plt.ylabel("Loss", fontsize=20)
    plt.plot(step_log_interval, train_loss_record)
    plt.subplot(2, 2, 3)
    plt.title("Validation Acc Curve", fontsize=23)
    plt.xlabel("Epoch", fontsize=20)
    plt.ylabel("Accuracy", fontsize=20)
    plt.plot(step_valid_interval, valid_acc_record)
    plt.subplot(2, 2, 4)
    plt.title("Validation Loss Curve", fontsize=23)
    plt.xlabel("Step", fontsize=20)
    plt.ylabel("Loss", fontsize=20)
    plt.plot(step_valid_interval, valid_loss_record)

    plt.savefig(config.result_folder + '/' + fig_name + '.png')
    plt.show()


def draw_figure_train_test(config, fig_name):
    sns.set(style="darkgrid")
    plt.figure(22, figsize=(16, 12))
    plt.subplots_adjust(wspace=0.2, hspace=0.3)

    for i, e in enumerate(train_acc_record):
        train_acc_record[i] = e.cpu().detach()

    for i, e in enumerate(train_loss_record):
        train_loss_record[i] = e.cpu().detach()

    for i, e in enumerate(test_acc_record):
        test_acc_record[i] = e.cpu().detach()

    for i, e in enumerate(test_loss_record):
        test_loss_record[i] = e.cpu().detach()

    plt.subplot(2, 2, 1)
    plt.title("Train Acc Curve", fontsize=23)
    plt.xlabel("Step", fontsize=20)
    plt.ylabel("Accuracy", fontsize=20)
    plt.plot(step_log_interval, train_acc_record)
    plt.subplot(2, 2, 2)
    plt.title("Train Loss Curve", fontsize=23)
    plt.xlabel("Step", fontsize=20)
    plt.ylabel("Loss", fontsize=20)
    plt.plot(step_log_interval, train_loss_record)
    plt.subplot(2, 2, 3)
    plt.title("Test Acc Curve", fontsize=23)
    plt.xlabel("Epoch", fontsize=20)
    plt.ylabel("Accuracy", fontsize=20)
    plt.plot(step_test_interval, test_acc_record)
    plt.subplot(2, 2, 4)
    plt.title("Test Loss Curve", fontsize=23)
    plt.xlabel("Step", fontsize=20)
    plt.ylabel("Loss", fontsize=20)
    plt.plot(step_test_interval, test_loss_record)

    plt.savefig(config.result_folder + '/' + fig_name + '.png')
    plt.show()


def cal_loss_dist_by_cosine(model):
    embedding = model.embedding
    loss_dist = 0

    vocab_size = embedding[0].tok_embed.weight.shape[0]
    d_model = embedding[0].tok_embed.weight.shape[1]

    Z_norm = vocab_size * (len(embedding) ** 2 - len(embedding)) / 2

    for i in range(len(embedding)):
        for j in range(len(embedding)):
            if i < j:
                cosin_similarity = torch.cosine_similarity(embedding[i].tok_embed.weight, embedding[j].tok_embed.weight)
                loss_dist -= torch.sum(cosin_similarity)
                # print('cosin_similarity.shape', cosin_similarity.shape)
    loss_dist = loss_dist / Z_norm
    return loss_dist


def get_entropy(probs):
    ent = -(probs.mean(0) * torch.log2(probs.mean(0) + 1e-12)).sum(0, keepdim=True)
    return ent


def get_cond_entropy(probs):
    cond_ent = -(probs * torch.log(probs + 1e-12)).sum(1).mean(0, keepdim=True)
    return cond_ent


def get_val_loss(logits, label, criterion):
    loss = criterion(logits.view(-1, config.num_class), label.view(-1))
    loss = (loss.float()).mean()
    loss = (loss - config.b).abs() + config.b

    # Q_sum = len(logits)
    logits = F.softmax(logits, dim=1)  # softmax归一化
    # hat_sum_p0 = logits[:, 0].sum()/Q_sum  # 概率和 和  logits.mean(0)一致
    # hat_sum_p1 = logits[:, 1].sum()/Q_sum
    # mul_hat_p0 = hat_sum_p0.mul(torch.log(hat_sum_p0))
    # mul_hat_p1 = hat_sum_p1.mul(torch.log(hat_sum_p1))
    # mul_p0 = logits[:, 0].mul(torch.log(logits[:, 0])).sum()/Q_sum
    # mul_p1 = logits[:, 1].mul(torch.log(logits[:, 1])).sum()/Q_sum

    sum_loss = loss + get_entropy(logits) - get_cond_entropy(logits)
    return sum_loss[0]


def get_loss(logits, label, criterion):
    loss = criterion(logits.view(-1, config.num_class), label.view(-1))
    loss = (loss.float()).mean()
    loss = (loss - config.b).abs() + config.b

    # multi-sense loss
    # alpha = -0.1
    # loss_dist = alpha * cal_loss_dist_by_cosine(model)
    # loss += loss_dist

    return loss


def periodic_test(test_iter, model, criterion, config, sum_epoch):
    print('#' * 60 + 'Periodic Test' + '#' * 60)
    test_metric, test_loss, test_repres_list, test_label_list, \
    test_roc_data, test_prc_data = model_eval(test_iter, model, criterion, config)

    print('test current performance')
    print('[ACC,\tPrecision,\tSensitivity,\tSpecificity,\tF1,\tAUC,\tMCC]')
    print(test_metric.numpy())
    print('#' * 60 + 'Over' + '#' * 60)

    step_test_interval.append(sum_epoch)
    test_acc_record.append(test_metric[0])
    test_loss_record.append(test_loss)

    return test_metric, test_loss, test_repres_list, test_label_list, test_roc_data, test_prc_data


def periodic_valid(valid_iter, model, criterion, optimizer, config, sum_epoch):
    print('#' * 60 + 'Periodic Validation' + '#' * 60)

    valid_metric, valid_loss, valid_repres_list, valid_label_list, \
    valid_roc_data, valid_prc_data = model_eval(valid_iter, model, criterion, config)  # TO-DO iDNA-AB
    # valid_caculate_metric, valid_prc_data = model_val_eval_grad(valid_iter, model, criterion, optimizer, config)  # TO-DO iDNA-ABT

    print('validation current performance')
    print('[ACC,\tPrecision,\tSensitivity,\tSpecificity,\tF1,\tAUC,\tMCC]')
    print(valid_metric.numpy())
    print('#' * 60 + 'Over' + '#' * 60)

    step_valid_interval.append(sum_epoch)
    valid_acc_record.append(valid_metric[0])
    valid_loss_record.append(valid_loss)

    return valid_metric, valid_loss, valid_repres_list, valid_label_list


def train_ACP(train_iter, valid_iter, test_iter, model, optimizer, criterion, config, iter_k):
    steps = 0
    best_acc = 0
    best_performance = 0

    for epoch in range(1, config.epoch + 1):
        repres_list = []
        label_list = []
        for batch in train_iter:
            if config.if_multi_scaled:
                input, origin_input, label = batch
                # logits, output = model(input, origin_input)
            #     20230601
                logits, output = model(input)
            else:
                input, label = batch
                logits, output = model(input)
                # repres_list.extend(output.cpu().detach().numpy())
                # label_list.extend(label.cpu().detach().numpy())
            # #   20230902 提取特征文件
            # if epoch == 100:
            #         features = logits
            #         features1 = features.cpu()
            #         features2 = features1.detach().numpy()
            #         labels = label.cpu()
            #         labels1 = labels.detach().numpy()
            #         output_file = "6mA_Tolypocladium_100_features_and_labels.csv"
            #
            #         # 将特征和标签写入CSV文件
            #         with open(output_file, mode='a', newline='') as file:
            #             writer = csv.writer(file)
            #
            #             # 写入文件的表头，如果需要的话
            #             # writer.writerow(['Feature1', 'Feature2', 'Feature3', 'Label'])
            #
            #             # 将特征和标签一行一行地写入文件
            #             for feature3, label2 in zip(features2, labels1):
            #                 # feature = features2.numpy()
            #                 writer.writerow([feature3, label2])
            #
            #         print(f"特征和标签已写入文件: {output_file}")
            # # #20230902
            loss = get_loss(logits, label, criterion)
                # loss = get_val_loss(logits, label, criterion)  # 加入信息熵

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            steps += 1

        '''Periodic Train Log'''
        if steps % config.interval_log == 0:
            corrects = (torch.max(logits, 1)[1] == label).sum()
            the_batch_size = label.shape[0]
            train_acc = 100.0 * corrects / the_batch_size
            sys.stdout.write(
                '\rEpoch[{}] Batch[{}] - loss: {:.6f} | ACC: {:.4f}%({}/{})'.format(epoch, steps,
                                                                                    loss,
                                                                                    train_acc,
                                                                                    corrects,
                                                                                    the_batch_size))
            print()

            step_log_interval.append(steps)
            train_acc_record.append(train_acc)
            train_loss_record.append(loss)

        sum_epoch = iter_k * config.epoch + epoch

        '''Periodic Validation'''
        if valid_iter and sum_epoch % config.interval_valid == 0:
            valid_metric, valid_loss, valid_repres_list, valid_label_list = periodic_valid(valid_iter,
                                                                                           model,
                                                                                           criterion,
                                                                                           optimizer,
                                                                                           config,
                                                                                           sum_epoch,
                                                                                           )
            # valid_acc = valid_metric[0]
            # if valid_acc > best_acc:
            #     best_acc = valid_acc
            #     best_performance = valid_metric

        '''Periodic Test'''
        if test_iter and sum_epoch % config.interval_test == 0:
            test_metric, test_loss, test_repres_list, test_label_list, test_roc_data, test_prc_data = periodic_test(
                test_iter,
                model,
                criterion,
                config,
                sum_epoch)

            '''Periodic Save'''
            # save the model if specific conditions are met
            test_acc = test_metric[0]
            if test_acc > best_acc:
                best_acc = test_acc
                roc_path = config.species + '_roc.txt'
                pr_path = config.species + '_pr.txt'
                file = open(roc_path, "wb")
                pickle.dump(test_roc_data, file)  # 保存list到文件
                file.close()
                file = open(pr_path, "wb")
                pickle.dump(test_prc_data, file)  # 保存list到文件
                file.close()
                best_performance = test_metric
                if config.save_best and best_acc > config.threshold:
                    save_model(model.state_dict(), best_acc, config.result_folder, config.learn_name)

            # test_label_list = [x + 2 for x in test_label_list]
            repres_list.extend(test_repres_list)
            label_list.extend(test_label_list)
            # print( 'repres_list', len(repres_list))
            '''feature dimension reduction'''
            # if sum_epoch % 1 == 0 or epoch == 1:
            #     dimension_reduction(repres_list, label_list, epoch, config)

            '''reduction feature visualization'''
            # if sum_epoch % 10 == 0 or epoch == 1 or (epoch % 2 == 0 and epoch <= 10):
            #     penultimate_feature_visulization(repres_list, label_list, epoch, config)

    return best_performance


def model_eval(data_iter, model, criterion, config):
    device = torch.device("cuda" if config.cuda else "cpu")
    label_pred = torch.empty([0], device=device)
    label_real = torch.empty([0], device=device)
    pred_prob = torch.empty([0], device=device)
    AUC0 = 0
    print('model_eval data_iter', len(data_iter))

    iter_size, corrects, avg_loss = 0, 0, 0
    repres_list = []
    label_list = []

    with torch.no_grad():
        for batch in data_iter:
            if config.if_multi_scaled:
                input, origin_inpt, label = batch
                logits, output = model(input, origin_inpt)
            else:
                input, label = batch
                logits, output = model(input)
            # 20230915 提取测试数据的特征
            #   20230902 提取特征文件
            #
            # features = logits
            # features1 = features.cpu()
            # features2 = features1.detach().numpy()
            # labels = label.cpu()
            # labels1 = labels.detach().numpy()
            # output_file = "4mC_S.cerevisiae_features_and_labels.csv"
            #
            # # 将特征和标签写入CSV文件
            # with open(output_file, mode='a', newline='') as file:
            #     writer = csv.writer(file)
            #
            #     # 写入文件的表头，如果需要的话
            #     # writer.writerow(['Feature1', 'Feature2', 'Feature3', 'Label'])
            #
            #     # 将特征和标签一行一行地写入文件
            #     for feature3, label2 in zip(features2, labels1):
            #         # feature = features2.numpy()
            #         writer.writerow([feature3, label2])
            #
            #     print(f"特征和标签已写入文件: {output_file}")
            #     #20230902

            #

            repres_list.extend(output.cpu().detach().numpy())
            label_list.extend(label.cpu().detach().numpy())

            loss = criterion(logits.view(-1, config.num_class), label.view(-1))
            loss = (loss.float()).mean()

            avg_loss += loss

            pred_prob_all = F.softmax(logits, dim=1)
            # Prediction probability [batch_size, class_num]
            pred_prob_positive = pred_prob_all[:, 1]
            # Probability of predicting positive classes [batch_size]
            pred_prob_sort = torch.max(pred_prob_all, 1)
            # The maximum probability of prediction in each sample [batch_size]
            pred_class = pred_prob_sort[1]
            # The location (class) of the predicted maximum probability in each sample [batch_size]
            corrects += (pred_class == label).sum()

            iter_size += label.shape[0]

            label_pred = torch.cat([label_pred, pred_class.float()])
            label_real = torch.cat([label_real, label.float()])
            pred_prob = torch.cat([pred_prob, pred_prob_positive])

    metric, roc_data, prc_data, AUC, fpr, tpr = util_metric.caculate_metric(label_pred, label_real, pred_prob)
    if AUC > AUC0:
        AUC0 = AUC
        fpr1 = fpr
        tpr1 = tpr
        # util_metric.ROC(fpr1, tpr1, AUC0)
        with open("r_k_roc_curve_data.txt", "w") as file:
            # 遍历 fpr 和 tpr 列表，将数据写入文件
            for i in range(len(fpr)):
                file.write(f"FPR: {fpr[i]}, TPR: {tpr[i]}\n")

    avg_loss /= iter_size
    # accuracy = 100.0 * corrects / iter_size
    accuracy = metric[0]
    print('Evaluation - loss: {:.6f}  ACC: {:.4f}%({}/{})'.format(avg_loss,
                                                                  accuracy,
                                                                  corrects,
                                                                  iter_size))

    return metric, avg_loss, repres_list, label_list, roc_data, prc_data


def k_fold_CV(train_iter_orgin, test_iter, config):
    test_performance_list = []

    for iter_k in range(1):  # config.k_fold
        print('=' * 50, 'iter_k={}'.format(iter_k + 1), '=' * 50)

        # Cross validation on training set
        train_iter = [x for i, x in enumerate(train_iter_orgin) if i % config.k_fold != iter_k]
        valid_iter = [x for i, x in enumerate(train_iter_orgin) if i % config.k_fold == iter_k]

        # train_iter = train_iter.to(config.device)
        # valid_iter = valid_iter.to(config.device)
        print('----------Data Selection----------')
        print('train_iter index', [i for i, x in enumerate(train_iter_orgin) if i % config.k_fold != iter_k])
        print('valid_iter index', [i for i, x in enumerate(train_iter_orgin) if i % config.k_fold == iter_k])

        print('len(train_iter_orgin)', len(train_iter_orgin))
        print('len(train_iter)', len(train_iter))
        print('len(valid_iter)', len(valid_iter))
        if test_iter:
            print('len(test_iter)', len(test_iter))
        print('----------Data Selection Over----------')

        if config.model_name == 'iDNA_ABT':
            model = iDNA_ABT.BERT(config)

        if config.cuda: model.cuda()
        adjust_model(model)

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.reg)
        # criterion = nn.CrossEntropyLoss()
        # alpha = torch.tensor([0.5, 0.5])
        # alpha.cuda()
        # criterion = Focal_Loss.FocalLoss(2, alpha=alpha, gamma=2)
        criterion = nn.CrossEntropyLoss()
        model.train()

        print('=' * 50 + 'Start Training' + '=' * 50)
        test_performance = train_ACP(train_iter, valid_iter, test_iter, model, optimizer, criterion, config, iter_k)
        print('=' * 50 + 'Train Finished' + '=' * 50)

        print('=' * 40 + 'C'
                         'ross Validation iter_k={}'.format(iter_k + 1), '=' * 40)
        valid_metric, valid_loss, valid_repres_list, valid_label_list, \
        valid_roc_data, valid_prc_data = model_eval(valid_iter, model, criterion, config)
        print('[ACC,\tPrecision,\tSensitivity,\tSpecificity,\tF1,\tAUC,\tMCC]')
        print(valid_metric.numpy())
        print('=' * 40 + 'Cross Validation Over' + '=' * 40)

        test_performance_list.append(test_performance)

        '''draw figure'''
        draw_figure_CV(config, config.learn_name + '_k[{}]'.format(iter_k + 1))

        '''reset plot data'''
        step_log_interval = []
        train_acc_record = []
        train_loss_record = []
        step_valid_interval = []
        valid_acc_record = []
        valid_loss_record = []

    return model, test_performance_list


def train_test(train_iter, test_iter, config):
    print('=' * 50, 'train-test', '=' * 50)
    print('len(train_iter)', len(train_iter))
    print('len(test_iter)', len(test_iter))

    if config.model_name == 'iDNA_ABT':
        print('model_name', config.model_name)
        model = iDNA_ABT.BERT(config)
    elif config.model_name == 'TextCNN':
        print('model_name', config.model_name)
        model = TextCNN.TextCNN(config)
    # elif config.model_name == 'BiLSTM':
    #     print('model_name', config.model_name)
    # # # 20230523
    # data_name = train_iter.dataset
    # batch_size = 16
    # epochs = 40
    # learning_rate = 1e-4
    # best_acc = 0.0
    # test_dataset = test_iter.dataset
    # model = BiLSTM_main(data_name, epochs, batch_size, learning_rate, best_acc, test_dataset)
    #
    # # 20230523
    if config.cuda: model.cuda()
    adjust_model(model)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config.lr, weight_decay=config.reg)
    # criterion = nn.CrossEntropyLoss()
    # alpha = torch.tensor([0.5, 0.5])
    # alpha.cuda()
    # self.loss_func = Focal_Loss.FocalLoss(self.config.num_class, alpha=alpha,
    #                                                   gamma=self.config.gamma)
    criterion = nn.CrossEntropyLoss()
    model.train()

    print('=' * 50 + 'Start Training' + '=' * 50)
    best_performance = train_ACP(train_iter, None, test_iter, model, optimizer, criterion, config, 0)
    print('=' * 50 + 'Train Finished' + '=' * 50)

    print('*' * 60 + 'The Last Test' + '*' * 60)
    last_test_metric, last_test_loss, last_test_repres_list, last_test_label_list, \
    last_test_roc_data, last_test_prc_data = model_eval(test_iter, model, criterion, config)
    print('[ACC,\tPrecision,\tSensitivity,\tSpecificity,\tF1,\tAUC,\tMCC]')
    print(last_test_metric.numpy())
    print('*' * 60 + 'The Last Test Over' + '*' * 60)

    return model, best_performance, last_test_metric


def select_dataset():
    # RNA
    # path_train_data = 'D:/yuxia/test/20230817/iDNA_ABT-test/data/RNA/h_b_all.tsv'
    # path_test_data = 'D:/yuxia/test/20230817/iDNA_ABT-test/data/RNA/h_b_Test.tsv'
    # path_train_data = '../data/RNA/h_k_all.tsv'
    # path_test_data = '../data/RNA/h_k_Test.tsv'
    # path_train_data = '../data/RNA/r_l_all.tsv'
    # path_test_data = '../data/RNA/r_l_Test.tsv'
    # path_train_data = '../data/RNA/h_l_all.tsv'
    # path_test_data = '../data/RNA/h_l_Test.tsv'
    # path_train_data = '../data/RNA/m_b_all.tsv'
    # path_test_data = '../data/RNA/m_b_Test.tsv'
    # path_train_data = '../data/RNA/m_h_all.tsv'
    # path_test_data = '../data/RNA/m_h_Test.tsv'
    # path_train_data = '../data/RNA/m_k_all.tsv'
    # path_test_data = '../data/RNA/m_k_Test.tsv'
    # path_train_data = '../data/RNA/m_l_all.tsv'
    # path_test_data = '../data/RNA/m_l_Test.tsv'
    # path_train_data = '../data/RNA/m_t_all.tsv'
    # path_test_data = '../data/RNA/m_t_Test.tsv'
    # path_train_data = '../data/RNA/r_b_all.tsv'
    # path_test_data = '../data/RNA/r_b_Test.tsv'
    path_train_data = '../data/RNA/r_k_all.tsv'
    path_test_data = '../data/RNA/r_k_Test.tsv'
    # path_train_data = '../data/RNA/r_l_all.tsv'
    # path_test_data = '../data/RNA/r_l_Test.tsv'

    # DNA-MSb
    # path_train_data = '../data/DNA_MS/tsv/5hmC/5hmC_H.sapiens/train.tsv'
    # path_test_data = '../data/DNA_MS/tsv/5hmC/5hmC_H.sapiens/test.tsv'
    # path_train_data = '../data/DNA_MS/tsv/5hmC/5hmC_M.musculus/train.tsv'
    # path_test_data = '../data/DNA_MS/tsv/5hmC/5hmC_M.musculus/test.tsv'
    # path_train_data = '../data/DNA_MS/tsv/4mC/4mC_C.equisetifolia/train.tsv'
    # path_test_data = '../data/DNA_MS/tsv/4mC/4mC_C.equisetifolia/test.tsv'
    # path_train_data = '../data/DNA_MS/tsv/4mC/4mC_F.vesca/train.tsv'
    # path_test_data = '../data/DNA_MS/tsv/4mC/4mC_F.vesca/test.tsv'
    # path_train_data = '../data/DNA_MS/tsv/4mC/4mC_S.cerevisiae/train.tsv'
    # path_test_data = '../data/DNA_MS/tsv/4mC/4mC_S.cerevisiae/test.tsv'
    # path_train_data = '../data/DNA_MS/tsv/4mC/4mC_Tolypocladium/train.tsv'
    # path_test_data = '../data/DNA_MS/tsv/4mC/4mC_Tolypocladium/test.tsv'
    # path_train_data = '../data/DNA_MS/tsv/6mA/6mA_A.thaliana/train.tsv'
    # path_test_data = '../data/DNA_MS/tsv/6mA/6mA_A.thaliana/test.tsv'
    # path_train_data = '../data/DNA_MS/tsv/6mA/6mA_C.elegans/train.tsv'
    # path_test_data = '../data/DNA_MS/tsv/6mA/6mA_C.elegans/test.tsv'
    # path_train_data = '../data/DNA_MS/tsv/6mA/6mA_C.equisetifolia/train.tsv'
    # path_test_data = '../data/DNA_MS/tsv/6mA/6mA_C.equisetifolia/test.tsv'
    # path_train_data = '../data/DNA_MS/tsv/6mA/6mA_D.melanogaster/train.tsv'
    # path_test_data = '../data/DNA_MS/tsv/6mA/6mA_D.melanogaster/test.tsv'
    # path_train_data = '../data/DNA_MS/tsv/6mA/6mA_F.vesca/train.tsv'
    # path_test_data = '../data/DNA_MS/tsv/6mA/6mA_F.vesca/test.tsv'
    # path_train_data = '../data/DNA_MS/tsv/6mA/6mA_H.sapiens/train.tsv'
    # path_test_data = '../data/DNA_MS/tsv/6mA/6mA_H.sapiens/test.tsv'
    # path_train_data = '../data/DNA_MS/tsv/6mA/6mA_R.chinensis/train.tsv'
    # path_test_data = '../data/DNA_MS/tsv/6mA/6mA_R.chinensis/test.tsv'
    # path_train_data = '../data/DNA_MS/tsv/6mA/6mA_S.cerevisiae/train.tsv'
    # path_test_data = '../data/DNA_MS/tsv/6mA/6mA_S.cerevisiae/test.tsv'
    # path_train_data = '../data/DNA_MS/tsv/6mA/6mA_T.thermophile/train.tsv'
    # path_test_data = '../data/DNA_MS/tsv/6mA/6mA_T.thermophile/test.tsv'
    # path_train_data = '../data/DNA_MS/tsv/6mA/6mA_Tolypocladium/train.tsv'
    # path_test_data = '../data/DNA_MS/tsv/6mA/6mA_Tolypocladium/test.tsv'
    # path_train_data = '../data/DNA_MS/tsv/6mA/6mA_Xoc BLS256/train.tsv'
    # path_test_data = '../data/DNA_MS/tsv/6mA/6mA_Xoc BLS256/test.tsv'
    return path_train_data, path_test_data


def load_config():
    '''The following variables need to be actively determined for each training session:
       1.train-name: Name of the training
       2.path-config-data: The path of the model configuration. 'None' indicates that the default configuration is loaded
       3.path-train-data: The path of training set
       4.path-test-data: Path to test set

       Each training corresponds to a result folder named after train-name, which contains:
       1.report: Training report
       2.figure: Training figure
       3.config: model configuration
       4.model_save: model parameters
       5.others: other data
       '''

    '''Set the required variables in the configuration'''
    train_name = 'iDNA_ABT'
    path_config_data = None
    path_train_data, path_test_data = select_dataset()

    '''Get configuration'''
    if path_config_data is None:
        config = cf.get_train_config()
    else:
        config = pickle.load(open(path_config_data, 'rb'))

    '''Modify default configuration'''
    # config.epoch = 50

    '''Set other variables'''
    # flooding method
    b = 0.06
    model_name = 'iDNA_ABT'
    # model_name = 'TextCNN'

    '''initialize result folder'''
    result_folder = '../result/' + config.learn_name
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    '''Save all variables in configuration'''
    config.train_name = train_name
    config.path_train_data = path_train_data
    config.path_test_data = path_test_data
    config.species = path_train_data.replace('/train.tsv', '')
    config.b = b
    config.if_multi_scaled = False
    # config.if_multi_scaled = True
    config.model_name = model_name
    config.result_folder = result_folder
    config.train_data = path_train_data  # 方便保存图片

    return config


if __name__ == '__main__':
    np.set_printoptions(linewidth=400, precision=4)
    time_start = time.time()

    '''load configuration'''
    config = load_config()

    '''set device'''
    torch.cuda.set_device(config.device)

    '''load data'''
    train_iter, test_iter = load_data(config)
    print('=' * 20, 'load data over', '=' * 20)

    '''draw preparation'''
    step_log_interval = []
    train_acc_record = []
    train_loss_record = []
    step_valid_interval = []
    valid_acc_record = []
    valid_loss_record = []
    step_test_interval = []
    test_acc_record = []
    test_loss_record = []

    '''train procedure'''
    valid_performance = 0
    best_performance = 0
    last_test_metric = 0
    # 20230601
    config.is_train = False
    if config.k_fold == -1:
        # train and test
        model, best_performance, last_test_metric = train_test(train_iter, test_iter, config)
        pass
    else:
        # k cross validation
        model, test_performance_list = k_fold_CV(train_iter, test_iter, config)  # none

    '''draw figure'''
    draw_figure_train_test(config, config.learn_name)

    '''report result'''
    print('*=' * 50 + 'Result Report' + '*=' * 50)
    if config.k_fold != -1:
        print('test_performance_list', test_performance_list)
        tensor_list = [x.view(1, -1) for x in test_performance_list]
        cat_tensor = torch.cat(tensor_list, dim=0)
        metric_mean = torch.mean(cat_tensor, dim=0)

        print('valid mean performance')
        print('\t[ACC,\tPrecision,\tSensitivity,\tSpecificity,\tF1,\tAUC,\tMCC]')
        print('\t{}'.format(metric_mean.numpy()))

        print('valid_performance list')
        print('\t[ACC,\tPrecision,\tSensitivity,\tSpecificity,\tF1,\tAUC,\tMCC]')
        for tensor_metric in test_performance_list:
            print('\t{}'.format(tensor_metric.numpy()))
    else:
        print('last test performance')
        print('\t[ACC,\tPrecision,\tSensitivity,\tSpecificity,\tF1,\tAUC,\tMCC]')
        print('\t{}'.format(last_test_metric))
        print()
        print('best_performance')
        print('\t[ACC,\tPrecision,\tSensitivity,\tSpecificity,\tF1,\tAUC,\tMCC]')
        print('\t{}'.format(best_performance))

    print('*=' * 50 + 'Report Over' + '*=' * 50)

    '''save train result'''
    # save the model if specific conditions are met
    if config.k_fold == -1:
        best_acc = best_performance[0]
        last_test_acc = last_test_metric[0]
        if last_test_acc > best_acc:
            best_acc = last_test_acc
            best_performance = last_test_metric
            if config.save_best and best_acc >= config.threshold:
                save_model(model.state_dict(), best_acc, config.result_folder, config.learn_name)

    # save the model configuration
    with open(config.result_folder + '/config.pkl', 'wb') as file:
        pickle.dump(config, file)
    print('-' * 50, 'Config Save Over', '-' * 50)

    time_end = time.time()
    print('total time cost', time_end - time_start, 'seconds')
