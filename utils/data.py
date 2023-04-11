# 载入数据集 并且做到划分
# import h5py
import numpy as np
import scipy.io as sio
import torch
from sklearn import preprocessing
import sys
import h5py
import path
import os
import datetime

def generate_syn_feature(opt, netG, classes, attribute, num):
    # print(classes)
    nclass = len(classes)
    syn_feature = torch.FloatTensor(nclass * num, opt.resSize)
    syn_label = torch.LongTensor(nclass * num)
    syn_att = torch.FloatTensor(num, opt.attSize)
    syn_noise = torch.FloatTensor(num, opt.nz)
    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()

    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]
        syn_att.copy_(iclass_att.repeat(num, 1))
        syn_noise.normal_(0, 1)
        with torch.no_grad():
            output = netG(syn_noise, syn_att)
        syn_feature.narrow(0, i * num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i * num, num).fill_(iclass)
    return syn_feature, syn_label


class Logger(object):
    def __init__(self, filename):
        self.filename = filename
        f = open(self.filename + '.log', "a")
        f.close()

    def write(self, message):
        f = open(self.filename + '.log', "a")
        f.write(message)
        f.close()


class DATA_LOADER(object):
    def __init__(self, opt):
        if opt.matdataset:
            if opt.dataset == 'imagenet':
                self.read_matimagenet(opt)
            else:
                self.read_matdataset(opt)
        self.index_in_epoch = 0
        self.epochs_completed = 0

        self.task_num = opt.task_num
        self.current_taskid = 0
        self.current_total_class = 0
        self.current_total_class_ = 0
        self.new_label = []
        self.pre_label = []

        self.seen_label = []
        self.seen_label_ = []
        self.unseen_label = []
        self.unseen_label_ = []
        self.split_seen_class()
        self.split_unseen_class()

    def read_matdataset(self, opt):
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat")
        feature = matcontent['features'].T
        self.all_file = matcontent['image_files']
        label = matcontent['labels'].astype(int).squeeze() - 1
        self.label = torch.from_numpy(label)
        print(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat",os.getcwd())

        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + "_splits.mat")
        # numpy array index starts from 0, matlab starts from 1
        train_loc = matcontent['train_loc'].squeeze() - 1
        val_unseen_loc = matcontent['val_loc'].squeeze() - 1

        trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1

        self.attribute = torch.from_numpy(matcontent['att'].T).float()
        print('att:', self.attribute.shape)
        if not opt.validation:
            self.train_image_file = self.all_file[trainval_loc]
            self.test_seen_image_file = self.all_file[test_seen_loc]
            self.test_unseen_image_file = self.all_file[test_unseen_loc]

            if opt.preprocessing:
                if opt.standardization:
                    print('standardization...')
                    scaler = preprocessing.StandardScaler()
                else:
                    scaler = preprocessing.MinMaxScaler()

                _train_feature = scaler.fit_transform(feature[trainval_loc])
                _test_seen_feature = scaler.transform(feature[test_seen_loc])
                _test_unseen_feature = scaler.transform(feature[test_unseen_loc])
                self.train_feature = torch.from_numpy(_train_feature).float()
                mx = self.train_feature.max()
                self.train_feature.mul_(1 / mx)
                self.train_label = torch.from_numpy(label[trainval_loc]).long()
                self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
                self.test_unseen_feature.mul_(1 / mx)
                self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long()
                self.test_seen_feature = torch.from_numpy(_test_seen_feature).float()
                self.test_seen_feature.mul_(1 / mx)
                self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()
            else:
                self.train_feature = torch.from_numpy(feature[trainval_loc]).float()
                self.train_label = torch.from_numpy(label[trainval_loc]).long()
                self.test_unseen_feature = torch.from_numpy(feature[test_unseen_loc]).float()
                self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long()
                self.test_seen_feature = torch.from_numpy(feature[test_seen_loc]).float()
                self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()
        else:
            self.train_feature = torch.from_numpy(feature[train_loc]).float()
            self.train_label = torch.from_numpy(label[train_loc]).long()
            self.test_unseen_feature = torch.from_numpy(feature[val_unseen_loc]).float()
            self.test_unseen_label = torch.from_numpy(label[val_unseen_loc]).long()

        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
        self.ntrain = self.train_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class + self.ntest_class).long()
        self.attribute_seen = self.attribute[self.seenclasses]

        # collect the data of each class

        self.train_samples_class_index = torch.tensor(
            [self.train_label.eq(i_class).sum().float() for i_class in self.train_class])

        # 打乱类别顺序划分task
        if opt.shuffer_class:
            idx = torch.randperm(self.unseenclasses.shape[0])
            self.unseenclasses = self.unseenclasses[idx]
            idx = torch.randperm(self.seenclasses.shape[0])
            self.seenclasses = self.seenclasses[idx]

        # print(self.seenclasses)
        # print(self.unseenclasses)

    def next_batch_one_class(self, batch_size):
        if self.index_in_epoch == self.ntrain_class:
            self.index_in_epoch = 0
            perm = torch.randperm(self.ntrain_class)
            self.train_class[perm] = self.train_class[perm]

        iclass = self.train_class[self.index_in_epoch]
        idx = self.train_label.eq(iclass).nonzero().squeeze()
        perm = torch.randperm(idx.size(0))
        idx = idx[perm]
        iclass_feature = self.train_feature[idx]
        iclass_label = self.train_label[idx]
        self.index_in_epoch += 1
        return iclass_feature[0:batch_size], iclass_label[0:batch_size], self.attribute[iclass_label[0:batch_size]]

    # 划分task
    def split_seen_class(self):
        a = self.seenclasses.shape[0]
        n = self.task_num
        k, m = divmod(a, n)
        # print(self.seenclasses)
        self.splited_seen_class = [self.seenclasses[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] \
                                   for i in list(range(n))]


    def split_unseen_class(self):
        a = self.unseenclasses.shape[0]
        n = self.task_num
        k, m = divmod(a, n)

        self.splited_unseen_class = [self.unseenclasses[i * k + min(i, m):(i + 1) * k + min(i + 1, m)]\
                                   for i in list(range(n))]

    # 当前时间的所有已见过的类
    def current_seen_class(self):
        i = 0
        res = []
        while (i <= self.current_taskid - 1):
            res.extend(self.splited_seen_class[i].numpy())
            i = i + 1
        # print('res is {}'.format(res))
        res = torch.from_numpy(np.array(res)).unsqueeze(dim=1)
        current_data_index = torch.where(self.train_label == res)[1]
        cur_saw_feature = self.train_feature[current_data_index]
        cur_saw_label = self.train_label[current_data_index]
        return cur_saw_feature, cur_saw_label

    def current_seen_class_index(self):
        i = 0
        res = []
        while (i <= self.current_taskid - 1):
            res.extend(self.splited_seen_class[i].numpy())
            i = i + 1
        print('current seen res is {}'.format(res))
        return res
    def current_seen_class_(self):
        res = []
        res.extend(self.splited_seen_class[self.current_taskid - 1].numpy())
        # print('res is {}'.format(res))
        res = torch.from_numpy(np.array(res)).unsqueeze(dim=1)
        current_data_index = torch.where(self.train_label == res)[1]
        cur_saw_feature = self.train_feature[current_data_index]
        cur_saw_label = self.train_label[current_data_index]
        return cur_saw_feature, cur_saw_label
    # 当前时间的所有未见过的类
    def current_unseen_class(self):
        i = 0
        res = []
        while(i <= self.current_taskid - 1):
            res.extend(self.splited_unseen_class[i].numpy())
            i = i + 1
        print('res is {}'.format(res))
        return res
    # 当前task的未见类
    def current_unseen_class_(self):
        res = []
        res.extend(self.splited_unseen_class[self.current_taskid - 1].numpy())
        return res

    # 当前时间的所有已见过的类
    def current_test_seen_class(self):
        i = 0
        res = []
        while (i <= self.current_taskid - 1):
            res.extend(self.splited_seen_class[i].numpy())
            i = i + 1
        # print('res is {}'.format(res))
        res = torch.from_numpy(np.array(res)).unsqueeze(dim=1)
        current_data_index = torch.where(self.test_seen_label == res)[1]

        cur_seen_feature = self.test_seen_feature[current_data_index]
        cur_seen_label = self.test_seen_label[current_data_index]
        # print('test data inform')
        # print(cur_seen_feature.shape, cur_seen_label.shape)
        return cur_seen_feature, cur_seen_label

    def current_test_seen_class_(self):
        res = []
        res.extend(self.splited_seen_class[self.current_taskid - 1].numpy())
        # print('res is {}'.format(res))
        res = torch.from_numpy(np.array(res)).unsqueeze(dim=1)
        current_data_index = torch.where(self.test_seen_label == res)[1]

        cur_seen_feature = self.test_seen_feature[current_data_index]
        cur_seen_label = self.test_seen_label[current_data_index]
        # print('test data inform')
        # print(cur_seen_feature.shape, cur_seen_label.shape)
        return cur_seen_feature, cur_seen_label

    # 当前时间的所有未见过的类
    def current_test_unseen_class(self):
        i = 0
        res = []
        while (i <= self.current_taskid - 1):
            res.extend(self.splited_unseen_class[i].numpy())
            i = i + 1
        res = torch.from_numpy(np.array(res)).unsqueeze(dim=1)
        current_data_index = torch.where(self.test_unseen_label == res)[1]

        cur_unseen_feature = self.test_unseen_feature[current_data_index]
        cur_unseen_label = self.test_unseen_label[current_data_index]

        return cur_unseen_feature, cur_unseen_label

    def current_test_unseen_class_(self):
        res = []
        res.extend(self.splited_unseen_class[self.current_taskid - 1].numpy())
        res = torch.from_numpy(np.array(res)).unsqueeze(dim=1)
        current_data_index = torch.where(self.test_unseen_label == res)[1]

        cur_unseen_feature = self.test_unseen_feature[current_data_index]
        cur_unseen_label = self.test_unseen_label[current_data_index]

        return cur_unseen_feature, cur_unseen_label


    # 获取当前task的类别索引
    def current_class_index(self):
        print('loading task {}th data'.format(self.current_taskid))
        current_data = self.splited_seen_class[self.current_taskid].unsqueeze(dim=1)
        # train seen 7057 test unseen 2967  seen 1764
        # print(len(self.train_label), len(self.test_unseen_label), len(self.test_seen_label))
        # 获取训练集中所有的类别下标作为当前task的子dataset
        self.current_data_index = torch.where(self.train_label == current_data)[1]
        # print(current_data, self.current_data_index)
        # 构造最终用于分类的label
        begin = len(self.new_label)
        end = begin + len(self.splited_seen_class[self.current_taskid])
        temp = [i for i in range(begin, end)]
        self.new_label.extend(temp)
        self.seen_label.extend(temp)
        self.seen_label_ = temp

        begin = len(self.new_label)
        end = begin + len(self.splited_unseen_class[self.current_taskid])
        temp = [i for i in range(begin, end)]
        self.new_label.extend(temp)
        self.unseen_label.extend(temp)
        self.unseen_label_ = temp
        # 构建用于存放原始label的list
        self.pre_label.extend(self.splited_seen_class[self.current_taskid].tolist())
        self.pre_label.extend(self.splited_unseen_class[self.current_taskid].tolist())

        print('current seen_label is {}'.format(self.seen_label))
        print('current unseen_label is {}'.format(self.unseen_label))
        print('pre_label is {}'.format(self.pre_label))

        # 准备下一次task的更新
        self.current_total_class_ = len(self.splited_seen_class[self.current_taskid]) + \
                                   len(self.splited_unseen_class[self.current_taskid])
        print('current_total_class_ for cls is {}'.format(self.current_total_class_))
        self.current_total_class = self.current_total_class + \
                                   len(self.splited_seen_class[self.current_taskid]) + \
                                   len(self.splited_unseen_class[self.current_taskid])

        self.current_taskid = self.current_taskid + 1


    def next_batch_current(self, batch_size):
        lens = np.random.permutation(len(self.current_data_index))
        idx = self.current_data_index[lens][0:batch_size]
        batch_feature = self.train_feature[idx]
        batch_label = self.train_label[idx]
        batch_att = self.attribute[batch_label]
        return batch_feature, batch_label, batch_att



