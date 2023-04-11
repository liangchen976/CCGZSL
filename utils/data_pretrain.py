# 载入数据集 并且做到划分
# 预先使用一部分的类别作为训练 剩余的类别进行lifelong学习
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
import copy

def generate_syn_feature(opt, netG, classes, attribute, num):
    # print('current label is {}'.format(classes))
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

def generate_syn_feature_rp(opt, netG, classes, attribute, num):
    # print('current label is {}'.format(classes))
    nclass = len(classes)
    # syn_feature = torch.FloatTensor(nclass * num, opt.resSize)
    # syn_label = torch.LongTensor(nclass * num)
    syn_att = torch.FloatTensor(num, opt.attSize)
    att = torch.FloatTensor(nclass * num, opt.attSize)
    syn_noise = torch.FloatTensor(num, opt.nz)
    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()

    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]
        syn_att.copy_(iclass_att.repeat(num, 1))
        # syn_noise.normal_(0, 1)
        # with torch.no_grad():
        #     output = netG(syn_noise, syn_att)
        # syn_feature.narrow(0, i * num, num).copy_(output.data.cpu())
        att.narrow(0, i * num, num).copy_(syn_att)
        # syn_label.narrow(0, i * num, num).fill_(iclass)
    # return syn_feature, att, syn_label
    return att
def generate_syn_rp(opt, netG, classes, attribute, num):
    print('重放:past task label is {}'.format(classes))
    nclass = len(classes)
    syn_feature = torch.FloatTensor(nclass * num, opt.resSize)
    syn_label = torch.LongTensor(nclass * num)
    syn_att = torch.FloatTensor(num, opt.attSize)
    att = torch.FloatTensor(nclass * num, opt.attSize)
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
        att.narrow(0, i * num, num).copy_(syn_att)
        syn_label.narrow(0, i * num, num).fill_(iclass)

    return syn_feature, syn_label, att
def map_label(label, classes):
	mapped_label = torch.LongTensor(label.size())
	for i in range(classes.size(0)):
		mapped_label[label == classes[i]] = i

	return mapped_label
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
                pass
            else:
                self.read_matdataset(opt)
        self.index_in_epoch = 0
        self.epochs_completed = 0

        self.task_num = opt.task_num
        self.current_taskid = 0
        # self.current_total_class = 0
        self.current_total_class = opt.pretrain_class_number
        # self.current_total_class_ = 0
        if opt.pretrain_gan:
            # 为了在最后实现分类 所以对每个新样本都实时构建新label 是连续整型
            self.new_label = [i for i in range(0,0 + opt.pretrain_class_number)]
        else:
            self.new_label = []
        # pre_label 是原先类别的标签
        self.pre_label = []
        # 截止到目前为止所有已见类别 用新标签表示
        self.seen_label = []
        self.pre_seen_label = []

        self.curr_seen_label = []
        self.curr_unseen_label = []
        self.unseen_label = []
        self.pretrain_nclass = opt.pretrain_class_number
        self.split_seen_class()
        self.split_unseen_class()
        self.pretrain_index()

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

    # 划分task
    def split_seen_class(self):

        self.pretrain_class = self.seenclasses[:self.pretrain_nclass]

        self.lifelong_class = self.seenclasses[self.pretrain_nclass:]
        # 对cub的设置 8+2
        # self.lifelong_class = self.lifelong_class[:140]
        # 对sun的设置 43+24
        self.lifelong_class = self.lifelong_class[:645]
        a = self.lifelong_class.shape[0]
        n = self.task_num
        k, m = divmod(a, n)

        self.splited_seen_class = [self.lifelong_class[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] \
                                   for i in list(range(n))]
        print('splited_seen_class is {}'.format(self.splited_seen_class))

    def split_unseen_class(self):
        # self.unseenclasses = self.unseenclasses[:40]

        a = self.unseenclasses.shape[0]

        n = self.task_num
        k, m = divmod(a, n)
        # 对cub的设置
        # self.unseenclasses = self.unseenclasses[:40]
        # 对sun的设置 43+24
        self.unseenclasses = self.unseenclasses[:60]
        self.splited_unseen_class = [self.unseenclasses[i * k + min(i, m):(i + 1) * k + min(i + 1, m)]\
                                   for i in list(range(n))]
        print('splited_unseen_class is {}'.format(self.splited_unseen_class))

    # 预训练条件gan的类别
    def pretrain_index(self):
        pretrain_data = self.pretrain_class.unsqueeze(dim=1)
        self.pretrain_data_index = torch.where(self.train_label == pretrain_data)[1]
        self.pre_label.extend(self.pretrain_class.tolist())
        self.seen_label = copy.deepcopy(self.new_label)
        self.curr_seen_label = copy.deepcopy(list(self.pretrain_class.numpy()))

        print('new_label is {}'.format(self.new_label))
        print('seen_label is {}'.format(self.seen_label))
        print('pre_label is {}'.format(self.pre_label))
        print('curr_seen_label is {}'.format(self.curr_seen_label))
        print('*'*50)

    # 传入的是类别构成的tensor
    def getData_by_class_in_trainSet(self, classes):
        pretrain_data = classes.unsqueeze(dim=1)
        test_pretrain_data_index = torch.where(self.train_label == pretrain_data)[1]
        seen_feature = self.train_feature[test_pretrain_data_index]
        seen_label = self.train_label[test_pretrain_data_index]
        return seen_feature, seen_label

    def getData_by_class_in_testSet(self, classes):
        pretrain_data = classes.unsqueeze(dim=1)
        test_pretrain_data_index = torch.where(self.test_seen_label == pretrain_data)[1]
        seen_feature = self.test_seen_feature[test_pretrain_data_index]
        seen_label = self.test_seen_label[test_pretrain_data_index]
        return seen_feature, seen_label

    def getData_by_class_in_testUnseenSet(self, classes):
        pretrain_data = classes.unsqueeze(dim=1)
        test_pretrain_data_index = torch.where(self.test_unseen_label == pretrain_data)[1]
        seen_feature = self.test_unseen_feature[test_pretrain_data_index]
        seen_label = self.test_unseen_label[test_pretrain_data_index]
        return seen_feature, seen_label

    # 当前task的类的数据
    def train_current_seen_data(self):
        classes = torch.tensor(self.splited_seen_class[self.current_taskid - 1])
        # print('xxxxxxxx is {}'.format(classes))
        return self.getData_by_class_in_testSet(classes=classes)
    # seen测试集
    def test_seen_data(self):
        # print(f'测试集的seen标签为{self.curr_seen_label}')
        classes = torch.tensor(self.curr_seen_label)
        return self.getData_by_class_in_testSet(classes=classes)
    # unseen测试集
    def test_unseen_data(self):
        # print(f'测试集的unseen标签为{self.curr_unseen_label}')

        classes = torch.tensor(self.curr_unseen_label)
        # classes = torch.tensor(self.unseenclasses)
        # print('curr_unsen_label is {}'.format(self.curr_unseen_label))
        # _, x = self.getData_by_class_in_testUnseenSet(classes=classes)
        # print(x)
        return self.getData_by_class_in_testUnseenSet(classes=classes)


    def test_pretrain_seen_data(self):
        classes = torch.tensor(self.curr_seen_label[:self.pretrain_nclass])
        return self.getData_by_class_in_testSet(classes=classes)



    # 获取当前task的类别索引
    def current_class_index(self):
        print('-'*60)
        print('loading task {}th data'.format(self.current_taskid + 1))
        current_data = self.splited_seen_class[self.current_taskid].unsqueeze(dim=1)
        print(f'current class is {current_data.squeeze()}')
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
        # print('seen_label is {}'.format(self.seen_label))


        begin = len(self.new_label)
        end = begin + len(self.splited_unseen_class[self.current_taskid])
        temp = [i for i in range(begin, end)]
        self.new_label.extend(temp)
        self.unseen_label.extend(temp)
        self.sofar_saw_label = self.curr_seen_label.copy()
        # print('sofar_saw_label1 is {}'.format(self.sofar_saw_label))

        # 构建用于存放原始label的list
        self.pre_label.extend(self.splited_seen_class[self.current_taskid].tolist())
        self.pre_seen_label = self.splited_seen_class[self.current_taskid].tolist()

        self.curr_seen_label.extend(self.splited_seen_class[self.current_taskid].tolist())

        self.pre_label.extend(self.splited_unseen_class[self.current_taskid].tolist())
        self.curr_unseen_label.extend(self.splited_unseen_class[self.current_taskid].tolist())


        print('current seen_label is {}'.format(self.curr_seen_label))
        print('curr_unseen_label is {}'.format(self.curr_unseen_label))
        print('pre_label is {}'.format(self.pre_label))
        print('sofar_saw_label is {}'.format(self.sofar_saw_label))

        print('-'*60)


        # print('new_label is {}'.format(self.new_label))


        # 准备下一次task的更新
        # self.current_total_class_ = len(self.splited_seen_class[self.current_taskid]) + \
        #                            len(self.splited_unseen_class[self.current_taskid])
        # # print('current_total_class_ for cls is {}'.format(self.current_total_class_))
        self.current_total_class = self.current_total_class + \
                                   len(self.splited_seen_class[self.current_taskid]) + \
                                   len(self.splited_unseen_class[self.current_taskid])
        print('current_total_class is {}'.format(self.current_total_class))

        self.current_taskid = self.current_taskid + 1


    def next_batch_current(self, batch_size):
        lens = np.random.permutation(len(self.current_data_index))
        # print('current_data_index is {}'.format(self.current_data_index))
        idx = self.current_data_index[lens][0:batch_size]
        # print('idx is {}'.format(idx))
        batch_feature = self.train_feature[idx]
        batch_label = self.train_label[idx]
        batch_att = self.attribute[batch_label]

        # print(self.seenclasses, torch.unique(batch_label))
        # batch_label = self.seenclasses[batch_label]

        # batch_label = map_label(batch_label, self.seenclasses)

        return batch_feature, batch_label, batch_att

    def current_training_data(self):
        lens = np.random.permutation(len(self.current_data_index))
        # print('current_data_index is {}'.format(self.current_data_index))
        idx = self.current_data_index[lens]
        # print('idx is {}'.format(idx))
        batch_feature = self.train_feature[idx]
        batch_label = self.train_label[idx]
        batch_att = self.attribute[batch_label]
        return batch_feature, batch_label, batch_att


    def next_batch_pretrain(self, batch_size):
        lens = np.random.permutation(len(self.pretrain_data_index))
        idx = self.pretrain_data_index[lens][0:batch_size]
        batch_feature = self.train_feature[idx]
        batch_label = self.train_label[idx]
        batch_att = self.attribute[batch_label]
        return batch_feature, batch_label, batch_att

    def next_batch_replay(self, batch_size, opt, G):
        # 每个类生成一定数量的样本 相当于原文的n_re
        # syn_feature_rp, syn_att_rp, syn_label_rp = generate_syn_feature_rp(opt, G, self.pre_seen_label, \
        #                                               self.attribute, opt.syn_num_rp)
        # print(f'self.pre_seen_label is {self.sofar_saw_label}')
        syn_att_rp = generate_syn_feature_rp(opt, G, self.sofar_saw_label, \
                                                self.attribute, opt.syn_num_rp)
        syn_att_rp = syn_att_rp.cpu()
        lens = np.random.permutation(syn_att_rp.shape[0])
        idx = lens[0:batch_size]
        # print(idx)
        # batch_feature = syn_feature_rp[idx]
        batch_att = syn_att_rp[idx]
        # batch_label = syn_label_rp[idx]
        # print(batch_att.shape)
        # return batch_feature, batch_label, batch_att
        return batch_att


