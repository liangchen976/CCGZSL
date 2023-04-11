import torch
import numpy as np
import argparse
import os
import yaml
from shutil import copyfile
import torch.nn as nn
from models_pre import *
from config import opt
from sklearn import manifold
import matplotlib.pyplot as plt
from visualize.colors import cnames
from mpl_toolkits.mplot3d import Axes3D
from utils.data_pretrain import DATA_LOADER
import data_utils.datasets as datasets

# 数据占位
input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att = torch.FloatTensor(opt.batch_size, opt.attSize)
noise_gen = torch.FloatTensor(opt.batch_size, opt.nz)
input_label = torch.LongTensor(opt.batch_size)

# 当前task的训练样本采样
def sample():
    batch_feature, batch_label, batch_att = data.next_batch_current(opt.batch_size)
    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)
    input_label.copy_(map_label(batch_label, data.seenclasses))



if __name__ == '__main__':
    lifelongGAN = LifelongGAN(opt)
    id = 4
    os.chdir('../')


    data_lifelong = DATA_LOADER(opt)
    print(f'cur task id is {data_lifelong.current_taskid}')
    while data_lifelong.current_taskid < id:
        # 数据的顺序划分
        data_lifelong.current_class_index()
    # 根据task id载入模型
    # print(f'current seen label is {data_lifelong.pre_label, len(data_lifelong.pre_label)}')
    lifelongGAN.load(id)
    print('已载入模型')
    # print(f'模型在{lifelongGAN.device}设备上')
    thresh, anchors = data_lifelong.search_thres_by_traindata(opt, lifelongGAN, 200, 0.9, flag=True)
    # thresh = 0.998
    print(f'threshold is {thresh}')
    # anchors = data_lifelong.get_anchors(opt, lifelongGAN)
    # print(anchors)
    # 载入测试数据
    name = 'CUB' # 名字没啥用
    path = 'data/AWA1' #数据集的载入只看路径名
    data = datasets.AwA2(name, path)

    dataset_setup_train = datasets.Dataset_setup(
        data=data.train_set,
        attrs=data.attrs,
        labels=data.train_labels
    )
    dataset_loader_train_seen = torch.utils.data.DataLoader(dataset_setup_train,
                                                            batch_size=512, shuffle=True,
                                                            num_workers=4)

    dataset_setup_test_seen = datasets.Dataset_setup(
        data=data.test_seen_set,
        attrs=data.attrs,
        labels=data.test_seen_labels
    )
    dataset_loader_test_seen = torch.utils.data.DataLoader(dataset_setup_test_seen,
                                                           batch_size=512, shuffle=True,
                                                           num_workers=4)
    #
    dataset_setup_test_unseen = datasets.Dataset_setup(
        data=data.test_unseen_set,
        attrs=data.attrs,
        labels=data.test_unseen_labels
    )
    dataset_loader_test_unseen = torch.utils.data.DataLoader(dataset_setup_test_unseen,
                                                             batch_size=512, shuffle=True,
                                                             num_workers=4)

    for i in range(10):
        for i_batch, sample_batcded in enumerate(dataset_loader_test_seen):
            input_data = sample_batcded['feature']
            input_label = sample_batcded['label']
            input_attr = sample_batcded['attr']

            p = 0
            n = 0
            with torch.no_grad():
                input_data = input_data.to(torch.float32)
                embed, z = lifelongGAN.Map(input_data)
                z = embed
                temp = []
                for item in z:
                    item = torch.unsqueeze(item,dim=0)
                    # print(item.shape, anchors.shape)
                    score = F.cosine_similarity(item, anchors, dim=1)
                    biggest = torch.max(score)
                    # sorted, indict = torch.sort(score)
                    # print(sorted)
                    temp.append(biggest.data)
                    if biggest > thresh:
                        p+=1
                    else:
                        n+=1
                print(f'pos_acc = {p/input_data.shape[0]}, neg_acc = {n/input_data.shape[0]}')
                temp.sort(reverse=True)
                print(temp)


            # tsne = manifold.TSNE(n_components=3)
            # x_tsne = tsne.fit_transform(embed.cpu().detach().numpy())
            # x_tsne = torch.from_numpy(x_tsne)
            # x_norm = x_tsne / x_tsne.norm(p=2)
            # x_norm = x_norm.numpy()
            # fig = plt.figure(figsize=(8, 8))
            # colors = list(cnames.values())
            # ax = Axes3D(fig)
            # ax1 = plt.axes(projection='3d')
            # x = x_norm[:, 0].reshape(-1, 1)
            # y = x_norm[:, 1].reshape(-1, 1)
            # z = x_norm[:, 2].reshape(-1, 1)
            # for i in range(x_norm.shape[0]):
            #     if input_label[i][0] < 50:
            #         ax1.scatter3D(x[i,:], y[i,:], z[i,:], c=colors[input_label[i][0]])
            # plt.xticks([])
            # plt.yticks([])
            # # plt.savefig('./tsne{}_{}_{}.png'.format(name, batch, 'bigk', i_batch), dpi=100)
            # plt.show()



