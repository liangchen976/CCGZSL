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
    id = 1
    os.chdir('../')


    # data = DATA_LOADER(opt)
    # print(f'cur task id is {data.current_taskid}')
    # if data.current_taskid < id:
    #     # 数据的顺序划分
    #     data.current_class_index()
    # 根据task id载入模型
    lifelongGAN.load(id)
    print('已载入模型')
    # 载入测试数据
    name = 'CUB'
    path = 'data/AWA2'
    data = datasets.AwA2(name, path)

    dataset_setup_train = datasets.Dataset_setup(
        data=data.train_set,
        attrs=data.attrs,
        labels=data.train_labels
    )

    dataset_setup_test_seen = datasets.Dataset_setup(
        data=data.test_seen_set,
        attrs=data.attrs,
        labels=data.test_seen_labels
    )
    dataset_setup_test_unseen = datasets.Dataset_setup(
        data=data.test_unseen_set,
        attrs=data.attrs,
        labels=data.test_unseen_labels
    )
    dataset_loader_test_seen = torch.utils.data.DataLoader(dataset_setup_test_seen,
                                                           batch_size=2048, shuffle=True,
                                                           num_workers=4)

    dataset_loader_train_seen = torch.utils.data.DataLoader(dataset_setup_train,
                                                           batch_size=2048, shuffle=True,
                                                           num_workers=4)

    dataset_loader_test_unseen = torch.utils.data.DataLoader(dataset_setup_test_unseen,
                                                             batch_size=1000, shuffle=True,
                                                             num_workers=4)

    for i in range(10):
        for i_batch, sample_batcded in enumerate(dataset_loader_test_seen):
            input_data = sample_batcded['feature']
            input_label = sample_batcded['label']
            input_attr = sample_batcded['attr']

            with torch.no_grad():
                input_data = input_data.to(torch.float32)
                embed, z = lifelongGAN.Map(input_data)
                print(z.shape)
            tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)

            x_tsne = tsne.fit_transform(embed.cpu().detach().numpy())

            # x_norm = x_tsne
            # x_min, x_max = np.min(x_tsne, 0), np.max(x_tsne, 0)
            # x_norm = x_tsne / (x_max -x_min)
            x_tsne = torch.from_numpy(x_tsne)
            x_norm = x_tsne / x_tsne.norm(p=2)
            x_norm = x_norm.numpy()
            fig = plt.figure(figsize=(8, 8))
            colors = list(cnames.values())
            x = x_norm[:, 0].reshape(-1, 1)
            y = x_norm[:, 1].reshape(-1, 1)
            # ax.scatter(x_norm[:, 0], x_norm[:, 1], x_norm[:, 2], c=plt.cm.Set1(np.argmax(input_label, axis=1)))
            # ax.plot_surface(x, y,z, rstride=1, cstride=1)
            # ax.contourf(x, y,z, zdir='z', offset=2)
            for i in range(x_norm.shape[0]):
                if input_label[i][0] < 50:
                    plt.scatter(x[i,:], y[i,:], c=colors[input_label[i][0]])
            plt.xticks([])
            plt.yticks([])
            plt.savefig('./tsne{}_{}_{}.eps'.format(name, f'middle_2d_{i}', i_batch), dpi=400)
            plt.show()


            # colors = list(cnames.values())
            # for i in range(x_norm.shape[0]):
            #     if input_label[i][0] < 100:
            #         plt.scatter(x_norm[i][0], x_norm[i][1], c=colors[input_label[i][0]])
            # plt.xticks([])
            # plt.yticks([])
            # plt.savefig('./tsne{}_{}_{}.png'.format(name, batch, 'bigk', i_batch), dpi=100)
            # plt.show()


