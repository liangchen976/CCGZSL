import os
import random
import torch
import torch.nn as nn
import torch.optim as optim

import torch.autograd as autograd
from models import *
from config import *
from utils.data import generate_syn_feature
import numpy as np
import copy


# prepare model and dataloader
lifelongGAN = LifelongGAN(opt)
pregan = LifelongGAN(opt)
data = DATA_LOADER(opt)
data.current_class_index()
from torch.autograd import Variable

# prepare data in advance
input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att = torch.FloatTensor(opt.batch_size, opt.attSize)
noise_gen = torch.FloatTensor(opt.batch_size, opt.nz)
input_label = torch.LongTensor(opt.batch_size)

# saved model dir
model_path = './models/' + opt.dataset
if not os.path.exists(model_path):
    os.makedirs(model_path)

def sample():
    batch_feature, batch_label, batch_att = data.next_batch_current(opt.batch_size)
    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)
    input_label.copy_(map_label(batch_label, data.seenclasses))

def calc_gradient_penalty(netD, real_data, fake_data, input_att):
    # print real_data.size()
    alpha = torch.rand(opt.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    if opt.cuda:
        alpha = alpha.cuda()
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    if opt.cuda:
        interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)
    disc_interpolates = netD(interpolates, input_att)
    ones = torch.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.cuda()
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1
    return gradient_penalty
# cuda
if opt.cuda:
    lifelongGAN.G.cuda()
    lifelongGAN.D.cuda()
    lifelongGAN.AttDec.cuda()

    input_res = input_res.cuda()
    noise_gen, input_att = noise_gen.cuda(), input_att.cuda()
    input_label = input_label.cuda()
def WeightedL1(pred, gt):
    wt = (pred-gt).pow(2)
    wt /= wt.sum(1).sqrt().unsqueeze(1).expand(wt.size(0),wt.size(1))
    loss = wt * (pred-gt).abs()
    return loss.sum()/loss.size(0)

def train_one_task(premodel = False):


    optimizerD = optim.Adam(lifelongGAN.D.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(lifelongGAN.G.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    # optimizerDec = optim.Adam(lifelongGAN.AttDec.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    average_H = []

    for epoch in range(opt.nepoch):
        # 首先训练判别器
        for p in lifelongGAN.G.parameters():  # reset requires_grad
            p.requires_grad = False
        for p in lifelongGAN.D.parameters():  # reset requires_grad
            p.requires_grad = True
        # for p in lifelongGAN.AttDec.parameters(): #unfreeze decoder
        #     p.requires_grad = True

        for _ in range(opt.critic_iter):
            sample()
            # recons = lifelongGAN.AttDec(input_res)
            # R_cost = opt.recons_weight * WeightedL1(recons, input_att)
            # R_cost.backward()
            # optimizerDec.step()
            # print(R_cost.data)

            lifelongGAN.D.zero_grad()
            criticD_real = lifelongGAN.D(input_res, input_att)
            criticD_real = criticD_real.mean()

            # train with fakeG
            noise_gen.normal_(0, 1)
            fake = lifelongGAN.G(noise_gen, input_att)
            # 切断梯度传向 GAN的生成器部分
            criticD_fake = lifelongGAN.D(fake.detach(), input_att)
            criticD_fake = criticD_fake.mean()
            # gradient penalty
            gradient_penalty = calc_gradient_penalty(lifelongGAN.D, input_res, fake.data, input_att)
            Wasserstein_D = criticD_real - criticD_fake
            D_cost = criticD_fake - criticD_real + gradient_penalty
            D_cost.backward()
            optimizerD.step()

        # 训练生成器
        for p in lifelongGAN.G.parameters():  # reset requires_grad
            p.requires_grad = True
        for p in lifelongGAN.D.parameters():  # reset requires_grad
            p.requires_grad = False
        for p in lifelongGAN.AttDec.parameters():  # freeze decoder
            p.requires_grad = False
            lifelongGAN.G.zero_grad()
            noise_gen.normal_(0, 1)
            if premodel:

                fake = lifelongGAN.G(noise_gen, input_att)

                # aux_att = lifelongGAN.AttDec(fake)

                # fake_pre = premodel.G(noise_gen, aux_att)
                fake_pre = premodel.G(noise_gen, input_att)
                # fake_aux = lifelongGAN.G(noise_gen, aux_att)

                kl_loss = nn.L1Loss()
                # loss_dl_g = kl_loss(fake_aux, fake_pre)
                loss_dl_g = kl_loss(fake, fake_pre)

                criticG_fake = lifelongGAN.D(fake, input_att)
                criticG_fake = criticG_fake.mean()
                G_cost = -criticG_fake
                errG = G_cost + opt.beta * loss_dl_g
                if epoch % 500 == 0:
                    print('kl_loss is {}'.format(loss_dl_g))
                errG.backward()
                optimizerG.step()
            else:
                fake = lifelongGAN.G(noise_gen, input_att)
                criticG_fake = lifelongGAN.D(fake, input_att)
                criticG_fake = criticG_fake.mean()
                G_cost = -criticG_fake
                errG = G_cost
                errG.backward()
                optimizerG.step()
        if (epoch + 1) % 200 == 0:
            print('[%d/%d] Loss_D: %.4f Loss_G: %.4f, Wasserstein_dist: %.4f' \
          % (epoch, opt.nepoch, D_cost, G_cost, Wasserstein_D))
        if (epoch + 1) % 200 == 0:
            lifelongGAN.save(data.current_taskid)
            # frezze
            lifelongGAN.G.eval()

            # syn_feature, syn_label = generate_syn_feature(opt, lifelongGAN.G, data.current_unseen_class(),\
            #                                                   data.attribute, opt.syn_num)
            # syn_feature, syn_label = generate_syn_feature(opt, lifelongGAN.G, data.current_unseen_class_(), \
            #                                               data.attribute, opt.syn_num)

            syn_feature_, syn_label_ = generate_syn_feature(opt, lifelongGAN.G, data.current_seen_class_index(), \
                                                          data.attribute, opt.syn_num_)
            # res = torch.from_numpy(np.array(syn_label)).unsqueeze(dim=1)
            # index_temp = torch.where(torch.tensor(data.pre_label) == res)[1]
            # syn_label = index_temp
            # print(index_temp, data.new_label)
            # syn_label = data.new_label[index_temp]
            # print(syn_label)

            # img, label = data.current_seen_class()


            # train_X = torch.cat((img, syn_feature), 0)
            # train_Y = torch.cat((label, syn_label), 0)

            train_X = syn_feature_
            train_Y = syn_label_

            # train_X = torch.cat((syn_feature_, syn_feature), 0)
            # train_Y = torch.cat((syn_label_, syn_label), 0)

            nclass = data.current_total_class
            # nclass = data.current_total_class_
            print('cls traing')
            cls = CLASSIFIER(_train_X = train_X, _train_Y = train_Y, embed_size=opt.resSize, data_loader=data,
                             _nclass=nclass, _cuda=opt.cuda,_lr=opt.classifier_lr, _beta1=0.5, _nepoch=25,
                             _batch_size=opt.syn_num, generalized=False)
            # print('unseen=%.4f, seen=%.4f, h=%.4f' % (cls.acc_unseen, cls.acc_seen, cls.H))
            print('seen=%.4f' % (cls.acc))
            # average_H.append(cls.H)
            average_H.append(cls.acc)
            lifelongGAN.G.train()

    # print('average_H is {}'.format(np.mean(average_H[1:])))
    print('average_acc is {}'.format(np.mean(average_H[1:])))
    print('-' * 40)
    # next dataloader
    data.current_class_index()




if __name__ == '__main__':
    for i in range(opt.task_num):
        if data.current_taskid == 1:
            print('training from scratch ···')
            train_one_task()
        else:
            print('incremental training task {}···'.format(data.current_taskid - 1))
            # pre = LifelongGAN(opt)
            # pre.load(data.current_taskid - 1)
            # pregan = copy.deepcopy(lifelongGAN)
            pregan = LifelongGAN(opt)
            print('loading task {}'.format(data.current_taskid - 1))
            pregan.load(data.current_taskid - 1)
            if opt.cuda:
                pregan.D.cuda()
                pregan.G.cuda()
            for p in pregan.G.parameters():  # reset requires_grad
                p.requires_grad = False
            # train_one_task(premodel = pregan)
            train_one_task()
        # opt.syn_num_ = opt.syn_num_ * 2




