import os
import random
import torch
import torch.nn as nn
import torch.optim as optim

import torch.autograd as autograd
from models_pre import *
from config import opt
from utils.data_pretrain import DATA_LOADER
from utils.data_pretrain import generate_syn_feature
from utils.data_pretrain import generate_syn_rp

import numpy as np
import losses
import copy
import itertools
contras_criterion = losses.SupConLoss_clear(opt.ins_temp)

# prepare model and dataloader
lifelongGAN = LifelongGAN(opt)
pregan = LifelongGAN(opt)
data = DATA_LOADER(opt)
# data.current_class_index()
from torch.autograd import Variable

# prepare data in advance
input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att = torch.FloatTensor(opt.batch_size, opt.attSize)
noise_gen = torch.FloatTensor(opt.batch_size, opt.nz)
input_label = torch.LongTensor(opt.batch_size)

# saved model dir
model_path = './pre_models/' + opt.dataset
if not os.path.exists(model_path):
    os.makedirs(model_path)

def sample():
    batch_feature, batch_label, batch_att = data.next_batch_current(opt.batch_size)
    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)
    # input_label.copy_(map_label(batch_label, data.seenclasses))
    input_label.copy_(batch_label)
    # print(f'training label is {torch.unique(input_label)}')



def synPreData(premodel):
    syn_feature_pre, syn_label, syn_att_pre = generate_syn_rp(opt, premodel, data.sofar_saw_label, \
                                                      data.attribute, opt.syn_num_replay)
    print(f'replay label is {torch.unique(syn_label), syn_feature_pre.shape}')

    batch_feature, batch_label, batch_att = data.current_training_data()
    print(f'cur task label is {torch.unique(batch_label), batch_feature.shape}')

    # feature_fusion = torch.cat((syn_feature_pre, batch_feature), 0)
    # att_fusion = torch.cat((syn_att_pre, batch_att), 0)
    # label_fusion = torch.cat((syn_label, batch_label), 0)

    feature_fusion = torch.cat((batch_feature, syn_feature_pre), 0)
    att_fusion = torch.cat((batch_att, syn_att_pre), 0)
    label_fusion = torch.cat((batch_label, syn_label), 0)
    return feature_fusion, label_fusion, att_fusion

def sample_pre_replay(pregan):
    batch_att = data.next_batch_replay(opt.batch_size, opt, pregan)
    # input_res.copy_(batch_feature)
    input_att.copy_(batch_att)
    # input_label.copy_(map_label(batch_label, data.seenclasses))

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
def sample_replay(fusiondata=None):
    if not fusiondata:
        print("没有处理好的融合数据，请重新融合")
    else:
        feature_fusion, label_fusion, att_fusion = fusiondata
        # print('feature_fusion shape is {}, label_fusion shape is {},'
        #       'att_fusion shape is {}'.format(feature_fusion.shape, label_fusion.shape,
        #                                       att_fusion.shape))
        lens = feature_fusion.shape[0]
        # print('have {}'.format(lens))
        idx = random.sample(range(0,lens), opt.batch_size)
        # print(idx)
        input_res.copy_(feature_fusion[idx])
        input_label.copy_(label_fusion[idx])
        input_att.copy_(att_fusion[idx])
# cuda
if opt.cuda:
    lifelongGAN.G.cuda()
    lifelongGAN.D.cuda()
    # lifelongGAN.AttDec.cuda()
    lifelongGAN.P.cuda()
    lifelongGAN.Map.cuda()

    input_res = input_res.cuda()
    noise_gen, input_att = noise_gen.cuda(), input_att.cuda()
    input_label = input_label.cuda()

entory_loss = nn.CrossEntropyLoss()


def pre_train():
    optimizerD = optim.Adam(itertools.chain(lifelongGAN.D.parameters(),
                                            lifelongGAN.Map.parameters()
                             ), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(lifelongGAN.G.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    average_H = []
    preh = 0
    bu, bs, bh = 0, 0, 0
    for epoch in range(opt.nepoch):
        # 首先训练判别器
        for p in lifelongGAN.G.parameters():  # reset requires_grad
            p.requires_grad = False
        for p in lifelongGAN.D.parameters():  # reset requires_grad
            p.requires_grad = True
        for p in lifelongGAN.Map.parameters():  # reset requires_grad
            p.requires_grad = True
        for _ in range(opt.critic_iter):
            sample()
            lifelongGAN.D.zero_grad()
            lifelongGAN.Map.zero_grad()

            embed_real, outz_real = lifelongGAN.Map(input_res)

            criticD_real = lifelongGAN.D(input_res, input_att)
            criticD_real = criticD_real.mean()

            # CONTRASITVE LOSS
            real_ins_contras_loss = contras_criterion(outz_real, input_label)

            # train with fakeG
            noise_gen.normal_(0, 1)
            fake = lifelongGAN.G(noise_gen, input_att)
            # 切断梯度传向 GAN的生成器部分
            criticD_fake = lifelongGAN.D(fake.detach(), input_att)
            criticD_fake = criticD_fake.mean()
            # gradient penalty
            gradient_penalty = calc_gradient_penalty(lifelongGAN.D, input_res, fake.data, input_att)
            Wasserstein_D = criticD_real - criticD_fake
            D_cost = criticD_fake - criticD_real + gradient_penalty + real_ins_contras_loss
            D_cost.backward()
            optimizerD.step()
        # 训练生成器
        for p in lifelongGAN.G.parameters():  # reset requires_grad
            p.requires_grad = True
        for p in lifelongGAN.D.parameters():  # reset requires_grad
            p.requires_grad = False
        for p in lifelongGAN.Map.parameters():  # reset requires_grad
            p.requires_grad = False

            lifelongGAN.G.zero_grad()
            noise_gen.normal_(0, 1)
            fake = lifelongGAN.G(noise_gen, input_att)

            embed_fake, outz_fake = lifelongGAN.Map(fake)

            criticG_fake = lifelongGAN.D(fake, input_att)
            criticG_fake = criticG_fake.mean()
            G_cost = -criticG_fake


            embed_real, outz_real = lifelongGAN.Map(input_res)
            # cls loss
            # pseudo_classification_loss = entory_loss(embed_fake, input_label.squeeze())

            all_outz = torch.cat((outz_fake, outz_real.detach()), dim=0)

            fake_ins_contras_loss = contras_criterion(all_outz, torch.cat((input_label, input_label), dim=0))

            errG = G_cost + opt.ins_weight * fake_ins_contras_loss \
                   # + 0.01 * pseudo_classification_loss

            errG.backward()
            optimizerG.step()
        if epoch % 50 == 0:
            print('[%d/%d] Loss_D: %.4f Loss_G: %.4f, Wasserstein_dist: %.4f, real_ins_contras_loss: %.4f,'
                  'fake_ins_contras_loss : %.4f'
                  % (epoch, opt.nepoch, D_cost, G_cost, Wasserstein_D,
                     real_ins_contras_loss, fake_ins_contras_loss))
        if (epoch + 1) % 200 == 0:
            # lifelongGAN.save(data.current_taskid)
            lifelongGAN.G.eval()
            syn_feature_s, syn_label_s = generate_syn_feature(opt, lifelongGAN.G, data.curr_seen_label,data.attribute, opt.syn_num_s)

            syn_feature_u, syn_label_u = generate_syn_feature(opt, lifelongGAN.G, data.curr_unseen_label, data.attribute, opt.syn_num_u)

            train_X = torch.cat((syn_feature_s, syn_feature_u), 0)
            train_Y = torch.cat((syn_label_s, syn_label_u), 0)
            # nclass = len(data.lifelong_class)
            nclass = data.current_total_class

            print('cls traing')
            cls = CLASSIFIER(_train_X = train_X, _train_Y = train_Y, embed_size=opt.resSize, data_loader=data,
                             _nclass=nclass, _cuda=opt.cuda,_lr=opt.classifier_lr, _beta1=0.5, _nepoch=25,
                             _batch_size=opt.syn_num, generalized=True, map_net=lifelongGAN.Map)
            print('unseen=%.4f, seen=%.4f, h=%.4f' % (cls.acc_unseen, cls.acc_seen, cls.H))
            u, s, h = cls.acc_unseen, cls.acc_seen, cls.H
            if h > bh:
                bs = s
                bu = u
                bh = h
            average_H.append(cls.H)
            if preh < h:
                preh = h
                print('current model is better than the past one, so save it.')
                lifelongGAN.save(data.current_taskid)
            # print('seen=%.4f' % (cls.acc))
            # average_H.append(cls.H)
            # average_H.append(cls.acc)
            lifelongGAN.G.train()
    print('average_acc is {}'.format(np.mean(average_H[1:])))
    print(' hightest s is {}, u is {}, H is {}, '.format(bs, bu, bh))
    print('-' * 40)
    # next dataloader
    # data.current_class_index()


def train_one_task(premodel = False):
    optimizerD = optim.Adam(itertools.chain(lifelongGAN.D.parameters(),
                                            lifelongGAN.Map.parameters(),
                                            lifelongGAN.P.parameters()
                                            ), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(lifelongGAN.G.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    # optimizerP = optim.Adam(lifelongGAN.P.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    fusiondata = synPreData(premodel.G)

    average_H = []
    preh = 0
    bu, bs, bh = 0, 0, 0
    # opt.beta = opt.beta * 1.1
    for epoch in range(opt.nepoch):
        # 首先训练判别器 和 map predictor
        for p in lifelongGAN.G.parameters():  # reset requires_grad
            p.requires_grad = False
        for p in lifelongGAN.D.parameters():  # reset requires_grad
            p.requires_grad = True
        for p in lifelongGAN.P.parameters():  # reset requires_grad
            p.requires_grad = True
        for p in lifelongGAN.Map.parameters():  # reset requires_grad
            p.requires_grad = True

        for _ in range(opt.critic_iter):
            # sample()
            sample_replay(fusiondata)
            # print(f'training label is {torch.unique(input_label)}')
            lifelongGAN.D.zero_grad()
            lifelongGAN.P.zero_grad()
            lifelongGAN.Map.zero_grad()

            embed_real, outz_real = lifelongGAN.Map(input_res)


            criticD_real = lifelongGAN.D(input_res, input_att)
            criticD_real = criticD_real.mean()

            real_ins_contras_loss = contras_criterion(outz_real, input_label)

            # train with fakeG
            noise_gen.normal_(0, 1)
            fake = lifelongGAN.G(noise_gen, input_att)
            # 切断梯度传向 GAN的生成器部分
            criticD_fake = lifelongGAN.D(fake.detach(), input_att)
            criticD_fake = criticD_fake.mean()
            # gradient penalty
            gradient_penalty = calc_gradient_penalty(lifelongGAN.D, input_res, fake.data, input_att)
            Wasserstein_D = criticD_real - criticD_fake

            D_cost = criticD_fake - criticD_real + gradient_penalty + real_ins_contras_loss

            if premodel:
                kl_loss = nn.L1Loss()
                _, pre_z = premodel.Map(input_res)
                _, fake_z = lifelongGAN.Map(input_res)
                predict_z = lifelongGAN.P(fake_z)
                CL_contra_loss = kl_loss(predict_z, pre_z.detach())
                D_cost = D_cost + opt.beta_1 * CL_contra_loss

            D_cost.backward()
            optimizerD.step()

        # 训练生成器
        for p in lifelongGAN.G.parameters():  # reset requires_grad
            p.requires_grad = True
        for p in lifelongGAN.D.parameters():  # reset requires_grad
            p.requires_grad = False
        for p in lifelongGAN.Map.parameters():  # reset requires_grad
            p.requires_grad = False
        for p in lifelongGAN.P.parameters():  # reset requires_grad
            p.requires_grad = False

            lifelongGAN.G.zero_grad()

            noise_gen.normal_(0, 1)
            fake = lifelongGAN.G(noise_gen, input_att)

            embed_fake, outz_fake = lifelongGAN.Map(fake)


            criticG_fake = lifelongGAN.D(fake, input_att)
            criticG_fake = criticG_fake.mean()

            embed_real, outz_real = lifelongGAN.Map(input_res)
            # cls loss
            all_outz = torch.cat((outz_fake, outz_real.detach()), dim=0)
            fake_ins_contras_loss = contras_criterion(all_outz, torch.cat((input_label, input_label), dim=0))
            real_ins_contras_loss = contras_criterion(outz_real, input_label)
            G_cost = -criticG_fake
            # pseudo_classification_loss = entory_loss(embed_fake, input_label.squeeze())

            errG = G_cost + opt.ins_weight * fake_ins_contras_loss \
                   # + 0.01 * pseudo_classification_loss

            if premodel:
                # 不确定是否加
                # _, pre_z = premodel.Map(input_res)
                # _, fake_z = lifelongGAN.Map(input_res)
                # predict_z = lifelongGAN.P(fake_z)
                # CL_contra_loss = kl_loss(predict_z, pre_z.detach())

                sample_pre_replay(premodel.G)
                fake = lifelongGAN.G(noise_gen, input_att)
                fake_pre = premodel.G(noise_gen, input_att)
                loss_dl_g = kl_loss(fake, fake_pre.detach())
                # print('fake device is on {}'.format(fake.device))
                # kl_loss = nn.MSELoss()
                # loss_dl_g = kl_loss(pred, fake_pre)

                errG = errG + opt.beta * loss_dl_g \
                       # + opt.beta_1 * CL_contra_loss
                       # + loss_dl_pre
                if epoch % 500 == 0:
                    print('kl_loss G is {}'.format(loss_dl_g))
                    # print('kl_loss is {}, loss_dl_pre is {}'.format(loss_dl_g, loss_dl_pre))

            errG.backward()
            optimizerG.step()

        if epoch % 100 == 0:
            print('[%d/%d] Loss_D: %.4f Loss_G: %.4f, Wasserstein_dist: %.4f, real_ins_contras_loss: %.4f,'
                  'fake_ins_contras_loss : %.4f, CL_contra_loss : %.4f'
                  % (epoch, opt.nepoch, D_cost, G_cost, Wasserstein_D,
                     real_ins_contras_loss, fake_ins_contras_loss, CL_contra_loss))
        if epoch % 200 == 0:
            # if  preh < h:
            # lifelongGAN.save(data.current_taskid)
            # freeze
            lifelongGAN.G.eval()
            for p in lifelongGAN.Map.parameters():  # reset requires_grad
                p.requires_grad = False
            sep = True
            syn_feature_s, syn_label_s = generate_syn_feature(opt, lifelongGAN.G, data.curr_seen_label,\
                                                              data.attribute, opt.syn_num_s)
            syn_feature_u, syn_label_u = generate_syn_feature(opt, lifelongGAN.G, data.curr_unseen_label, \
                                                          data.attribute, opt.syn_num_u)


            train_X = torch.cat((syn_feature_s, syn_feature_u), 0)
            train_Y = torch.cat((syn_label_s, syn_label_u), 0)

            # print(syn_label_s.shape, syn_label_s[:100])
            # print(syn_label_u.shape, syn_label_u[:500])

            # train_X = syn_feature
            # train_Y = syn_label

            # train_X = torch.cat((syn_feature_, syn_feature), 0)
            # train_Y = torch.cat((syn_label_, syn_label), 0)

            nclass = data.current_total_class
            print(f'nclass is {nclass}')
            # nclass = data.current_total_class_
            print('cls traing')
            # print(' now {} classes'.format(nclass))
            if sep:
                cls = CLASSIFIER(_train_X = train_X, _train_Y = train_Y, embed_size=opt.resSize, data_loader=data,
                             _nclass=nclass, _cuda=opt.cuda,_lr=opt.classifier_lr, _beta1=0.5, _nepoch=25,
                             _batch_size=opt.syn_num, generalized=True, map_net=lifelongGAN.Map)
                print('unseen=%.4f, seen=%.4f, h=%.4f' % (cls.acc_unseen, cls.acc_seen, cls.H))
                u, s, h = cls.acc_unseen, cls.acc_seen, cls.H
                if h > bh:
                    bs = s
                    bu = u
                    bh = h
                average_H.append(cls.H)
                if preh < h:
                    preh = h
                    print('current model is better than the past one, so save it.')
                    lifelongGAN.save(data.current_taskid)

            else:
                #
                cls_s = CLASSIFIER(_train_X=syn_feature_s, _train_Y=syn_label_s, embed_size=opt.resSize, data_loader=data,
                                 _nclass=nclass, _cuda=opt.cuda, _lr=opt.classifier_lr, _beta1=0.5, _nepoch=25,
                                 _batch_size=opt.syn_num, generalized=False, map_net=None)

                cls_u = CLASSIFIER(_train_X=syn_feature_u, _train_Y=syn_label_u, embed_size=opt.resSize,
                                   data_loader=data,
                                   _nclass=nclass, _cuda=opt.cuda, _lr=opt.classifier_lr, _beta1=0.5, _nepoch=25,
                                   _batch_size=opt.syn_num, generalized=False, map_net=lifelongGAN.Map, sss=False)
                print('unseen=%.4f, seen=%.4f, h=%.4f' % (cls_u.acc, cls_s.acc, 2*(cls_u.acc*cls_s.acc)/(cls_u.acc+cls_s.acc)))
                lifelongGAN.save(data.current_taskid)

            # print('seen=%.4f' % (cls.acc))

            # average_H.append(cls.acc)
            lifelongGAN.G.train()

    # print('average_H is {}'.format(np.mean(average_H[1:])))
    print('average_acc is {}'.format(np.mean(average_H[1:])))
    print(' hightest s is {}, u is {}, H is {}, '.format(bs, bu, bh))
    print('-' * 40)
    # next dataloader
    data.current_class_index()


if __name__ == '__main__':
    for i in range(opt.task_num + 1):
        if data.current_taskid == 0:
            print('pretraining cgan model ···')
            data.current_class_index()
            # pre_train()
            data.current_class_index()

            # print(f'cur task id is {data.current_taskid}')
            # lifelongGAN.load(data.current_taskid-1)

            # print('finish pretraining cgan model ···')
        else:
            print('incremental training task {}···'.format(data.current_taskid))
            pregan = LifelongGAN(opt)
            print('loading task {} data and the pre model'.format(data.current_taskid-1))
            pregan.load(data.current_taskid-1)
            # opt.syn_num_u = int(opt.syn_num_u * 1.4)
            if opt.cuda:
                # pregan.D.cuda()
                pregan.G.cuda()
                pregan.Map.cuda()
            for p in pregan.G.parameters():  # reset requires_grad
                p.requires_grad = False
            for p in pregan.Map.parameters():  # reset requires_grad
                p.requires_grad = False
            train_one_task(pregan)
            # train_one_task()




