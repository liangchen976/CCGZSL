import torch.nn as nn
from config import *
import os
import torch
import torch.optim as optim
from sklearn import preprocessing
import numpy as np

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label == classes[i]] = i

    return mapped_label

class D(nn.Module):
    def __init__(self, opt):
        super(D, self).__init__()
        self.fc1 = nn.Linear(opt.resSize + opt.attSize, opt.ndh)
        #self.fc2 = nn.Linear(opt.ndh, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

        self.apply(weights_init)

    def forward(self, x, att):
        h = torch.cat((x, att), 1)
        h = self.lrelu(self.fc1(h))
        h = self.fc2(h)
        return h

class AttDec(nn.Module):
    def __init__(self, opt):
        super(AttDec, self).__init__()
        self.fc1 = nn.Linear(opt.resSize, opt.ngh)
        self.fc3 = nn.Linear(opt.ngh, opt.attSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.hidden = None
        self.sigmoid = None
        self.apply(weights_init)

    def forward(self, feat):
        h = feat
        self.hidden = self.lrelu(self.fc1(h))
        h = self.fc3(self.hidden)

        # h = self.sigmoid(h)
        h = h/h.pow(2).sum(1).sqrt().unsqueeze(1).expand(h.size(0),h.size(1))
        self.out = h
        return h


class E(nn.Module):
    def __init__(self, opt):
        super(E, self).__init__()
        self.fc1 = nn.Linear(opt.resSize, 2 * opt.resSize)
        self.fc2 = nn.Linear(2 * opt.resSize, opt.neh)
        self.fc3 = nn.Linear(opt.neh, opt.nz)

        self.lrelu = nn.LeakyReLU(0.2, True)
        self.dropout = nn.Dropout(p=0.5)
        self.apply(weights_init)


    def forward(self, img):
        out = self.fc1(img)
        # out = self.dropout(out)
        out = self.lrelu(out)
        out = self.fc2(out)
        # out = self.dropout(out)
        out = self.lrelu(out)
        # hs_l2_real = F.normalize(self.fc3(hs), dim=1)
        z = self.fc3(out)
        return z


class G(nn.Module):
    def __init__(self, opt):
        super(G, self).__init__()
        self.fc1 = nn.Linear(opt.nz + opt.attSize, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.resSize)
        # self.fc3 = nn.Linear(2 * opt.resSize, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(p=0.5)
        self.apply(weights_init)

    def forward(self, z, att):
        h = torch.cat((z, att), 1)
        out = self.fc1(h)
        out = self.relu(out)

        out = self.fc2(out)
        out = self.lrelu(out)
        return out

class LifelongGAN(object):
    count = 0
    def __init__(self, opt):
        LifelongGAN.count += 1
        self.G = G(opt)
        self.D = D(opt)
        self.AttDec = AttDec(opt)

        self.G.train()
        self.D.train()
        self.AttDec.train()

    def calc_loss(self, a, z):

        pass

    def save(self, i):
        torch.save(self.G.state_dict(), os.path.join(opt.dir, 'G_task{}.pkl'.format(i)))
        torch.save(self.D.state_dict(), os.path.join(opt.dir, 'D_task{}.pkl'.format(i)))


    def load(self, pre_task_num):
        self.G.load_state_dict(torch.load(os.path.join(opt.dir, 'G_task{}.pkl'.format(pre_task_num))),False)
        self.D.load_state_dict(torch.load(os.path.join(opt.dir, 'D_task{}.pkl'.format(pre_task_num))),False)





class CLASSIFIER:
    # train_Y is interger
    def __init__(self, _train_X, _train_Y, embed_size, data_loader, _nclass, _cuda, _lr=0.001, _beta1=0.5,
                 _nepoch=20, _batch_size=100, generalized=True):
        self.train_X = _train_X
        self.train_Y = _train_Y
        self.test_seen_feature, self.test_seen_label = data_loader.current_test_seen_class()
        self.test_unseen_feature, self.test_unseen_label = data_loader.current_test_unseen_class()
        self.seenclasses = data_loader.seenclasses
        self.unseenclasses = data_loader.unseenclasses
        self.batch_size = _batch_size
        self.nepoch = _nepoch
        self.nclass = _nclass
        self.input_dim = embed_size
        self.cuda = _cuda
        self.model = LINEAR_LOGSOFTMAX(self.input_dim, self.nclass)
        self.model.apply(weights_init)
        self.criterion = nn.NLLLoss()
        # self.criterion = nn.CrossEntropyLoss()
        self.data = data_loader
        self.input = torch.FloatTensor(_batch_size, _train_X.size(1))
        self.label = torch.LongTensor(_batch_size)

        # print('train x shape is {}, train y shape is {}'.format(self.train_X.shape, self.train_Y.shape))
        self.lr = _lr
        self.beta1 = _beta1
        # setup optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=_lr, betas=(_beta1, 0.999))

        if self.cuda:
            self.model.cuda()
            self.criterion.cuda()
            self.input = self.input.cuda()
            self.label = self.label.cuda()

        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.ntrain = self.train_X.size()[0]

        if generalized:
            self.acc_seen, self.acc_unseen, self.H = self.fit()
        else:
            self.acc = self.fit_zsl()

    def fit_zsl(self):
        best_acc = 0
        mean_loss = 0
        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain, self.batch_size):
                self.model.zero_grad()
                batch_input, batch_label = self.next_batch(self.batch_size)
                self.input.copy_(batch_input)

                res = batch_label.unsqueeze(dim=1)
                batch_label = torch.where(torch.tensor(self.data.pre_label) == res)[1]
                self.label.copy_(batch_label)

                output = self.model(self.input)
                loss = self.criterion(output, self.label)
                mean_loss += loss.data
                loss.backward()
                self.optimizer.step()

            acc = self.val(self.test_seen_feature, self.test_seen_label, self.data.seen_label)
            if acc > best_acc:
                best_acc = acc
        print('Training classifier loss= %.4f' % (loss))
        return best_acc

    def fit(self):
        best_H = 0
        best_seen = 0
        best_unseen = 0
        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain, self.batch_size):
                self.model.zero_grad()
                batch_input, batch_label = self.next_batch(self.batch_size)
                # print(batch_input.shape, batch_label.shape)
                # print('data.pre_label is {}'.format(self.data.pre_label))
                # print('batch_label is {}'.format(batch_label))
                res = batch_label.unsqueeze(dim=1)
                # print('res is {}'.format(res))
                # print('data.pre_label is {}'.format(self.data.pre_label))
                batch_label = torch.where(torch.tensor(self.data.pre_label) == res)[1]
                # print('batch_label is {}'.format(batch_label))
                # print('-' * 20)


                self.input.copy_(batch_input)
                self.label.copy_(batch_label)
                output = self.model(self.input)
                # print('-' * 20)
                # print('before label is {}'.format(self.label))
                # le = preprocessing.LabelEncoder()
                # le.fit(self.label)
                # self.label = torch.from_numpy(le.transform(self.label))
                # print('after label is {}'.format(self.label))

                loss = self.criterion(output, self.label)

                loss.backward()
                self.optimizer.step()

            # print(self.model, self.test_seen_feature.device)
            acc_seen = self.val_gzsl(self.test_seen_feature, self.test_seen_label, self.data.seen_label)
            # acc_seen = self.val_gzsl(self.test_seen_feature, self.test_seen_label, self.data.seen_label_)
            acc_unseen = self.val_gzsl(self.test_unseen_feature, self.test_unseen_label, self.data.unseen_label)
            # acc_unseen = self.val_gzsl(self.test_unseen_feature, self.test_unseen_label, self.data.unseen_label_)
            # print('acc_seen is {}, acc_unseen is {}'.format(acc_seen, acc_unseen))


            if (acc_seen + acc_unseen) == 0:
                print('a bug')
                H = 0
            else:
                H = 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen)
            if H > best_H:
                best_seen = acc_seen
                best_unseen = acc_unseen
                best_H = H
        return best_seen, best_unseen, best_H

    def next_batch(self, batch_size):
        start = self.index_in_epoch
        # shuffle the data at the first epoch
        if self.epochs_completed == 0 and start == 0:
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
        # the last batch
        if start + batch_size > self.ntrain:
            self.epochs_completed += 1
            rest_num_examples = self.ntrain - start
            if rest_num_examples > 0:
                X_rest_part = self.train_X[start:self.ntrain]
                Y_rest_part = self.train_Y[start:self.ntrain]
            # shuffle the data
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
            # start next epoch
            start = 0
            self.index_in_epoch = batch_size - rest_num_examples
            end = self.index_in_epoch
            X_new_part = self.train_X[start:end]
            Y_new_part = self.train_Y[start:end]
            # print(start, end)
            if rest_num_examples > 0:
                return torch.cat((X_rest_part, X_new_part), 0), torch.cat((Y_rest_part, Y_new_part), 0)
            else:
                return X_new_part, Y_new_part
        else:
            self.index_in_epoch += batch_size
            end = self.index_in_epoch
            # print(start, end)
            # from index start to index end-1
            return self.train_X[start:end], self.train_Y[start:end]

    def val_gzsl(self, test_X, test_label, target_classes):
        start = 0
        ntest = test_X.size()[0]
        test_X = test_X.cuda()
        # print(test_label)
        # print('ntest is {}'.format(ntest))
        predicted_label = torch.LongTensor(test_label.size())
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start + self.batch_size)
            with torch.no_grad():
                output = self.model(test_X[start:end])
            _, predicted_label[start:end] = torch.max(output, 1)
            start = end
        # test_X = test_X.cuda()
        # output = self.model(test_X)
        # _, predicted_label = torch.max(output, 1)
        # print(predicted_label.shape, test_label.shape, test_label)

        current_data_index = torch.where(torch.from_numpy(
            np.array(self.data.pre_label)) == test_label.unsqueeze(dim=1))[1]
        test_label = current_data_index
        # print('-'*40)
        # print('predicted_label:')
        # print(predicted_label[:40])
        # print('true_label:')
        # print(test_label[:40])
        # print(target_classes[:40])

        acc = self.compute_per_class_acc_gzsl(test_label, predicted_label, target_classes)
        return acc

    def compute_per_class_acc_gzsl(self, test_label, predicted_label, target_classes):
        acc_per_class = 0
        for i in target_classes:
            idx = (test_label == i)
            acc_per_class += float(torch.sum(test_label[idx] == predicted_label[idx])) / float(torch.sum(idx)+1)
            # acc_per_class += float(torch.sum(test_label[idx] == predicted_label[idx])) / float(torch.sum(idx))
        acc_per_class /= len(target_classes)
        return acc_per_class

        # test_label is integer

    def val(self, test_X, test_label, target_classes):
        start = 0
        ntest = test_X.size()[0]
        test_X = test_X.cuda()

        predicted_label = torch.LongTensor(test_label.size())
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start + self.batch_size)
            with torch.no_grad():
                if self.cuda:
                    output = self.model(test_X[start:end])

            _, predicted_label[start:end] = torch.max(output, 1)
            start = end
        current_data_index = torch.where(torch.from_numpy(
            np.array(self.data.pre_label)) == test_label.unsqueeze(dim=1))[1]
        test_label = current_data_index
        acc = self.compute_per_class_acc_gzsl(test_label, predicted_label, target_classes)
        return acc

    def compute_per_class_acc(self, test_label, predicted_label, target_classes):
        acc_per_class = 0
        for i in target_classes:
            idx = (test_label == i)
            acc_per_class += float(torch.sum(test_label[idx] == predicted_label[idx])) / float(torch.sum(idx) + 1)
            # acc_per_class += float(torch.sum(test_label[idx] == predicted_label[idx])) / float(torch.sum(idx))
        acc_per_class /= len(target_classes)
        return acc_per_class
    # 次分类器的训练集是当前task的样本 测试集是已见过所有类的


class LINEAR_LOGSOFTMAX(nn.Module):
    def __init__(self, input_dim, nclass):
        super(LINEAR_LOGSOFTMAX, self).__init__()
        self.fc = nn.Linear(input_dim, nclass)
        self.logic = nn.LogSoftmax(dim=1)

    def forward(self, x):
        o = self.logic(self.fc(x))
        return o

    # test
# if __name__ == '__main__':
#     LifelongGAN = LifelongGAN(opt)
