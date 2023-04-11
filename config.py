import argparse
import sys
from utils.data import DATA_LOADER
sys.path.append("..")

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')

# dataset
parser.add_argument('--dataroot', default='data', help='path to dataset')
parser.add_argument('--matdataset', default=True, help='Data in matlab format')
parser.add_argument('--dataset', default='AWA1', help='AWA1,AWA2,SUN,CUB')
parser.add_argument('--image_embedding', default='res101')

# parser.add_argument('--class_embedding', default='sent', help='att or sent')
parser.add_argument('--class_embedding', default='att', help='att or sent')

parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
parser.add_argument('--preprocessing', type=bool, default=True, help='enbale MinMaxScaler on visual features')
parser.add_argument('--standardization', action='store_true', default=False)
parser.add_argument('--syn_num_s', type=int, default=2000, help='number features to generate per class')
parser.add_argument('--syn_num_u', type=int, default=1000, help='number features to generate per class')
parser.add_argument('--syn_num', type=int, default=256, help='number features to generate per class')


parser.add_argument('--nclass_all', type=int, default=200, help='number of all classes')
parser.add_argument('--nclass_seen', type=int, default=150, help='number of all classes')

# para in NN
parser.add_argument('--nepoch', type=int, default=2000, help='number of epochs to train for')

parser.add_argument('--nz', type=int, default=256, help='dimension of noise for generation')

parser.add_argument('--neh', type=int, default=4096, help='size of the hidden units in encoder E')
parser.add_argument('--ngh', type=int, default=4096, help='size of the hidden units in generator G')
parser.add_argument('--ndh', type=int, default=4096, help='size of the hidden units in discriminator D')
parser.add_argument('--distill_proj_hidden_dim', type=int, default=4096,
                    help='size of the distill_proj_hidden_dim in predictor P')

parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--beta', type=float, default=1, help='weight for distillation loss of G')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate to training')
parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
parser.add_argument('--recons_weight', type=float, default=0.001, help='weight for AttDec')

parser.add_argument('--ins_temp', type=float, default=0.1, help='temperature in instance-level supervision')
parser.add_argument('--ins_weight', type=float, default=0.001, help='weight of the classification loss when learning G')


parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--resSize', type=int, default=2048, help='size of visual features')
parser.add_argument('--attSize', type=int, default=1024, help='size of semantic features')
parser.add_argument('--embedSize', type=int, default=2048, help='size of embedding h')
parser.add_argument('--outzSize', type=int, default=512, help='size of non-liner projection z')


# continue learning para
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--task_num", default=10, type=int)
parser.add_argument("--pretrain_class_number", default=20, type=int, help = 'pretrain cgan in pretrain_class_number')
parser.add_argument('--pretrain_gan', default=True, help='shuffering the class order or not')
parser.add_argument('--shuffer_class', default=False, help='shuffering the class order or not')

# model
parser.add_argument('--dir', default='.\models\CUB',help='dir of model')

opt = parser.parse_args()

if opt.dataset == 'CUB':
    opt.resSize = 2048
    opt.nclass_all = 200
    opt.dir = 'models\CUB'
    opt.nclass_seen = 150
    opt.lr = 1e-4
    opt.manualSeed = 2424
    opt.batch_size = 200
    opt.nepoch = 3001
    opt.beta = 100
    opt.critic_iter = 5
    opt.syn_num_u = 200
    opt.syn_num_s = 100
    opt.syn_num = 100
    opt.syn_num_replay = 20
    opt.ins_weight = 0.001
    opt.beta_1 = 10
    opt.syn_num_rp = 100
    opt.task_num = 5
    opt.ins_temp = 0.1
    opt.outzSize = 256

    opt.pretrain_class_number = 0
    opt.shuffer_class = False
    opt.cuda = True
    if opt.class_embedding == 'att':
        opt.nz = 312
        opt.attSize = 312
    else:# sent
        opt.nz = 1024
        opt.attSize = 1024
elif opt.dataset == 'AWA1':
    opt.dir = 'models\AWA1'
    opt.resSize = 2048
    opt.nclass_all = 50
    opt.nclass_seen = 40
    opt.manualSeed = 2424
    # opt.manualSeed = 3407
    opt.batch_size = 512
    opt.nepoch = 200001
    opt.syn_num_u = 2800
    opt.syn_num_s = 600
    opt.syn_num_rp = 100
    opt.syn_num_replay = 100
    opt.outzSize = 256
    opt.attSize = 85
    opt.task_num = 5
    opt.pretrain_class_number = 0
    opt.beta = 100
    opt.beta_1 = 0.01
    opt.ins_weight = 1
    opt.ins_temp = 1
    opt.critic_iter = 5
    opt.lr = 1e-4
    opt.shuffer_class = False
elif opt.dataset == 'AWA2':
    opt.dir = 'models\AWA2'
    opt.resSize = 2048
    opt.nclass_all = 50
    opt.nclass_seen = 40
    opt.manualSeed = 2424
    # opt.manualSeed = 3407
    opt.batch_size = 512
    opt.nepoch = 2001
    opt.syn_num_u = 2800
    opt.syn_num_s = 600
    opt.syn_num_rp = 100
    opt.syn_num_replay = 100
    opt.outzSize = 256
    opt.attSize = 85
    opt.task_num = 5
    opt.pretrain_class_number = 0
    opt.beta = 60
    opt.beta_1 = 0.01

    opt.ins_weight = 0.01
    opt.ins_temp = 10
    opt.critic_iter = 5
    opt.lr = 1e-4
    opt.shuffer_class = False
elif opt.dataset == 'SUN':
    opt.resSize = 2048
    opt.nclass_all = 717
    opt.nclass_seen = 645
    opt.batch_size = 512
    opt.dir = 'models\SUN'
    opt.nz = 102
    opt.attSize = 102
    opt.nepoch = 2001
    opt.syn_num_u = 100
    opt.syn_num_s = 60
    opt.syn_num_rp = 50
    opt.task_num = 5
    opt.syn_num_replay = 100
    opt.pretrain_class_number = 0
    opt.beta = 1
    opt.beta_1 = 0.1
    opt.ins_weight = 0.1
    opt.outzSize = 128
    opt.critic_iter = 5
    opt.lr = 1e-4
    opt.shuffer_class = False
elif opt.dataset == 'APY':
    opt.resSize = 2048
    opt.attSize = 64
    opt.nz = 64
    opt.nclass_all = 32
    opt.nclass_seen = 20
    opt.batch_size = 64
    opt.dir = 'models\APY'
    opt.syn_num_u = 100
    opt.syn_num_s = 50
    opt.syn_num_rp = 50
    opt.nepoch = 1001
    opt.syn_num_replay = 300
    opt.task_num = 4
    opt.pretrain_class_number = 0
    opt.ins_weight = 0.1
    opt.beta = 110
    opt.beta_1 = 0.01
    opt.ins_temp = 0.1
    opt.outzSize = 128
    opt.critic_iter = 5
    opt.lr = 1e-4
    opt.shuffer_class = False

elif opt.dataset == 'FLO':
    opt.dir = 'models\FLO'
    opt.resSize = 2048
    opt.nz = 1024
    # opt.manualSeed = 3407
    opt.nclass_all = 102
    opt.nclass_seen =82
    opt.attSize = 1024
    opt.task_num = 5
    opt.critic_iter = 5
    opt.lr = 1e-4
    opt.nepoch = 2001
    opt.shuffer_class = False


    opt.batch_size = 512
    opt.syn_num_u = 300
    opt.syn_num_s = 100
    opt.syn_num_rp = 50
    opt.syn_num_replay = 10
    opt.pretrain_class_number = 0
    opt.beta = 120
    opt.beta_1 = 0.1
    opt.ins_weight = 0.001
    opt.outzSize = 128
    opt.ins_temp = 0.1


print(opt)
if __name__ == '__main__':
    print(opt)
    data = DATA_LOADER(opt)
    data.current_class_index()
    data.next_batch_current(16)

    # print(data.split_unseen_class())