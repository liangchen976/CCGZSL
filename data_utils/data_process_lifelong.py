import numpy as np
import torch
# 需要找到划分后的对应于之前的index
def split_label_average(rawlabel_list, num, shuffle = False):
    '''
    对label均匀划分
    :param rawlabel_list: 传入的数据label list
    :param num: 需要划分成几份数据 或者叫几个task
    :param shuffle:是否需要将label打乱
    :return:划分为num个的label的index
    '''
    unique_label = np.unique(rawlabel_list)
    if shuffle:
        unique_label = np.random.permutation(unique_label)
    a = unique_label.shape[0]
    n = num
    k, m = divmod(a, n)
    splited_class_label = [unique_label[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] \
                               for i in list(range(n))]

    # print(f'splited_class_label is {splited_class_label}')
    # print(f'rawlabel_list is {rawlabel_list}')
    rawlabel_list_torch = torch.from_numpy(np.array(rawlabel_list)).unsqueeze(dim=1)
    index_res = []
    for item in splited_class_label:
        # print(f'temp is {temp[i]}')
        temp = torch.from_numpy(np.array(item))
        index1 = torch.where(rawlabel_list_torch == temp)[0].numpy()
        index_res.append(index1)
    # res = []
    # for item in splited_class_label:
    #     temp = np.array(item)
    #     res.append(temp)
    #
    # return res
    return index_res

if __name__ == '__main__':
    # l = [i for i in range(20)]
    l = np.array([2, 2, 2, 3, 3, 3, 4, 4, 4, 2, 1, 9, 8, 9, 34, 25, 67, 34, 2, 12, 32, 12, 5])
    index = [1, 3, 5]
    print(l[index])