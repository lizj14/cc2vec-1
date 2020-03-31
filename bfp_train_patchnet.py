import pickle
from parameters import read_args_cnn
import numpy as np
from ultis import mini_batches_extended,mini_batches_noftr
import torch
import os
import datetime
from hierarchical_patchnet_noftr_classification import PatchNetExtented
import torch.nn as nn


def save(model, save_dir, save_prefix, epochs):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_{}.pt'.format(save_prefix, epochs)
    torch.save(model.state_dict(), save_path)


def running_train(batches, model, params):
    optimizer = torch.optim.Adam(model.parameters(), lr=params.l2_reg_lambda)
    steps, num_epoch = 0, 1
    for epoch in range(1, params.num_epochs + 1):
        for batch in batches:
            pad_msg, pad_added_code, pad_removed_code, labels = batch
            pad_msg, pad_added_code, pad_removed_code, labels =  torch.tensor(
                pad_msg).cuda(), torch.tensor(pad_added_code).cuda(), torch.tensor(
                pad_removed_code).cuda(), torch.cuda.FloatTensor(labels)

            optimizer.zero_grad()
            predict = model.forward_noftr(pad_msg, pad_added_code, pad_removed_code)
            loss = nn.BCELoss()
            loss = loss(predict, labels)
            loss.backward()
            optimizer.step()

            steps += 1
            if steps % params.log_interval == 0:
                print('\rEpoch: {} step: {} - loss: {:.6f}'.format(num_epoch, steps, loss.item()))

            # if steps % params.test_interval == 0:
            #     if torch.cuda.is_available():
            #         predict, labels = predict.cpu().detach().numpy(), labels.cpu().detach().numpy()
            #     else:
            #         predict, labels = predict.detach().numpy(), labels.detach().numpy()
            #     predict = [1 if p >= 0.5 else 0 for p in predict]
            #     accuracy = accuracy_score(y_true=labels, y_pred=predict)
            #     print(
            #         '\rEpoch: {} Step: {} - loss: {:.6f}  acc: {:.4f}'.format(num_epoch, steps, loss.item(), accuracy))

        save(model, params.save_dir, 'epoch', num_epoch)
        num_epoch += 1


def train_model(data, params):
    pad_msg, pad_added_code, pad_removed_code, labels, dict_msg, dict_code = data
    # batches = mini_batches_extended(X_ftr=pad_extended_ftr, X_msg=pad_msg, X_added_code=pad_added_code,
    #                                 X_removed_code=pad_removed_code, Y=labels, mini_batch_size=input_option.batch_size)
    batches = mini_batches_noftr( X_msg=pad_msg, X_added_code=pad_added_code,
                                    X_removed_code=pad_removed_code, Y=labels, mini_batch_size=input_option.batch_size)
    params.cuda = (not params.no_cuda) and torch.cuda.is_available()
    del params.no_cuda

    params.filter_sizes = [int(k) for k in params.filter_sizes.split(',')]
    params.save_dir = os.path.join(params.save_dir, params.datetime)
    params.vocab_msg, params.vocab_code = len(dict_msg), len(dict_code)
    if len(labels.shape) == 1:
        params.class_num = 1
    else:
        params.class_num = labels.shape[1]

    # Device configuration
    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PatchNetExtented(args=params)
    if torch.cuda.is_available():
        model = model.cuda()
    running_train(batches=batches, model=model, params=params)


if __name__ == '__main__':
    # loading data
    ##########################################################################################################
    # data = pad_msg, pad_added_code, pad_removed_code, labels, dict_msg, dict_code
    # pad_msg: padding commit message
    # pad_added_code: padding added code
    # pad_removed_code: padding removed code
    # labels: label of our data, stable or non-stable patches
    # dict_msg: dictionary of commit message
    # dict_code: dictionary of commit code

    with open('./data/linux_bfp_train.pickle', 'rb') as input:
        data = pickle.load(input)
    pad_msg, pad_added_code, pad_removed_code, labels, dict_msg, dict_code = data
    pad_msg, pad_added_code, pad_removed_code = np.array(pad_msg), np.array(pad_added_code), np.array(pad_removed_code)
    ##########################################################################################################
    print(pad_msg.shape, pad_added_code.shape, pad_removed_code.shape, labels.shape)
    print('Shape of the commit message:', pad_msg.shape)
    print('Shape of the added/removed code:', (pad_added_code.shape, pad_removed_code.shape))
    print('Total words in the message dictionary: ', len(dict_msg))
    print('Total words in the code dictionary: ', len(dict_code))

    input_option = read_args_cnn().parse_args()
    input_help = read_args_cnn().print_help()

    input_option.datetime = 'patchnet'

    data = (pad_msg, pad_added_code, pad_removed_code, labels, dict_msg, dict_code)
    train_model(data=data, params=input_option)
