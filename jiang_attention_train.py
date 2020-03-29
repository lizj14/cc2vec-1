import pickle
from parameters import read_args
from hierarchical_attention import HierachicalRNN

from jiang_prepared import load_Jiang_code_data
from ultis import load_file
# from jiang_train import train_model
import os
import torch
from jiang_padding import padding_commit_code, commit_msg_label
from preprocessing_linux_bfp.ultis import mini_batches_topwords
import numpy as np
import datetime
from jiang_patch_embedding import PatchEmbedding
import torch.nn as nn
from numpy import dot
from numpy.linalg import norm

def save(model, save_dir, save_prefix, epochs):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_{}.pt'.format(save_prefix, epochs)
    torch.save(model.state_dict(), save_path)


def convert_msg_to_label(pad_msg, dict_msg):
    nrows, ncols = pad_msg.shape
    labels = list()
    for i in range(nrows):
        column = list(set(list(pad_msg[i, :])))
        label = np.zeros(len(dict_msg))
        for c in column:
            label[c] = 1
        labels.append(label)
    return np.array(labels)

def reshape_code(data):
    ncommit, nline, nlength = data.shape[0], data.shape[1], data.shape[2]
    data = np.reshape(data, (ncommit, 1, nline, nlength))
    return data

def running_train(batches, model, params):
    optimizer = torch.optim.Adam(model.parameters(), lr=params.l2_reg_lambda)
    steps, num_epoch = 0, 1
    train = batches[:409]
    test = batches[409:]
    for epoch in range(1, params.num_epochs + 1):
        for batch in batches[-5:]:
            model.batch_size = batch[0].shape[0]
            # reset the hidden state of hierarchical attention model
            state_word = model.init_hidden_word()
            state_sent = model.init_hidden_sent()
            state_hunk = model.init_hidden_hunk()

            pad_added_code, pad_removed_code, labels, msg = batch
            if torch.cuda.is_available():
                labels = torch.cuda.FloatTensor(labels)
            else:
                # labels = torch.FloatTensor(labels)
                labels = torch.tensor(labels).float()
            optimizer.zero_grad()
            predict = model.forward(pad_added_code, pad_removed_code, state_hunk, state_sent, state_word)
            loss = nn.BCELoss()
            loss = loss(predict, labels)
            loss.backward()
            optimizer.step()

            steps += 1
            if steps % params.log_interval == 0:
                print('\rEpoch: {} step: {} - loss: {:.6f}'.format(num_epoch, steps, loss.item()))

        save(model, params.save_dir, 'epoch', num_epoch)
        num_epoch += 1

    # calcu all
    all_vectors, cnt = list(), 0
    msg_list = list()
    output = './data/jiang_ase_2017/test.3000.msg.predict.attention'
    with torch.no_grad():
        model.eval()
        for batch in batches:
            model.batch_size = batch[0].shape[0]
            # reset the hidden state of hierarchical attention model
            state_word = model.init_hidden_word()
            state_sent = model.init_hidden_sent()
            state_hunk = model.init_hidden_hunk()
            # state_sent = model.init_hidden_sent()
            # state_hunk = model.init_hidden_hunk()
            # model.wordRNN.batch_size = model.batch_size

            pad_added_code, pad_removed_code, labels, msg = batch
            # optimizer.zero_grad()
            commits_vector = model.forward_commit_embeds(pad_added_code, pad_removed_code, state_hunk, state_sent,
                                                         state_word)
            if torch.cuda.is_available():
                commits_vector = commits_vector.cpu().detach().numpy()
            else:
                commits_vector = commits_vector.detach().numpy()

            if cnt == 0:
                all_vectors = commits_vector
                msg_list = msg
            else:
                all_vectors = np.concatenate((all_vectors, commits_vector), axis=0)
                msg_list = np.concatenate((msg_list, msg), axis=0)
            print('Batch numbers:', cnt)
            cnt += 1
    test_predict_result = []
    for i in range(-3001, len(all_vectors)):
        best_sim = -1
        best_index = 0
        for j in range(len(all_vectors) - 2998):
            a, b = all_vectors[i], all_vectors[j]
            # np.cosine_similarity()
            cos_sim = dot(a, b) / (norm(a) * norm(b))
            # cos_sim = pairwise.cosine_similarity(a.reshape((1,-1)),b.reshape((1,-1)))
            if cos_sim > best_sim:
                best_sim = cos_sim
                best_index = j
        test_predict_result.append(msg_list[best_index][0])


    # train_vectors, cnt = list(), 0
    # msg_list = list()
    # with torch.no_grad():
    #     model.eval()
    #     for batch in train:
    #         model.batch_size = batch[0].shape[0]
    #         # reset the hidden state of hierarchical attention model
    #         state_word = model.init_hidden_word()
    #         state_sent = model.init_hidden_sent()
    #         state_hunk = model.init_hidden_hunk()
    #
    #         pad_added_code, pad_removed_code, labels, msg = batch
    #         commits_vector = model.forward_commit_embeds(pad_added_code, pad_removed_code, state_hunk, state_sent,
    #                                                      state_word)
    #
    #         if torch.cuda.is_available():
    #             commits_vector = commits_vector.cpu().detach().numpy()
    #         else:
    #             commits_vector = commits_vector.detach().numpy()
    #
    #         if cnt == 0:
    #             train_vectors = commits_vector
    #             msg_list = msg
    #         else:
    #             train_vectors = np.concatenate((train_vectors, commits_vector), axis=0)
    #             msg_list = np.concatenate((msg_list, msg), axis=0)
    #         print('Batch numbers:', cnt)
    #         cnt += 1
    #
    # # test
    # test_predict_result = []
    # output = './data/jiang_ase_2017/test.3000.msg.predict.attention'
    # with torch.no_grad():
    #     model.eval()
    #     for batch in test:
    #         model.batch_size = batch[0].shape[0]
    #         # reset the hidden state of hierarchical attention model
    #         state_word = model.init_hidden_word()
    #         state_sent = model.init_hidden_sent()
    #         state_hunk = model.init_hidden_hunk()
    #
    #         pad_added_code, pad_removed_code, labels, msg = batch
    #         commits_vector = model.forward_commit_embeds(pad_added_code, pad_removed_code, state_hunk, state_sent,
    #                                                      state_word)
    #
    #         if torch.cuda.is_available():
    #             commits_vector = commits_vector.cpu().detach().numpy()
    #         else:
    #             commits_vector = commits_vector.detach().numpy()
    #
    #         for vec in commits_vector:
    #             best_sim = -1
    #             best_index = 0
    #             for index in range(len(train_vectors)):
    #                 a, b = vec, train_vectors[index]
    #                 # np.cosine_similarity()
    #                 cos_sim = dot(a, b) / (norm(a) * norm(b))
    #                 # cos_sim = pairwise.cosine_similarity(a.reshape((1,-1)),b.reshape((1,-1)))
    #                 if cos_sim > best_sim:
    #                     best_sim = cos_sim
    #                     best_index = index
    #
    #             test_predict_result.append(msg_list[best_index][0])

    test_predict_result = test_predict_result[-3000:]
    with open(output, 'w+') as file:
        for line in test_predict_result:
            file.write(line + '\n')


def train_model(data, params):
    commit_diff, commit_msg = data

    max_line, max_length = padding_code_info[0], padding_code_info[1]
    pad_removed_code, pad_added_code, dict_code = padding_commit_code(data=commit_diff, max_line=max_line,
                                                                      max_length=max_length)
    pad_removed_code, pad_added_code = reshape_code(data=pad_removed_code), reshape_code(data=pad_added_code)
    print('Dictionary of commit code has size: %i' % (len(dict_code)))
    print('Shape of removed code and added code:', pad_removed_code.shape, pad_added_code.shape)

    labels, dict_msg = commit_msg_label(data=commit_msg)
    batches = mini_batches_topwords(X_added_code=pad_added_code, X_removed_code=pad_removed_code, Y=labels,
                                    mini_batch_size=params.batch_size,msg=np.array(commit_msg).reshape((len(commit_msg),-1)))

    params.cuda = (not params.no_cuda) and torch.cuda.is_available()
    del params.no_cuda

    params.save_dir = os.path.join(params.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    params.vocab_code = len(dict_code)
    if len(labels.shape) == 1:
        params.class_num = 1
    else:
        params.class_num = labels.shape[1]

    # Device configuration
    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hierarchical_attention = HierachicalRNN(args=params)
    if torch.cuda.is_available():
        model = hierarchical_attention.cuda()
    else:
        model = hierarchical_attention.cpu()
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

    # input_option = read_args_jiang().parse_args()
    # input_help = read_args_jiang().print_help()

    # loading the commit code
    ##################################################################################
    ##################################################################################
    path_train_diff = './data/jiang_ase_2017/train.26208.diff'
    data_train_diff = load_Jiang_code_data(pfile=path_train_diff)
    path_test_diff = './data/jiang_ase_2017/test.3000.diff'
    data_test_diff = load_Jiang_code_data(pfile=path_test_diff)
    data_diff = data_train_diff + data_test_diff
    # data_diff = data_test_diff
    padding_code_info = (15, 40)  # max_line = 15; max_length = 40
    ##################################################################################
    ##################################################################################

    # loading the commit msg
    ##################################################################################
    ##################################################################################
    path_train_msg = './data/jiang_ase_2017/train.26208.msg'
    data_train_msg = load_file(path_file=path_train_msg)
    path_test_msg = './data/jiang_ase_2017/test.3000.msg'
    data_test_msg = load_file(path_file=path_test_msg)
    data_msg = data_train_msg + data_test_msg
    # data_msg = data_test_msg


    # pad_msg, pad_added_code, pad_removed_code, labels, dict_msg, dict_code = data
    # ##########################################################################################################
    # print(pad_msg.shape, pad_added_code.shape, pad_removed_code.shape, labels.shape)
    # print('Shape of the commit message:', pad_msg.shape)
    # print('Shape of the added/removed code:', (pad_added_code.shape, pad_removed_code.shape))
    # print('Total words in the message dictionary: ', len(dict_msg))
    # print('Total words in the code dictionary: ', len(dict_code))
    #
    # pad_msg_labels = convert_msg_to_label(pad_msg=pad_msg, dict_msg=dict_msg)
    # print('Shape of the output labels: ', pad_msg_labels.shape)

    # data = (pad_added_code, pad_removed_code, pad_msg_labels, dict_msg, dict_code)

    input_option = read_args().parse_args()
    input_help = read_args().print_help()

    input_option.embed_size = 64
    input_option.hidden_size = 32

    data = (data_diff, data_msg)

    input_option.num_epochs = 5

    train_model(data=data, params=input_option)
