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
from numpy import loadtxt

def save(model, save_dir, save_prefix, epochs):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_{}.pt'.format(save_prefix, epochs)
    torch.save(model.state_dict(), save_path)


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
        for batch in batches:
            pad_added_code, pad_removed_code, labels, msg = batch
            if torch.cuda.is_available():
                pad_added_code, pad_removed_code, labels = torch.tensor(pad_added_code).cuda(), torch.tensor(
                    pad_removed_code).cuda(), torch.cuda.FloatTensor(labels)
            else:
                pad_added_code, pad_removed_code, labels = torch.tensor(pad_added_code).long(), torch.tensor(
                    pad_removed_code).long(), torch.tensor(labels).float()

            optimizer.zero_grad()
            predict = model.forward(pad_added_code, pad_removed_code)
            loss = nn.BCELoss()
            loss = loss(predict, labels)
            loss.backward()
            optimizer.step()

            steps += 1
            if steps % params.log_interval == 0:
                print('\rEpoch: {} step: {} - loss: {:.6f}'.format(num_epoch, steps, loss.item()))

        save(model, params.save_dir, 'epoch', num_epoch)
        num_epoch += 1


    train_vectors, cnt = list(), 0
    msg_list = list()
    for batch in train:
        pad_added_code, pad_removed_code, labels, msg = batch
        if torch.cuda.is_available():
            pad_added_code, pad_removed_code, labels = torch.tensor(pad_added_code).cuda(), torch.tensor(
                pad_removed_code).cuda(), torch.cuda.FloatTensor(labels)
        else:
            pad_added_code, pad_removed_code, labels = torch.tensor(pad_added_code).long(), torch.tensor(
                pad_removed_code).long(), torch.tensor(labels).float()

        optimizer.zero_grad()
        commits_vector = model.forward_commit_embeds(pad_added_code, pad_removed_code)

        if torch.cuda.is_available():
            commits_vector = commits_vector.cpu().detach().numpy()
        else:
            commits_vector = commits_vector.detach().numpy()

        if cnt == 0:
            train_vectors = commits_vector
            msg_list = msg
        else:
            train_vectors = np.concatenate((train_vectors, commits_vector), axis=0)
            msg_list = np.concatenate((msg_list, msg), axis=0)
        print('Batch numbers:', cnt)
        cnt += 1

    #test
    test_predict_result = []
    output = './data/jiang_ase_2017/test.3000.msg.predict'
    for batch in test:
        pad_added_code, pad_removed_code, labels, msg = batch
        if torch.cuda.is_available():
            pad_added_code, pad_removed_code, labels = torch.tensor(pad_added_code).cuda(), torch.tensor(
                pad_removed_code).cuda(), torch.cuda.FloatTensor(labels)
        else:
            pad_added_code, pad_removed_code, labels = torch.tensor(pad_added_code).long(), torch.tensor(
                pad_removed_code).long(), torch.tensor(labels).float()

        optimizer.zero_grad()
        predict = model.forward_commit_embeds(pad_added_code, pad_removed_code)

        if torch.cuda.is_available():
            commits_vector = predict.cpu().detach().numpy()
        else:
            commits_vector = predict.detach().numpy()

        for vec in commits_vector:
            best_sim = -1
            best_index = 0
            for index in range(len(train_vectors)):
                a, b = vec, train_vectors[index]
                # np.cosine_similarity()
                cos_sim = dot(a, b) / (norm(a) * norm(b))
                # cos_sim = pairwise.cosine_similarity(a.reshape((1,-1)),b.reshape((1,-1)))
                if cos_sim > best_sim:
                    best_sim = cos_sim
                    best_index = index

            test_predict_result.append(msg_list[best_index][0])

    test_predict_result = test_predict_result[-3000:]
    with open(output,'w+') as file:
        for line in test_predict_result:
            file.write(line+'\n')




def train_model(commit_diff, commit_msg, params, padding_code_info):
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
    params.filter_sizes = [int(k) for k in params.filter_sizes.split(',')]
    params.save_dir = os.path.join(params.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    params.vocab_code = len(dict_code) + 1
    params.code_line = max_line
    if len(labels.shape) == 1:
        params.class_num = 1
    else:
        params.class_num = labels.shape[1]

    # # Device configuration
    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = PatchEmbedding(args=params)
    if torch.cuda.is_available():
        model = model.cuda()
    running_train(batches=batches, model=model, params=params)
