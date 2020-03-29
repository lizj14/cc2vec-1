#  * This file is part of PatchNet, licensed under the terms of the GPL v2.
#  * See copyright.txt in the PatchNet source code for more information.
#  * The PatchNet source code can be obtained at
#  * https://github.com/hvdthong/PatchNetTool

from preprocessing_linux_bfp.extracting import commit_id, commit_stable, commit_msg, commit_date, commit_code
from preprocessing_linux_bfp.reformating import reformat_file, reformat_hunk
import numpy as np
import math
import os
from preprocessing_linux_bfp.arguments import read_args
from preprocessing_linux_bfp.padding import padding_commit
import pickle
from sklearn.model_selection import StratifiedShuffleSplit, KFold
import pickle
def info_label(data):
    pos = [d for d in data if d == 1]
    neg = [d for d in data if d == 0]
    print('Positive: %i -- Negative: %i' % (len(pos), len(neg)))


def get_index(data, index):
    return [data[i] for i in index]

def load_file(path_file):
    lines = list(open(path_file, "r").readlines())
    lines = [l.strip() for l in lines]
    return lines


def load_dict_file(path_file):
    lines = list(open(path_file, "r").readlines())
    dictionary = dict()
    for line in lines:
        key, value = line.split('\t')[0], line.split('\t')[1]
        dictionary[key] = value
    return dictionary


def write_dict_file(path_file, dictionary):
    split_path = path_file.split("/")
    path_ = split_path[:len(split_path) - 1]
    path_ = "/".join(path_)

    if not os.path.exists(path_):
        os.makedirs(path_)

    with open(path_file, 'w') as out_file:
        for key in dictionary.keys():
            # write line to output file
            out_file.write(str(key) + '\t' + str(dictionary[key]))
            out_file.write("\n")
        out_file.close()


def write_file(path_file, data):
    split_path = path_file.split("/")
    path_ = split_path[:len(split_path) - 1]
    path_ = "/".join(path_)
    if len(split_path) > 1:
        if not os.path.exists(path_):
            os.makedirs(path_)

    with open(path_file, 'w') as out_file:
        for line in data:
            # write line to output file
            out_file.write(str(line))
            out_file.write("\n")
        out_file.close()


def select_commit_based_topwords(words, commits):
    new_commit = list()
    for c in commits:
        msg = c['msg'].split(',')
        for w in msg:
            if w in words:
                new_commit.append(c)
                break
    return new_commit


def commits_index(commits):
    commits_index = [i for i, c in enumerate(commits) if c.startswith("commit:")]
    return commits_index


def commit_info(commit):
    id = commit_id(commit)
    stable = commit_stable(commit)
    date = commit_date(commit)
    msg = commit_msg(commit)
    code = commit_code(commit)
    return id, stable, date, msg, code


def extract_commit(path_file):
    # extract commit from july data
    commits = load_file(path_file=path_file)
    indexes = commits_index(commits=commits)
    dicts = list()
    for i in range(0, len(indexes)):
        dict = {}
        if i == len(indexes) - 1:
            id, stable, date, msg, code = commit_info(commits[indexes[i]:])
        else:
            id, stable, date, msg, code = commit_info(commits[indexes[i]:indexes[i + 1]])
        dict["id"] = id
        dict["stable"] = stable
        dict["date"] = date
        dict["msg"] = msg
        dict["code"] = code
        dicts.append(dict)
    return dicts


def reformat_commit_code(commits, num_file, num_hunk, num_loc, num_leng):
    commits = reformat_file(commits=commits, num_file=num_file)
    commits = reformat_hunk(commits=commits, num_hunk=num_hunk, num_loc=num_loc, num_leng=num_leng)
    return commits


def random_mini_batch(X_msg, X_added_code, X_removed_code, Y, mini_batch_size=64, seed=0):
    m = X_msg.shape[0]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X_msg = X_msg[permutation, :]
    shuffled_X_added = X_added_code[permutation, :, :, :]
    shuffled_X_removed = X_removed_code[permutation, :, :, :]
    if len(Y.shape) == 1:
        shuffled_Y = Y[permutation]
    else:
        shuffled_Y = Y[permutation, :]
    # shuffled_Y = Y[permutation, :]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / float(mini_batch_size))  # number of mini batches of size mini_batch_size in your partitionning
    num_complete_minibatches = int(num_complete_minibatches)
    for k in range(0, num_complete_minibatches):
        mini_batch_X_msg = shuffled_X_msg[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_X_added = shuffled_X_added[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :]
        mini_batch_X_removed = shuffled_X_removed[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :]
        if len(Y.shape) == 1:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        else:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        # mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X_msg, mini_batch_X_added, mini_batch_X_removed, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X_msg = shuffled_X_msg[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch_X_added = shuffled_X_added[num_complete_minibatches * mini_batch_size: m, :, :, :]
        mini_batch_X_removed = shuffled_X_removed[num_complete_minibatches * mini_batch_size: m, :, :, :]
        if len(Y.shape) == 1:
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m]
        else:
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        # mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_X_msg, mini_batch_X_added, mini_batch_X_removed, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches


def mini_batches(X_msg, X_added_code, X_removed_code, Y, mini_batch_size=64, seed=0):
    m = X_msg.shape[0]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: No shuffle (X, Y)
    shuffled_X_msg = X_msg
    shuffled_X_added = X_added_code
    shuffled_X_removed = X_removed_code
    shuffled_Y = Y

    # Step 2: Partition (X, Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / float(mini_batch_size))  # number of mini batches of size mini_batch_size in your partitionning
    num_complete_minibatches = int(num_complete_minibatches)
    for k in range(0, num_complete_minibatches):
        mini_batch_X_msg = shuffled_X_msg[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_X_added = shuffled_X_added[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :]
        mini_batch_X_removed = shuffled_X_removed[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :]
        if len(Y.shape) == 1:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        else:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X_msg, mini_batch_X_added, mini_batch_X_removed, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X_msg = shuffled_X_msg[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch_X_added = shuffled_X_added[num_complete_minibatches * mini_batch_size: m, :, :, :]
        mini_batch_X_removed = shuffled_X_removed[num_complete_minibatches * mini_batch_size: m, :, :, :]
        if len(Y.shape) == 1:
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m]
        else:
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_X_msg, mini_batch_X_added, mini_batch_X_removed, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches


def mini_batches_topwords(X_added_code, X_removed_code, Y, msg ,mini_batch_size=64, seed=0):
    m = Y.shape[0]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: No shuffle (X, Y)
    shuffled_X_added = X_added_code
    shuffled_X_removed = X_removed_code
    shuffled_Y = Y

    # Step 2: Partition (X, Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / float(mini_batch_size))  # number of mini batches of size mini_batch_size in your partitioning
    num_complete_minibatches = int(num_complete_minibatches)
    for k in range(0, num_complete_minibatches):
        mini_batch_X_added = shuffled_X_added[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :]
        mini_batch_X_removed = shuffled_X_removed[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :]
        if len(Y.shape) == 1:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
            mini_batch_msg = msg[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        else:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
            mini_batch_msg = msg[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X_added, mini_batch_X_removed, mini_batch_Y,mini_batch_msg)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X_added = shuffled_X_added[num_complete_minibatches * mini_batch_size: m, :, :, :]
        mini_batch_X_removed = shuffled_X_removed[num_complete_minibatches * mini_batch_size: m, :, :, :]
        if len(Y.shape) == 1:
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m]
            mini_batch_msg = msg[num_complete_minibatches * mini_batch_size: m]
        else:
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
            mini_batch_msg = msg[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_X_added, mini_batch_X_removed, mini_batch_Y,mini_batch_msg)
        mini_batches.append(mini_batch)
    return mini_batches

def folding_data_authordate(pad_msg, pad_added_code, pad_removed_code, labels, dict_msg, dict_code,ids, n_folds):
    kf = KFold(n_splits=n_folds, random_state=0)
    indexes = list(kf.split(pad_msg))
    train_index, test_index = indexes[len(indexes) - 1]

    pad_msg_train, pad_msg_test = get_index(data=pad_msg, index=train_index), get_index(data=pad_msg,
                                                                                        index=test_index)
    pad_added_code_train, pad_added_code_test = get_index(data=pad_added_code, index=train_index), get_index(data=pad_added_code,
                                                                                           index=test_index)
    pad_removed_code_train, pad_removed_code_test = get_index(data=pad_removed_code, index=train_index), get_index(data=pad_removed_code,
                                                                                           index=test_index)

    labels_train, labels_test = labels[train_index], labels[test_index]
    info_label(data=labels_train)
    info_label(data=labels_test)
    # ids_train, ids_test = get_index(data=ids, index=train_index), get_index(data=ids, index=test_index)

    train = (pad_msg_train, pad_added_code_train, pad_removed_code_train, labels_train, dict_msg, dict_code )
    test = (pad_msg_test, pad_added_code_test, pad_removed_code_test, labels_test, dict_msg, dict_code )
    return train, test

if __name__ == "__main__":
    # path_data = "./data/linux/newres_funcalls_words_jul28.out"
    path_data = "../data/linux/newres_funcalls_jul28.out"
    commits_ = extract_commit(path_file=path_data)
    nfile, nhunk, nloc, nleng = 1, 8, 10, 120
    commits = reformat_commit_code(commits=commits_, num_file=nfile, num_hunk=nhunk, num_loc=nloc, num_leng=nleng)

    input_option = read_args().parse_args()
    input_help = read_args().print_help()
    pad_msg, pad_added_code, pad_removed_code, labels, dict_msg, dict_code = padding_commit(commits=commits,
                                                                                            params=input_option)
    # data = (pad_msg, pad_added_code, pad_removed_code, labels, dict_msg, dict_code)

    train, test = folding_data_authordate(pad_msg,pad_added_code, pad_removed_code, labels, dict_msg, dict_code, None,5)
    print('Number of commits:', len(commits))
    print('Dictionary of commit message has size: %i' % (len(dict_msg)))
    print('Dictionary of commit code has size: %i' % (len(dict_code)))

    # with open('../data/linux_bfp.pickle', 'wb') as output:
    with open('../data/linux_bfp_train.pickle', 'wb') as output:
        pickle.dump(train, output, pickle.HIGHEST_PROTOCOL)
    with open('../data/linux_bfp_test.pickle', 'wb') as output:
        pickle.dump(test, output, pickle.HIGHEST_PROTOCOL)
