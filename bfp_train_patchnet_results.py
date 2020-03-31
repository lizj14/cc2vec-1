import pickle
from parameters import read_args_cnn
import numpy as np
import torch
from ultis import mini_batches_extended
import os
from hierarchical_cnn_classification import PatchNetExtented
from sklearn.metrics import precision_score


def load_model(data, params):
    pad_extended_ftr, pad_msg, pad_added_code, pad_removed_code, labels, dict_msg, dict_code = data
    batches = mini_batches_extended(X_ftr=pad_extended_ftr, X_msg=pad_msg, X_added_code=pad_added_code,
                                    X_removed_code=pad_removed_code, Y=labels, mini_batch_size=input_option.batch_size)
    params.cuda = (not params.no_cuda) and torch.cuda.is_available()
    del params.no_cuda

    params.filter_sizes = [int(k) for k in params.filter_sizes.split(',')]
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
    return batches, model


def convert_to_binary(y_pred):
    y_pred = list(y_pred)
    y_pred = [1 if value >= 0.5 else 0 for value in y_pred]
    return np.array(y_pred)


def run_results(path, batches, model, params, nepoch):
    model.load_state_dict(torch.load(path))
    true_labels, pred_labels, cnt = list(), list(), 0
    with torch.no_grad():
        model.eval()
        for batch in batches:
            pad_msg, pad_added_code, pad_removed_code, labels = batch
            pad_msg, pad_added_code, pad_removed_code, labels = torch.tensor(
                pad_msg).cuda(), torch.tensor(pad_added_code).cuda(), torch.tensor(
                pad_removed_code).cuda(), torch.cuda.FloatTensor(labels)
            pred = model.forward_noftr(pad_msg, pad_added_code, pad_removed_code)
            pred, labels = pred.cpu().detach().numpy(), labels.cpu().detach().numpy()
            if cnt == 0:
                pred_labels, true_labels = pred, labels
            else:
                pred_labels = np.concatenate((pred_labels, pred), axis=0)
                true_labels = np.concatenate((true_labels, labels), axis=0)
            # print('Batch numbers:', cnt)
            cnt += 1
        path_save = './results/' + params.path_model + '/'
        save_folder = os.path.dirname(path_save)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        print(pred_labels.shape, true_labels.shape)
        print(path_save + 'epoch_' + str(nepoch) + '.txt')
        print('Precision:', precision_score(y_true=true_labels, y_pred=convert_to_binary(pred_labels)))
        np.savetxt(path_save + 'epoch_' + str(nepoch) + '.txt', pred_labels)


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

    with open('./data/linux_bfp_test.pickle', 'rb') as input:
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

    # input_option.path_model = '2019-07-22_16-57-31'
    # input_option.start_model = 1
    # input_option.end_model = 100

    input_option.path_model = 'patchnet'
    input_option.start_model = 30
    input_option.end_model = 50

    batches, model = load_model(data=data, params=input_option)
    for epoch in range(input_option.start_model, input_option.end_model + 1):
        path_model = './snapshot/' + input_option.path_model + '/epoch_' + str(epoch) + '.pt'
        run_results(path=path_model, batches=batches, model=model, params=input_option, nepoch=epoch)
