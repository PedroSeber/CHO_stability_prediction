import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from collections import OrderedDict
from scipy.special import softmax

def predict_CHO_stability(data_file, threshold = 0.5, n_classes = 2, batch_size = 2048):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Data preparation
    data = pd.read_csv(data_file, header = None).values.squeeze()
    train_mean = np.array([1.82840880e+01, 1.70397486e+01, 1.58317159e+01, 1.76026134e+01, 1.57249573e+01, 1.61467173e+01, 1.45145586e+01, 1.69805766e+01, 9.57234042e+00, 1.40199965e+01, 1.47027288e+01, 1.03021206e+01, 6.37245424e+00, 9.32560683e+00, 9.23944490e+00, 1.27002144e+01,
       1.38708695e+01, 1.92559795e-02, 2.89617855e-02, 3.30471691e-02, 5.55090341e-02, 4.17017520e-02, 4.23555530e-02, 4.63854359e-02, 3.20464729e-02, 7.44406230e-02, 5.28907527e-02, 2.78059377e-02, 5.43169099e-02, 5.30831779e-02, 5.01524922e-02, 5.93472695e-02,
       5.86822057e-02, 9.37470343e-01, 9.86356851e-01, 8.54458334e-01, 1.00543187e+00, 1.16235658e+00, 9.15795524e-01, 6.33336741e-01, 8.19013105e-01, 1.29766218e+00, 1.28398847e+00, 1.00984769e+00, 1.05204390e+00, 9.07576157e-01, 1.01459555e+00, 1.25132354e+00,
       4.46036045e+00, 4.31669273e+00, 4.74749011e+00, 5.13905389e+00, 4.41582950e+00, 4.58836529e+00, 5.51125579e+00, 4.20455149e+00, 4.61294947e+00, 3.95586873e+00, 2.46571623e+00, 2.86086882e+00, 2.72904095e+00, 2.68002257e+00, 2.83204090e+00, 1.69189949e+00,
       1.50989140e+01, 1.05156124e+01, 1.54629590e+01, 1.53495199e+01, 1.40015810e+01, 1.49854223e+01, 1.65937758e+01, 1.55073462e+01, 1.80149690e+01, 1.79976555e+01, 8.55663474e+00, 1.10753332e+01, 1.06848211e+01, 1.08950663e+01, 1.68378565e+01, 1.61619550e+01,
       1.68741585e-02, 2.88046979e-02, 1.66976312e-02, 3.92256032e-02, 2.39928873e-02, 2.06106481e-02, 2.63527807e-02, 2.89405708e-02, 3.72232992e-02, 4.78889191e-02, 3.84057704e-02, 8.12487560e-03, 3.07756847e-02, 3.36274284e-02, 3.79508217e-02, 2.62033201e-02, 2.17206063e-02])
    train_std = np.array([22.00166595, 21.56184843, 21.15541242, 23.03635045, 20.19165995, 18.77740152, 17.66744693, 18.94419685, 12.40460467, 16.24445543, 16.95006457, 10.97282802,  6.71030577,  9.85822956, 10.07450332, 14.19681706, 15.97756777,  0.3007885 ,  0.37308602,  0.41580585,
        0.49822076,  0.4526807 ,  0.4638688 ,  0.49634013,  0.41713533, 0.64064113,  0.53215012,  0.395051  ,  0.54694894,  0.52539665, 0.5311468 ,  0.57121875,  0.55598972,  3.00706249,  3.01400019, 2.55269732,  3.14838857,  3.56461839,  2.84958162,  2.43466925,
        2.86693045,  4.0490198 ,  4.103727  ,  3.28267049,  3.49913446, 3.06540917,  3.42066746,  4.04069997,  4.78625417,  4.86068353, 5.0522888 ,  5.54170161,  4.97002748,  5.20517385,  5.99421298, 5.03756783,  5.41882892,  4.93390289,  3.99868588,  4.02308159,
        3.93546727,  3.83842355,  4.07779734,  3.16518537, 19.72781695, 15.95645752, 22.66682811, 23.35926131, 19.84517915, 20.35469212, 22.19251971, 20.2856508 , 23.22279421, 22.78419195, 12.78470736, 12.87313891, 13.28040096, 12.88065882, 21.10177732, 21.71419014,
        0.26929988,  0.34938891,  0.26901084,  0.4478692 ,  0.33224825, 0.29769651,  0.34160644,  0.35888538,  0.42970749,  0.51048819, 0.4430179 ,  0.19004174,  0.37715915,  0.40733123,  0.43156751, 0.39206143,  0.32521299])
    data = torch.Tensor( (data - train_mean)/train_std ) # Scaling, then converting to Tensor
    my_dataloader = DataLoader(data, batch_size, shuffle = False)

    # Model preparation
    if n_classes == 2:
        dict_path = '2class_chromatin_MLP_2seed_nonzero_diffMCC.pt'
    else:
        dict_path = '3class_chromatin_MLP_0seed_nonzero_diffMCC.pt'
    mydict = torch.load(dict_path, map_location = torch.device(device))
    layers = []
    for array_name, array in mydict.items(): # Getting the size of the model from mydict
        if 'weight' in array_name:
            layers.append(tuple(array.T.shape))
    # Building the model
    model = my_ANN(layers, 'relu')
    model.load_state_dict(mydict)
    model.to(device)
    model.eval()
    # Making predictions
    pred = torch.empty((len(my_dataloader.dataset), n_classes))
    for idx, data in enumerate(my_dataloader):
        data = data.to(device)
        temp_pred = model(data).cpu().detach()
        pred[idx*batch_size:(idx*batch_size)+len(temp_pred), :] = temp_pred
    pred = softmax(np.array(pred, dtype = float), axis = 1)
    pred[pred < 1e-20] = 0 # For convenience to eliminate very small numbers
    # Saving the predictions
    if n_classes == 2:
        pred_bool = pred[:, 1] >= threshold
        output = np.concatenate((list(range(pred.shape[0])), pred[:, 1], pred_bool)).reshape(-1, pred.shape[0]).T
        output = pd.DataFrame(output, columns = ['Sample Idx', 'Predicted Long-term Stability Chance', f'Chance >= {threshold}'])
    else:
        output = np.hstack((np.arange(pred.shape[0]).reshape(pred.shape[0], 1), pred))
        output = pd.DataFrame(output, columns = ['Sample Idx', 'Predicted Long-term Instability Chance', 'Predicted Long-term Stability Chance', 'Predicted Long-term High-Productivity Chance'])
    output.to_csv(''.join(data_file.split('.')[:-1]) + '_predictions.csv', index = False)

# MLP model
class my_ANN(torch.nn.Module):
    def __init__(self, layers, activ_fun = 'relu', lstm_input_size = 0, lstm_hidden_size = 0, device = 'cuda'):
        super(my_ANN, self).__init__()
        # Setup to convert string to activation function
        if activ_fun == 'relu':
            torch_activ_fun = torch.nn.ReLU()
        elif activ_fun == 'tanh':
            torch_activ_fun = torch.nn.Tanh()
        elif activ_fun == 'sigmoid':
            torch_activ_fun = torch.nn.Sigmoid()
        elif activ_fun == 'tanhshrink':
            torch_activ_fun = torch.nn.Tanhshrink()
        elif activ_fun == 'selu':
            torch_activ_fun = torch.nn.SELU()
        else:
            raise ValueError(f'Invalid activ_fun. You passed {activ_fun}')
        # Transforming layers list into OrderedDict with layers + activation
        mylist = list()
        for idx, elem in enumerate(layers):
            mylist.append((f'Linear{idx}', torch.nn.Linear(layers[idx][0], layers[idx][1]) ))
            if idx < len(layers)-1:
                mylist.append((f'{activ_fun}{idx}', torch_activ_fun))
        # OrderedDict into NN
        self.model = torch.nn.Sequential(OrderedDict(mylist))

    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    # Input setup
    import argparse
    parser = argparse.ArgumentParser(description = 'Loads a trained MLP model and predicts the long-term stability of CHO cells based on the levels of chromatin/histone modifications over time.')
    parser.add_argument('data_file', type = str, nargs = 1, help = 'An Nx97 .csv file, without any headers, containing the levels of relevant chromatin/histone modifications over time.')
    parser.add_argument('-t', '--threshold', metavar = '0.5', nargs = 1, type = float, default = [0.5], help = 'The minimum prediction threshold for a sample to be considered stable. Not relevant when n_classes == 3. Optional, default = 0.5.')
    parser.add_argument('-nc', '--n_classes', metavar = '2', nargs = 1, type = int, default = [2], help = 'Whether to predict with 2 classes (stable or unstable) or with 3 classes (highly-productive, stable, or unstable). Optional, default = 2.')
    parser.add_argument('-bs', '--batch_size', metavar = '2048', nargs = 1, type = int, default = [2048], help = 'The number of predictions done at a time. Lower only if getting out of memory errors. Optional, default = 2048.')
    args = parser.parse_args()
    data_file = args.data_file[0] # [0] to convert from list to string
    threshold = args.threshold[0]
    n_classes = args.n_classes[0]
    if n_classes not in {2, 3}:
        raise ValueError(f'n_classes should be 2 or 3, but you input {n_classes}')
    batch_size = args.batch_size[0]
    predict_CHO_stability(data_file, threshold, n_classes, batch_size)
