import json
import numpy as np
from sklearn.metrics import confusion_matrix

def results_at_thresholds(json_file, thresholds = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99], betas = []):
    # Loading the results and the test y data
    with open(json_file) as f:
        model_results = json.load(f)
    predictions = np.array(model_results[list(model_results.keys())[0]]['logits_test_normalized'])
    if 'methylation' in json_file:
        test_y = np.loadtxt('chromatin_data/test_data_2class.csv', comments = None, dtype = bool)
    else:
        test_y = np.loadtxt('chromatin_data/test_data_2class_nonzero.csv', comments = None, dtype = bool)
    # Other setup
    rec = np.empty_like(thresholds)
    pre = np.empty_like(thresholds)
    f1 = np.empty_like(thresholds)
    spec = np.empty_like(thresholds)
    npv = np.empty_like(thresholds)
    f1_negative = np.empty_like(thresholds)
    MCC = np.empty_like(thresholds)
    if betas:
        fbeta = np.empty((len(betas), len(thresholds)))
        fbeta_negative = np.empty_like(fbeta)
    for idx, threshold in enumerate(thresholds):
        above_cutoff = predictions[:, 1] > threshold
        CM = confusion_matrix(test_y, above_cutoff)
        # Positive-class metrics
        if CM[1,1]+CM[1,0] > 0:
            rec[idx] = CM[1,1]/(CM[1,1]+CM[1,0])
        else:
            rec[idx] = 0
        if CM[1,1]+CM[0,1] > 0:
            pre[idx] = CM[1,1]/(CM[1,1]+CM[0,1])
        else:
            pre[idx] = 0
        if rec[idx] and pre[idx]:
            f1[idx] = 2/(1/rec[idx] + 1/pre[idx])
            for beta_idx, this_beta in enumerate(betas):
                fbeta[beta_idx, idx] = (this_beta**2 + 1)/(this_beta**2/rec[idx] + 1/pre[idx])
        else:
            f1[idx] = 0
            for beta_idx, this_beta in enumerate(betas):
                fbeta[beta_idx, idx] = 0
        # Negative-class metrics
        if CM[0,0]+CM[0,1] > 0:
            spec[idx] = CM[0,0]/(CM[0,0]+CM[0,1])
        else:
            spec[idx] = 0
        if CM[0,0]+CM[1,0] > 0:
            npv[idx] = CM[0,0]/(CM[0,0]+CM[1,0])
        else:
            npv[idx] = 0
        if spec[idx] and npv[idx]:
            f1_negative[idx] = 2/(1/spec[idx] + 1/npv[idx])
            for beta_idx, this_beta in enumerate(betas):
                fbeta_negative[beta_idx, idx] = (this_beta**2 + 1)/(this_beta**2/spec[idx] + 1/npv[idx])
        else:
            f1_negative[idx] = 0
            for beta_idx, this_beta in enumerate(betas):
                fbeta_negative[beta_idx, idx] = 0
        f1_all = (f1 + f1_negative)/2
        if betas:
            fbeta_all = (fbeta + fbeta_negative)/2
        CM_float = np.array(CM, dtype = float) # To avoid integer overflows
        MCC[idx] = 100*(CM_float[1,1]*CM_float[0,0] - CM_float[0,1]*CM_float[1,0]) / np.sqrt(
            (CM_float[1,1]+CM_float[0,1]) * (CM_float[1,1]+CM_float[1,0]) * (CM_float[0,0]+CM_float[0,1]) * (CM_float[0,0]+CM_float[1,0]) + 1e-8)
    best_threshold = np.argmax(f1_all)
    if __name__ == '__main__':
        print(f'Best F1 average: {np.max(f1_all)*100:.2f} at a threshold of {thresholds[best_threshold]}')
        print(f'MCC at that threshold: {MCC[best_threshold]:.2f}')
        if betas:
            best_F_betas = np.max(fbeta_all, axis = 1)
            best_F_beta_thresholds = np.argmax(fbeta_all, axis = 1)
            for beta_idx, this_beta in enumerate(betas):
                print(f'Best F_{this_beta} average: {best_F_betas[beta_idx]*100:.2f} at a threshold of {thresholds[best_F_beta_thresholds[beta_idx]]}')
    elif betas:
        return f1_all, MCC, fbeta_all
    else:
        return f1_all, MCC
    # np.set_printoptions(linewidth = 500)
    # print(f'The F_beta all is {fbeta_all[:, 6]}')
    # print(f'The positive class F_1 is {f1[-1]}')
    # print(f'The positive class F_beta is {fbeta[:, -1]}')
    # print(f'The precision is {pre[-1]}')
    # breakpoint()
    # print(np.round(f1_all*100, 2))
    # print(np.round(fbeta_all*100, 2))
    # print(np.round(MCC, 2))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description = 'Loads the results of a trained model from its .json file and calculates test-set metrics at different thresholds.')
    parser.add_argument('json_file', type = str, nargs = 1, help = 'The .json file for the trained model.')
    parser.add_argument('-t', '--thresholds', metavar = '', nargs = '+', type = float, default = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99], help = 'The minimum prediction threshold for a sample to be considered stable. Optional, default = list with multiple values.')
    parser.add_argument('-b', '--betas', metavar = '', nargs = '+', type = float, default = [], help = 'The beta values used when calculating F_beta scores. No need to include 1, as the F_1 score is always calculated.')
    args = parser.parse_args()
    results_at_thresholds(args.json_file[0], args.thresholds, args.betas) # [0] to convert from list to string
