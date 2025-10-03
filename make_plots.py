from results_thresholds import results_at_thresholds
import numpy as np
import matplotlib.pyplot as plt
# Plotting settings
plt.rcParams.update({'font.size': 24, 'lines.markersize': 10})

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description = 'Makes plots used in the CHO stability paper.')
    parser.add_argument('-c', '--comparison', metavar='True | [False]', type=bool, nargs='?', default = False, const = True, choices = {True, False},
        help='Whether to make the "comparison between chromatin and all data" plot or the per-data type plots.')
    parser.add_argument('-b', '--betas', metavar = '', nargs = '+', type = float, default = [], help = 'The beta values used when calculating F_beta scores.')
    args = parser.parse_args()

    if args.comparison: # Plots comparing the F1 scores and MCCs of models trained with chromatin-only data vs. all data
        annotation_fontsize = 13
        # F1 comparison plot
        fig, ax = plt.subplots(figsize = (16, 9), dpi = 500)
        ax.errorbar(range(7), [49.6, 49.9, 53.8, 46.7, 47.7, 56.4, 59.1], [0, 0.3, 0.2, 0.5, 1.7, 1.3, 0.9], color = 'blue', fmt = 'o', capsize = 4)
        ax.errorbar(range(7), [49.2, 49.5, 54.4, 43.3, 49.6, 56.6, 59.5], [0, 0.3, 0.7, 2.3, 3.7, 0  , 0  ], color = 'red', fmt = 'o', capsize = 4)
        ax.hlines(51.9, -1, 7, color = 'blue', label = 'Chromatin only data')
        ax.annotate('Chromatin only data average', (6.5, 52.4), fontsize = annotation_fontsize, va = 'center', ha = 'right', color = 'blue')
        ax.hlines(51.7, -1, 7, color = 'red', label = 'Chromatin + methylation with WS = 20 data')
        ax.annotate('Chromatin + methylation with WS = 20 data average', (6.5, 51.2), fontsize = annotation_fontsize, va = 'center', ha = 'right', color = 'red')
        plt.plot([], [], ' ', label = 'Paired t-test p = 0.797') # Extra legend entry for the t-test p-value
        ax.set_ylabel(r'Averaged F$_1$ Score' + ' (%)')
        ax.set_xticks(range(7))
        ax.set_xticklabels(['LR', 'EN', 'LCEN', 'SVM', 'RF', 'MLP-CE', 'MLP-diffMCC'])
        ax.set_xlim(-0.5, 6.5)
        ax.legend(fontsize = 22)
        fig.tight_layout()
        fig.savefig(f'Chromatin_both-20window_comparison_F1.svg', bbox_inches = 'tight', dpi = 200)
        plt.close()
        # MCC comparison plot
        fig, ax = plt.subplots(figsize = (16, 9), dpi = 500)
        ax.errorbar(range(7), [7.37, 8.5 , 7.54,  5.01, 9.47, 17.1, 19.4], [0, 1.22, 0.48, 0.68,  0.2, 2.3, 1.1], color = 'blue', fmt = 'o', capsize = 4)
        ax.errorbar(range(7), [6.55, 7.83, 9.04, -0.13, 8.29, 17.9, 19.2], [0, 1.11, 1.24, 2.89, 2.69, 0  , 0  ], color = 'red', fmt = 'o', capsize = 4)
        ax.hlines(10.6, -1, 7, color = 'blue', label = 'Chromatin only data')
        ax.annotate('Chromatin only data average', (6.5, 11.1), fontsize = annotation_fontsize, va = 'center', ha = 'right', color = 'blue')
        ax.hlines(9.8, -1, 7, color = 'red', label = 'Chromatin + methylation with WS = 20 data')
        ax.annotate('Chromatin + methylation with WS = 20 data average', (6.5, 9.3), fontsize = annotation_fontsize, va = 'center', ha = 'right', color = 'red')
        plt.plot([], [], ' ', label = 'Paired t-test p = 0.349') # Extra legend entry for the t-test p-value
        ax.set_yticks(np.arange(0, 22, 2.5))
        ax.set_ylabel('MCC (%)')
        ax.set_xticks(range(7))
        ax.set_xticklabels(['LR', 'EN', 'LCEN', 'SVM', 'RF', 'MLP-CE', 'MLP-diffMCC'])
        ax.set_xlim(-0.5, 6.5)
        ax.legend(fontsize = 22)
        fig.tight_layout()
        fig.savefig(f'Chromatin_both-20window_comparison_MCC.svg', bbox_inches = 'tight', dpi = 200)
        plt.close()
    else: # F1, MCC, or F_beta plots at multiple thresholds
        thresholds = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
        n_classes = 2
        data_types_to_use = ['chromatin', 'both-20window_OHE'] if not args.betas else ['chromatin']
        for data_type in data_types_to_use:
            # F1 plot
            fig1, ax1 = plt.subplots(figsize = (16, 9), dpi = 500)
            ax1.set_xlabel('Minimum threshold for positive class classification')
            ax1.set_ylabel(r'Averaged F$_1$ score' + ' (%)')
            # MCC plot
            fig2, ax2 = plt.subplots(figsize = (16, 9), dpi = 500)
            ax2.set_xlabel('Minimum threshold for positive class classification')
            ax2.set_ylabel('MCC (%)')
            # F_betas plot
            fig3, ax3 = plt.subplots(figsize = (16, 9), dpi = 500)
            ax3.set_xlabel('Minimum threshold for positive class classification')
            ax3.set_ylabel(r'MLP-diffMCC averaged F$_\beta$ score' + ' (%)')
            for model in ['OLS', 'EN', 'LCEN', 'SVM', 'RF', 'MLP-CE', 'MLP-diffMCC']:
                if model == 'OLS':
                    F1, MCC = results_at_thresholds(f'{n_classes}class_{data_type}_{model}_nonzero.json', thresholds)
                    model = 'LR' # SPA calls it OLS internally, but I am calling it LR in the paper
                else:
                    F1 = np.zeros_like(thresholds)
                    MCC = np.zeros_like(thresholds)
                    F_beta = np.zeros((len(args.betas), len(thresholds)))
                    if 'MLP' in model and data_type != 'chromatin': # Currently, I have done only 1 run for the MLP models trained with both types of input data
                        n_repeats = 1
                    else:
                        n_repeats = 3 # How many runs with different seeds were done per data and model type
                    for seed in range(n_repeats):
                        if model == 'MLP-CE':
                            F1_temp, MCC_temp = results_at_thresholds(f'{n_classes}class_{data_type}_MLP_{seed}seed_nonzero.json', thresholds)
                        elif model == 'MLP-diffMCC' and args.betas:
                            F1_temp, MCC_temp, F_beta_temp = results_at_thresholds(f'{n_classes}class_{data_type}_MLP_{seed}seed_nonzero_diffMCC.json', thresholds, args.betas)
                            F_beta += F_beta_temp / n_repeats
                        elif model == 'MLP-diffMCC':
                            F1_temp, MCC_temp = results_at_thresholds(f'{n_classes}class_{data_type}_MLP_{seed}seed_nonzero_diffMCC.json', thresholds)
                        else:
                            F1_temp, MCC_temp = results_at_thresholds(f'{n_classes}class_{data_type}_{model}_{seed}seed_nonzero.json', thresholds)
                        F1 += F1_temp / n_repeats
                        MCC += MCC_temp / n_repeats
                ax1.plot(thresholds, F1*100, 'o-', label = model)
                ax2.plot(thresholds, MCC, 'o-', label = model) # MCC is already in %, so no need to multiply by 100
            for beta_idx, this_beta in enumerate(args.betas):
                ax3.plot(thresholds, F_beta[beta_idx]*100, 'o-', label = f'F$_{{{this_beta}}}$ Score (%)', markersize = 8)
            ax1.set_xlim(0, 1)
            ax1.set_ylim(18, 60)
            ax1.set_xticks(np.arange(0, 1.1, 0.1))
            ax1.set_xticklabels(np.round(np.arange(0, 1.1, 0.1), 2))
            ax1.hlines(40.476, 0, 0.5, colors = 'k', linestyles = 'dashed', label = 'Random guess line') # 40.476 is the f1_all at a threshold = 0
            ax1.vlines(0.5, 24.242, 40.476, colors = 'k', linestyles = 'dashed')
            ax1.hlines(24.242, 0.5, 1, colors = 'k', linestyles = 'dashed') # 24.242 is the f1_all at a threshold = 1
            ax1.legend(fontsize = 22, loc = 'lower left', bbox_to_anchor = (0, -0.01))
            fig1.tight_layout()
            # fig1.savefig(f'F1_score_{n_classes}class_{data_type}.svg', bbox_inches = 'tight')
            ax2.set_xlim(0, 1)
            ax2.set_xticks(np.arange(0, 1.1, 0.1))
            ax2.set_xticklabels(np.round(np.arange(0, 1.1, 0.1), 2))
            ax2.legend(fontsize = 22)
            fig2.tight_layout()
            # fig2.savefig(f'MCC_{n_classes}class_{data_type}.svg', bbox_inches = 'tight')
            if args.betas:
                ax3.set_xlim(0, 1)
                ax3.set_ylim(45, 62)
                ax3.set_xticks(np.arange(0, 1.1, 0.1))
                ax3.set_xticklabels(np.round(np.arange(0, 1.1, 0.1), 2))
                ax3.legend(fontsize = 22, loc = 'lower center')#, bbox_to_anchor = (0, -0.01))
                fig3.tight_layout()
                fig3.savefig(f'F_beta_score_{n_classes}class_{data_type}.svg', bbox_inches = 'tight')
            # Positive F_betas at threshold = 1 bar chart
            if data_type == 'chromatin': # This plot has been made only for the chromatin-only data
                fig4, ax4 = plt.subplots(figsize = (16, 9), dpi = 500)
                ax4.set_ylabel(r'MLP-diffMCC positive-class F$_\beta$ score' + ' (%)\n at threshold = 0.99')
                ax4.bar(['F$_1$ Score', 'F$_{0.50}$ Score', 'F$_{0.333}$ Score', 'F$_{0.25}$ Score', 'F$_{0.20}$ Score', 'F$_{0.10}$ Score', 'F$_0$ Score\n(Precision)'], [48.93, 60.10, 66.25, 69.41, 71.17, 73.89, 74.91], yerr = [19.37, 11.16, 6.12, 3.42, 1.92, 0.99, 1.77], color = plt.rcParams['axes.prop_cycle'].by_key()['color'])
                fig4.tight_layout()
                fig4.savefig(f'F_beta_score_positive_0.99threshold_{n_classes}class_{data_type}.svg', bbox_inches = 'tight')
            plt.close()
