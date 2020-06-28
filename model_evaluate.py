import pandas as pd
import matplotlib.pyplot as plt


def auc_roc_analysis(y, predicted_probs, gen_fig=True, fig_title=None, output_file=None):
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve, auc
    c_index = roc_auc_score(y, y_score=predicted_probs)

    if gen_fig:
        # Compute ROC curve and ROC area for each class
        fpr, tpr, _ = roc_curve(y, predicted_probs)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        if fig_title is not None:
            plt.title(fig_title)
        plt.legend(loc="lower right")
        plt.show()
        if output_file is not None:
            plt.savefig(output_file)
    return c_index


def carlibration_analysis(y, predicted_probs, gen_fig=True, fig_title=None, output_file=None):
    from sklearn.calibration import calibration_curve
    from sklearn.linear_model import LinearRegression
    fraction_of_positives, mean_predicted_value = \
        calibration_curve(y, predicted_probs, n_bins=20, strategy='uniform')
    if gen_fig:
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.plot(mean_predicted_value, fraction_of_positives, "s-")
        plt.xlabel('predicted probabilities of events')
        plt.ylabel('observed probabilities of events')
        if fig_title is not None:
            plt.title(fig_title)
        plt.show()
        if output_file is not None:
            plt.savefig(output_file)

    reg = LinearRegression().fit(mean_predicted_value.reshape(-1, 1), fraction_of_positives)
    return {'slope': reg.coef_[0], 'calibration-in-large': reg.intercept_}


def clinical_usefulness(y, predicted):
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y, predicted).ravel()
    ppv = 1.0 * tp / (tp + fp)
    sensitivity = 1.0 * tp / (tp + fn)
    f1 = 2 * ppv * sensitivity / (ppv + sensitivity)
    return {'ppv': ppv,
            'sensitivity': sensitivity,
            'f1-score': f1,
            'specificity': 1.0 * tn / (tn + fp),
            'npv': 1.0 * tn / (tn + fn)}


def evaluate_pipeline(y, predicted_probs, model_name=None, threshold=0.5, figs=False):
    return {'c-index': auc_roc_analysis(y, predicted_probs,
                                        gen_fig=figs,
                                        fig_title='ROC curve' +
                                                  (' of ' + model_name if model_name is not None else '')),
            'calibration': carlibration_analysis(y, predicted_probs,
                                                 gen_fig=figs,
                                                 fig_title='calibration plot' +
                                                           (' of ' + model_name if model_name is not None else '')),
            'clinical_usefulness': clinical_usefulness(y, [(1 if p >= threshold else 0) for p in predicted_probs])}


def format_result(model2result):
    result_names = []
    results = []
    for n in model2result:
        result_names.append(n)
        results.append(model2result[n])
    title_col = None
    cols = []
    for r in results:
        col = []
        cols.append(col)
        populate_title = False if title_col is not None else True
        if populate_title:
            title_col = []
        add_dict_to_col(r, col, title_col=None if not populate_title else title_col)
    data = {'': title_col}
    for idx in range(len(result_names)):
        data[result_names[idx]] = cols[idx]
    return pd.DataFrame(data=data)


def add_dict_to_col(r, col, title_col=None):
    for k in r:
        if title_col is not None:
            title_col.append(k)
        if isinstance(r[k], dict):
            col.append('')
            add_dict_to_col(r[k], col, title_col)
        else:
            col.append(r[k])
