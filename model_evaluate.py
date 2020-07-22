import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample
import numpy


def auc_roc_analysis(y_list, predicted_probs_list, gen_fig=True, labels=None, fig_title='',
                     cis=None,
                     output_file=None):
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve, auc
    results = []
    fig_data = []
    for idx in range(len(y_list)):
        results.append(roc_auc_score(y_list[idx], y_score=predicted_probs_list[idx]))
        if gen_fig:
            # Compute ROC curve and ROC area for each class
            fpr, tpr, _ = roc_curve(y_list[idx], predicted_probs_list[idx])
            roc_auc = auc(fpr, tpr)
            lw = 2
            label = 'ROC curve (area = %0.2f)' % roc_auc
            if labels is not None:
                if cis is None:
                    label = '%s: %0.2f' % (labels[idx], roc_auc)
                else:
                    label = '%s: %0.2f (%0.2f-%0.2f)' % (labels[idx], cis[idx][2], cis[idx][0], cis[idx][1])
            if idx == 0:
                plt.figure()
                plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.plot(fpr, tpr, lw=lw, label=label)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
    if gen_fig:
        plt.legend(loc="lower right")
        plt.title('%s Model ROC Curves' % fig_title)
        if output_file is not None:
            plt.savefig(output_file)
        plt.show()
    return results


def carlibration_analysis(y_list, predicted_probs_list, gen_fig=True, labels=None, fig_title='', output_file=None):
    from sklearn.calibration import calibration_curve
    from sklearn.linear_model import LinearRegression
    results = []
    for idx in range(len(y_list)):
        y = y_list[idx]
        predicted_probs = predicted_probs_list[idx]
        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y, predicted_probs, n_bins=15, strategy='uniform')
        if gen_fig:
            if idx == 0:
                plt.figure()
                plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
            label = 'calibration curve'
            if labels is not None:
                label = labels[idx]
            plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=label)
            plt.xlabel('predicted probabilities of events')
            plt.ylabel('observed probabilities of events')
            plt.title('%s Model Calibrations' % fig_title)
        reg = LinearRegression().fit(mean_predicted_value.reshape(-1, 1), fraction_of_positives)
        results.append({'slope': '{:.3f}'.format(reg.coef_[0]),
                        'calibration-in-large': '{:.3f}'.format(reg.intercept_)})

    if gen_fig:
        plt.legend(loc="lower right")
        if output_file is not None:
            plt.savefig(output_file)
        plt.show()
    return results


def clinical_usefulness(y_list, predicted_list, threshold=None):
    from sklearn.metrics import confusion_matrix
    results = []
    for idx in range(len(y_list)):
        y = y_list[idx]
        predicted = predicted_list[idx]
        if threshold is not None:
            predicted = [(1 if p >= threshold else 0) for p in predicted]
        tn, fp, fn, tp = confusion_matrix(y, predicted).ravel()
        ppv = 1.0 * tp / (tp + fp)
        sensitivity = 1.0 * tp / (tp + fn)
        f1 = 2 * ppv * sensitivity / (ppv + sensitivity)
        specificity = 1.0 * tn / (tn + fp)
        results.append({'ppv': '{:.3f}'.format(ppv),
                        'sensitivity': '{:.3f}'.format(sensitivity),
                        'f1-score': '{:.3f}'.format(f1),
                        'specificity': '{:.3f}'.format(specificity),
                        'npv': '{:.3f}'.format(1.0 * tn / (tn + fn)),
                        'predict rate': '{:.3f}'.format((tp+fp) / len(y_list[0])),
                        'predicted': (tp+fp),
                        'tp': tp,
                        'fp': fp,
                        'tn': tn,
                        'fn': fn})
    return results


def get_confidence_interval(results, alpha=0.95):
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, numpy.percentile(results, p))
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1.0, numpy.percentile(results, p))
    mean = numpy.average(results)
    return lower, upper, mean


def evaluate_pipeline(y_list, predicted_probs_list, model_names=None, threshold=0.5, figs=False, outcome='',
                      auc_fig_file=None, calibration_fig_file=None):
    n_iterations = 100
    total_results = None
    for i in range(n_iterations):
        # prepare train and test sets
        y, indices = resample(y_list[0], [idx for idx in range(len(y_list[0]))])
        results = auc_roc_analysis([[y_list[idx][i] for i in indices] for idx in range(len(y_list))],
                                   [[predicted_probs_list[idx][i] for i in indices]
                                    for idx in range(len(predicted_probs_list))],
                                   gen_fig=False)
        if total_results is None:
            total_results = numpy.array(results)
        else:
            total_results = numpy.vstack((total_results, numpy.array(results)))
    cis = [get_confidence_interval(total_results[:, c]) for c in range(total_results.shape[1])]

    if figs:
        auc_roc_analysis(y_list, predicted_probs_list,
                         gen_fig=figs,
                         labels=
                         [model_name if model_name is not None else ''
                          for model_name in model_names],
                         fig_title=outcome,
                         output_file=auc_fig_file,
                         cis=cis
                         )

    result = {
              'c-index': ['{:.3f} ({:.2f}-{:.2f})'.format(cis[idx][2], cis[idx][0], cis[idx][1])
                          for idx in range(len(cis))],
              'calibration': carlibration_analysis(y_list, predicted_probs_list,
                                                   gen_fig=figs,
                                                   labels=
                                                   [model_name if model_name is not None else ''
                                                    for model_name in model_names],
                                                   fig_title=outcome,
                                                   output_file=calibration_fig_file)
              }
    if threshold is None or not list == type(threshold):
        result['clinical_usefulness (threshold %s)' % threshold] = \
            clinical_usefulness(y_list, predicted_probs_list, threshold=threshold)
    else:
        for th in threshold:
            result['clinical_usefulness (threshold %s)' % th] = \
                clinical_usefulness(y_list, predicted_probs_list, threshold=th)
    return result


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
