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
                    label = '%s: %0.3f (%0.3f-%0.3f)' % (labels[idx], roc_auc, cis[idx][0], cis[idx][1])
            if idx == 0:
                plt.figure()
                plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.plot(fpr, tpr, lw=lw, label=label)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('1 - Specificity', fontsize=16)
            plt.ylabel('Sensitivity', fontsize=16)
    if gen_fig:
        plt.legend(loc="lower right", fontsize=12)
        plt.title('Outcome: %s' % fig_title)
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
        results.append({'slope': reg.coef_[0],
                        'calibration-in-large': reg.intercept_})

    if gen_fig:
        plt.legend(loc="lower right")
        if output_file is not None:
            plt.savefig(output_file)
        plt.show()
    return results


def clinical_usefulness(y_list, predicted_list, threshold=None, event_rate=None, idx_to_compare=0):
    from sklearn.metrics import confusion_matrix
    results = []
    pred_orig = None
    pred_improved = None
    for idx in range(len(y_list)):
        y = y_list[idx]
        predicted = predicted_list[idx]
        close_rate = None
        if event_rate is not None:
            num_expected = event_rate * len(y)
            min_diff = 50000
            closest_predicted = None
            for t in range(10, 500):
                th = (1.0 * t / 1000)
                predicted_binary = [(1 if p >= th else 0) for p in predicted]
                num_predicted = sum(predicted_binary)
                cur_diff = abs(num_expected - num_predicted)
                if cur_diff <= min_diff:
                    closest_predicted = predicted_binary
                    if cur_diff < min_diff:
                        # only use the first cut-off that leads to the prediction rate
                        threshold = th
                    min_diff = cur_diff
                else:
                    break
            close_rate = (1 - min_diff / num_expected) if min_diff < num_expected else 0
            predicted = closest_predicted
        else:
            if threshold is not None:
                predicted = [(1 if p >= threshold else 0) for p in predicted]
        if idx == idx_to_compare:
            pred_orig = predicted
        if idx == len(y_list) - 1:
            pred_improved = predicted
        tn, fp, fn, tp = confusion_matrix(y, predicted).ravel()
        ppv = 1.0 * tp / (tp + fp)
        sensitivity = 1.0 * tp / (tp + fn)
        f1 = 2 * ppv * sensitivity / (ppv + sensitivity)
        specificity = 1.0 * tn / (tn + fp)
        result = {'ppv': '{:.3f}'.format(ppv),
                  'sensitivity': '{:.3f}'.format(sensitivity),
                  'f1-score': '{:.3f}'.format(f1),
                  'specificity': '{:.3f}'.format(specificity),
                  'npv': '{:.3f}'.format(1.0 * tn / (tn + fn)),
                  'predict rate': '{:.3f}'.format((tp+fp) / len(y_list[0])),
                  'threshold': threshold,
                  'predicted': (tp+fp),
                  'tp': tp,
                  'fp': fp,
                  'tn': tn,
                  'fn': fn}
        if close_rate is not None:
            result['close_rate'] = close_rate
        results.append(result)
    nri = net_reclassification_improvement(y_list[0], pred_orig, pred_improved)
    print(nri)
    return results, nri


def net_reclassification_improvement(y_list, y_orig, y_improved):
    nri_ret = {'e': {'+': 0, '-': 0, '=': 0, 'orig': sum(y_orig)}, 'ne': {'+': 0, '-': 0, '=': 0, 'orig': len(y_orig) - sum(y_orig)}}
    for idx in range(len(y_list)):
        yi = y_improved[idx]
        yo = y_orig[idx]
        if y_list[idx] == 0:
            if yi > yo:
                nri_ret['ne']['-'] += 1
            elif yi < yo:
                nri_ret['ne']['+'] += 1
            else:
                nri_ret['ne']['='] += 1
        else:
            if yi > yo:
                nri_ret['e']['+'] += 1
            elif yi < yo:
                nri_ret['e']['-'] += 1
            else:
                nri_ret['e']['='] += 1
    nri_ret['NRI_e'] = (nri_ret['e']['+'] - nri_ret['e']['-'])/nri_ret['e']['orig']
    nri_ret['NRI_ne'] = (nri_ret['ne']['+'] - nri_ret['ne']['-'])/nri_ret['ne']['orig']
    return nri_ret


def get_confidence_interval(results, alpha=0.95):
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, numpy.percentile(results, p))
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1.0, numpy.percentile(results, p))
    mean = numpy.average(results)
    return lower, upper, mean


def create_add_row2ndarray(total_results, row):
    if total_results is None:
        if type(row[0]) == dict:
            total_results = {}
            for k in row[0]:
                total_results[k] = numpy.array([r[k] for r in row])
        else:
            total_results = numpy.array(row)
    else:
        if type(row[0]) == dict:
            for k in row[0]:
                total_results[k] = numpy.vstack((total_results[k], numpy.array([r[k] for r in row])))
        else:
            total_results = numpy.vstack((total_results, numpy.array(row)))
    return total_results


def get_dict_results(cal_results):
    cal_cis = []
    kys = [k for k in cal_results]
    for c in range(cal_results[kys[0]].shape[1]):
        t = {}
        for k in kys:
            l, h, m = get_confidence_interval(cal_results[k][:, c])
            t[k] = '{:.3f} ({:.3f}-{:.3f})'.format(m, l, h)
        cal_cis.append(t)
    return cal_cis


def evaluate_pipeline(y_list, predicted_probs_list, model_names=None, threshold=0.5, figs=False, outcome='',
                      auc_fig_file=None, calibration_fig_file=None, event_rate=None):
    n_iterations = 500
    auc_total_results = None
    cal_results = None
    for i in range(n_iterations):
        # prepare train and test sets
        y, indices = resample(y_list[0], [idx for idx in range(len(y_list[0]))])
        results = auc_roc_analysis([[y_list[idx][i] for i in indices] for idx in range(len(y_list))],
                                   [[predicted_probs_list[idx][i] for i in indices]
                                    for idx in range(len(predicted_probs_list))],
                                   gen_fig=False)
        auc_total_results = create_add_row2ndarray(auc_total_results, results)
        results = carlibration_analysis(y_list, predicted_probs_list,
                                        gen_fig=False)
        cal_results = create_add_row2ndarray(cal_results, results)

    auc_cis = [get_confidence_interval(auc_total_results[:, c]) for c in range(auc_total_results.shape[1])]

    idx_to_compare = 0
    if figs:
        results = auc_roc_analysis(y_list, predicted_probs_list,
                                   gen_fig=figs,
                                   labels=
                                   [model_name if model_name is not None else ''
                                    for model_name in model_names],
                                   fig_title=outcome,
                                   output_file=auc_fig_file,
                                   cis=auc_cis
                                   )
        idx_ret = [(idx, results[idx]) for idx in range(len(results) - 1)]
        idx_ret = sorted(idx_ret, key=lambda x: - x[1])
        idx_to_compare = idx_ret[0][0]
        carlibration_analysis(y_list, predicted_probs_list,
                              gen_fig=figs,
                              labels=[model_name if model_name is not None else ''
                                      for model_name in model_names],
                              output_file=calibration_fig_file)

    result = {
              'c-index': ['{:.3f} ({:.2f}-{:.2f})'.format(auc_cis[idx][2], auc_cis[idx][0], auc_cis[idx][1])
                          for idx in range(len(auc_cis))],
              'calibration': get_dict_results(cal_results)
              }
    nri_result = None
    if event_rate is not None:
        result['clinical_usefulness (threshold %s)' % threshold], nri_result = \
            clinical_usefulness(y_list, predicted_probs_list, threshold=threshold, event_rate=event_rate,
                                idx_to_compare=idx_to_compare)
    elif threshold is None or not list == type(threshold):
        result['clinical_usefulness (threshold %s)' % threshold], nri_result = \
            clinical_usefulness(y_list, predicted_probs_list, threshold=threshold,
                                idx_to_compare=idx_to_compare)
    else:
        for th in threshold:
            result['clinical_usefulness (threshold %s)' % th], nri_result = \
                clinical_usefulness(y_list, predicted_probs_list, threshold=th,
                                    idx_to_compare=idx_to_compare)
    return result, nri_result


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
