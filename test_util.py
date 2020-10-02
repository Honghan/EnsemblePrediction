import utils
from models import LogisticRegression, DistributionImputator, DecisionTree, ScoringModel, NomogramModel, NOCOS, BinaryImputator
import data_utils as du
import model_ensemble as me
import numpy as np
import model_evaluate as eval
import logging
from os import listdir
from os.path import isfile, join
import pandas as pd


def load_model(json_file):
    model_data = utils.load_json_data(json_file)
    if model_data['model_type'] == 'logistic_regression':
        model = LogisticRegression(model_data)
    elif model_data['model_type'] == 'decision_tree':
        model = DecisionTree(model_data)
    elif model_data['model_type'] == 'scoring':
        model = ScoringModel(model_data)
    elif model_data['model_type'] == 'nomogram':
        model = NomogramModel(model_data)
    elif model_data['model_type'] == 'NOCOS':
        model = NOCOS(model_data)
    else:
        raise Exception('model type [{0}] not recognised'.format(model_data['model_type']))
    logging.info('{0} loaded as a {1} model'.format(json_file, model.model_type))
    return model


def test_ensemble(model, x, outcome='death', threshold=0.5, severity_conf=None, generate_figs=False):
    """
    test ensemble method
    :param model:
    :param x:
    :param outcome:
    :param threshold:
    :param severity_conf: default None. severity configuration in the form of a dictionary -
    {'death': 1.0, 'poor_prognoses': 0.7}
    :param generate_figs: generate figs or not
    :return:
    """
    x = x.loc[x[outcome].notna()].copy()
    y = x[outcome].to_list()
    if severity_conf is not None:
        model.adjust_severity_weight(severity_conf[outcome], severity_conf)
    if model.mode in [me.VoteMode.average_score, me.VoteMode.max_score, me.VoteMode.most_competence,
                      me.VoteMode.competence_fusion, me.VoteMode.highest_in_top_competences]:
        probs = model.predict_probs(x)
        return y, probs
        # result = eval.evaluate_pipeline(y, probs, model_name='ensemble model', threshold=threshold,
        #                                 figs=generate_figs, outcome=outcome)
        # return result
    else:
        return None, None


def test_single_model(model, x, outcome=None, threshold=0.5):
    """
    test a single model
    :param model:
    :param x:
    :param outcome:
    :param threshold:
    :return:
    """
    if outcome is None:
        outcome = model.outcome
    x = x.loc[x[outcome].notna()].copy()
    dist = model.model_data['cohort_variable_distribution']
    di = DistributionImputator(dist)
    x = di.impute(x, variables=[k for k in dist])
    predicted_probs = np.array(model.predict_prob(x))
    y = x[outcome].to_list()
    return y, predicted_probs


def test_models_and_ensemble(model_files, x, weights=None, outcome='death', threshold=0.5, result_csv=None,
                             severity_conf=None, generate_figs=False, auc_fig_file=None,
                             calibration_fig_file=None, event_rate=None, nri_json=None):
    """
    do tests on individual models and also ensemble methods
    :param event_rate:
    :param model_files:
    :param x:
    :param weights:
    :param outcome:
    :param threshold:
    :param result_csv:
    :param severity_conf: severity configuration for setting weights on the alignments between model
    outcomes and what to predict
    :param generate_figs: generate figs or not
    :param auc_fig_file: roc curve figure output file
    :param calibration_fig_file: calibration figure output file
    :return:
    """
    data = {}
    ve = me.BasicEnsembler()
    y_list = []
    predicted_list = []
    models = []
    for idx in range(len(model_files)):
        mf = model_files[idx]
        m = load_model(mf)
        models.append(m)
        y, pred = test_single_model(m, x, outcome=outcome, threshold=threshold)
        y_list.append(y)
        predicted_list.append(pred)
        ve.add_model(m, 1 if weights is None else weights[idx])
        # results['{0}\n({1})'.format(m.id, m.model_type)] = result

    ve.mode = me.VoteMode.competence_fusion
    y, pred = test_ensemble(ve, x, threshold=threshold, outcome=outcome, severity_conf=severity_conf,
                            generate_figs=generate_figs)
    y_list.append(y)
    predicted_list.append(pred)
    results, nri_result = eval.evaluate_pipeline(y_list, predicted_list, model_names=[m.id for m in models] + ['ensemble model'],
                                     threshold=threshold,
                                     figs=generate_figs, outcome=outcome, auc_fig_file=auc_fig_file,
                                     calibration_fig_file=calibration_fig_file,
                                     event_rate=event_rate)
    model_labels = ['{0}\n({1})'.format(m.id, m.model_type) for m in models] + ['ensemble model']
    for idx in range(len(model_labels)):
        data[model_labels[idx]] = {}
        for k in results:
            data[model_labels[idx]][k] = results[k][idx]
    result_df = eval.format_result(data)
    if result_csv is not None:
        result_df.to_csv(result_csv, sep='\t', index=False)
    if nri_json is not None:
        utils.save_json_array(nri_result, nri_json)


def populate_col_by_or(x, cols, new_col_name):
    cm = []
    for idx, r in x.iterrows():
        v = 0
        for c in cols:
            if r[c] == 1:
                v = 1
                break
        cm.append(v)
    x[new_col_name] = cm
    return x


def do_test(config_file):
    """
    do the tests by using configuration file
    :param config_file:
    :return:
    """
    config = utils.load_json_data(config_file)
    partial_to_saturation_col = None if 'partial_to_saturation_col' not in config \
        else config['partial_to_saturation_col']
    x = du.read_data(config['data_file'],
                     sep='\t' if 'sep' not in config else config['sep'],
                     column_mapping=config['mapping'],
                     partial_to_saturation_col=partial_to_saturation_col,
                     hdf=False if 'hdf' not in config else config['hdf'])
    if 'binary_columns_to_impute' in config:
        # impute binary columns
        logging.info('binary columns to impute [%s]' % config['binary_columns_to_impute'])
        imputer = BinaryImputator()
        x = imputer.impute(x, config['binary_columns_to_impute'])
    if 'comorbidity_cols' in config:
        populate_col_by_or(x, config['comorbidity_cols'], new_col_name='comorbidity')
    model_files = config['model_files']
    for outcome in config['outcomes']:
        logging.info('testing for outcome [{0}] with #{1} models'.format(outcome, len(model_files)))
        result_file = '{0}/{1}_result.tsv'.format(config['result_tsv_folder'], outcome)
        test_models_and_ensemble(model_files,
                                 x,
                                 weights=config['weights'][outcome] if 'weights' in config else None,
                                 outcome=outcome,
                                 threshold=config['threshold'],
                                 result_csv=result_file,
                                 severity_conf=None if 'severity_scores' not in config else config['severity_scores'],
                                 generate_figs=False if 'generate_figs' not in config else config['generate_figs'],
                                 auc_fig_file=None if 'auc_fig_file_pattern' not in config
                                 else (config['auc_fig_file_pattern'] % outcome),
                                 calibration_fig_file=None if 'calibration_fig_file_pattern' not in config
                                 else (config['calibration_fig_file_pattern'] % outcome),
                                 event_rate=None if 'event_rate' not in config else config['event_rate'],
                                 nri_json=None if 'nri_json' not in config else config['nri_json']
                                 )
        logging.info('result saved to {0}'.format(result_file))


def get_all_variables_from_models(model_folder, conf):
    files = [join(model_folder, f) for f in listdir(model_folder) if isfile(join(model_folder, f))]
    vars = set()
    models = []
    for f in files:
        m = load_model(f)
        models.append(m)
        for v in m.model_data['cohort_variable_distribution']:
            vars.add(v)
    print(vars)
    summarise_models(models, conf)


def summarise_models(models, conf):
    data = {'model': ['outcome', 'model type',
                      'derivation cohort',
                      'country', 'region',
                      'N', 'age', 'followup period', 'death ratio', 'poor prognosis ratio',
                      'Model features']}
    for sect in conf['sections']:
        data['model'].append(sect['section'])
        for v in sect['variables']:
            data['model'].append('  %s' % v)
    for m in models:
        cohort = m.model_data['provenance']['derivation_cohort']
        data[m.id] = [
            m.outcome,
            m.model_type,
            ' ',
            m.model_data['provenance']['Country'],
            m.model_data['provenance']['region'],
            cohort['N'],
            '%s [%s-%s]' % (cohort['age']['median'],
                            cohort['age']['l25'],
                            cohort['age']['h25']),
            '%s to %s' % (cohort['follow_start'],
                       cohort['follow_end']),
            '-' if 'death_count' not in cohort else
            '{:.2%}'.format(cohort['death_count'] / cohort['N']),
            '-' if 'severe_count' not in cohort else
            '{:.2%}'.format(cohort['severe_count'] / cohort['N']),
        ]
        data[m.id].append(' ')  # empty line for model features
        for sect in conf['sections']:
            data[m.id].append(' ')
            for v in sect['variables']:
                if v in m.model_data['cohort_variable_distribution']:
                    vd = m.model_data['cohort_variable_distribution'][v]
                    if 'median' in vd:
                        data[m.id].append('%s [%s, %s]' % (vd['median'], vd['l25'], vd['h25']))
                    else:
                        data[m.id].append('x')
                else:
                    data[m.id].append(' ')
    df = pd.DataFrame(data)
    df.to_csv(conf['output_file'], index=False, sep='\t')
    logging.info('saved to %s' % conf['output_file'])


if __name__ == "__main__":
    utils.setup_basic_logging(log_level='INFO', file='ensemble.log')
    # do_test('./test/test_config_kl-wuhan.json')
    do_test('./test/test_config.json')
    # get_all_variables_from_models('./models', utils.load_json_data('./test/model_sum_conf.json'))
