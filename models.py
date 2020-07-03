import pandas as pd
import random
from enum import Enum
import logging
import math


class PredictionModel(object):
    """
    abstract prediction model class
    """
    def __init__(self, model_data):
        self._params = {}
        self._model_meta = model_data
        self.load_model_detail(model_data['model_detail'])

    @property
    def id(self):
        return self._model_meta['id']

    @id.setter
    def id(self, value):
        self._model_meta['id'] = value

    @property
    def doi(self):
        return self._model_meta['doi']

    @property
    def outcome(self):
        return self._model_meta['outcome']

    @property
    def outcome(self):
        return self._model_meta['outcome']

    @property
    def model_type(self):
        return self._model_meta['model_type']

    @property
    def model_data(self):
        return self._model_meta

    def predict_prob(self, x):
        pass

    def predict(self, x, threshold=0.5):
        probs = self.predict_prob(x)
        return [(1 if p >= threshold else 0) for p in probs]

    def load_model_detail(self, model_detail):
        pass

    def get_params(self):
        return self._params

    def check_x(self, x):
        if not isinstance(x, pd.DataFrame):
            raise Exception('the parameter x needs to be a pandas DataFrame instance')
        missing_params = []
        for p in self.get_params():
            if p not in x.columns:
                missing_params.append(p)
        if len(missing_params) > 0:
            raise Exception('x does not have these variables: {0}'.format(missing_params))


class LogisticRegression(PredictionModel):
    """
    Logistic Regression model
    """
    def __init__(self, model_data):
        self._intercept = 0.0
        super(LogisticRegression, self).__init__(model_data)

    def load_model_detail(self, model_detail):
        if 'Intercept' not in model_detail:
            raise Exception('intercept not found in model detail')
        self._intercept = model_detail['Intercept']
        for k in model_detail:
            if k != 'Intercept':
                self._params[k] = model_detail[k]

    def predict_prob(self, x):
        from math import exp
        self.check_x(x)
        y = []
        cols = [p for p in self.get_params()]
        for idx, r in x.iterrows():
            g = self._intercept
            for c in cols:
                g += self._params[c] * r[c]
            y.append(1.0 / (1 + exp(-g)))
        return y


class DecisionTree(PredictionModel):
    def __init__(self, model_data):
        self._params = {}
        self._root = None
        self._nodes = {}
        self._prob_mode = DTProbModel.m_estimation
        self._event_rate = 0.5
        super(DecisionTree, self).__init__(model_data)

    @property
    def prob_mode(self):
        return self._prob_mode

    @prob_mode.setter
    def prob_model(self, v):
        self._prob_mode = v

    def load_model_detail(self, model_detail):
        if 'nodes' not in model_detail:
            raise Exception('decision tree model does not find nodes')
        self._event_rate = model_detail['event_rate']
        nodes = model_detail['nodes']
        for n in nodes:
            node = TreeNode(n)
            if n['id'] == 'root':
                self._root = node
            else:
                self._nodes[n['id']] = node
            self._params[node.variable] = node.id

    def predict_prob(self, x):
        self.check_x(x)
        probs = []
        for idx, r in x.iterrows():
            n = self._root
            while n is not None:
                ret = n.to(r)
                if 'outcome' in ret:
                    # print('outcome at {0}, {1}'.format(n.op, n.variable), idx, ret)
                    prob = ret['outcome']
                    if 'support' in ret:
                        prob = self.cal_prob(ret['outcome'], ret['support'])
                    probs.append(prob)
                    break
                else:
                    n = self._nodes[ret['id']]
        return probs

    def cal_prob(self, outcome, support):
        s_t = support['T']
        s_f = support['F']
        if outcome == 0:
            s_t = support['F']
            s_f = support['T']
        if self.prob_mode == DTProbModel.maximum_likelihood:
            return DecisionTree.maximum_likelihood(s_t, s_f)
        elif self.prob_mode == DTProbModel.laplace_estimate:
            return DecisionTree.laplace_estimate(s_t, s_f)
        elif self.prob_mode == DTProbModel.m_estimation:
            return DecisionTree.m_estimation(s_t, s_f, self._event_rate)

    @staticmethod
    def maximum_likelihood(case_support, control_support):
        return 1.0 * case_support / (case_support + control_support)

    @staticmethod
    def laplace_estimate(case_support, control_support):
        return 1.0 * (case_support + 1)/ (case_support + control_support + 2)

    @staticmethod
    def m_estimation(case_support, control_support, prior_prob):
        m = 10 / prior_prob
        return 1.0 * (case_support + m * prior_prob)/ (case_support + control_support + m)


class DTProbModel(Enum):
    maximum_likelihood = 1,
    laplace_estimate = 2,
    m_estimation = 3


class TreeNode(object):
    def __init__(self, node_data):
        self._id = node_data['id']
        self._var = node_data['variable']
        self._op = node_data['to']['condition']
        self._yes = node_data['to']['yes']
        self._no = node_data['to']['no']

    @property
    def id(self):
        return self._id

    @property
    def variable(self):
        return self._var

    @property
    def op(self):
        return self._op

    def compute(self, x):
        if self._var not in x:
            raise Exception('column [{0}] not found when doing node computing'.format(self._var))
        v = x[self._var]
        return ModelUtil.binary_operator(self._op, v)

    def to(self, x):
        if self.compute(x):
            return self._yes
        else:
            return self._no


class ModelUtil(object):
    @staticmethod
    def binary_operator(op, v):
        if op['op'] == 'less':
            return v < op['val']
        elif op['op'] == 'lesseq':
            return v <= op['val']
        elif op['op'] == 'greater':
            return v > op['val']
        elif op['op'] == 'greatereq':
            return v >= op['val']
        elif op['op'] == 'eq':
            return v == op['val']
        else:
            raise Exception("unknown operator [{0}]".format(op['op']))


class ScoringModel(PredictionModel):
    def __init__(self, model_data):
        self._params = {}
        self._max_score = 0
        self._score_to_prob = {}
        super(ScoringModel, self).__init__(model_data)

    def load_model_detail(self, model_detail):
        for v in model_detail['variables']:
            self._params[v] = model_detail['variables'][v]
            self._max_score += model_detail['variables'][v]['yes'] \
                if model_detail['variables'][v]['yes'] > model_detail['variables'][v]['no'] else model_detail['variables'][v]['no']
        for t in model_detail['score_probs']:
            self._score_to_prob[str(t[0])] = t[1]

    @property
    def max_score(self):
        return self._max_score

    def predict_prob(self, x):
        from math import exp
        self.check_x(x)
        probs = []
        for idx, r in x.iterrows():
            score = 0
            for p in self._params:
                if p not in r:
                    raise Exception('variable [{0}] not found in [{1}]'.format(p, r))
                op = self._params[p]
                score += op['yes'] if ModelUtil.binary_operator(op, r[p]) else op['no']
            probs.append(self.get_prob(score))
            # probs.append(1 / (1 + exp(-score)))
        return probs

    def get_prob(self, score):
        s = str(score)
        if s not in self._score_to_prob:
            logging.warn('score {0} not in score_to_prob data'.format(s))
            return 1.0 * score / self.max_score
        else:
            return self._score_to_prob[str(score)]


class NomogramModel(PredictionModel):
    """
    Nmogram prediction - essentially a sequence of linear functions
    """
    def __init__(self, model_data):
        self._params = {}
        self._unit_point_scale = [0, 100]
        self._total_point_scale = [0, 350]
        self._pp_mappings = None
        super(NomogramModel, self).__init__(model_data)

    def load_model_detail(self, model_detail):
        self._unit_point_scale = model_detail['unit_point_scale']
        self._total_point_scale = model_detail['total_point_scale']
        self._pp_mappings = sorted(model_detail['point-to-prediction-mappings'], key=lambda x: x['point'])
        for v in model_detail['variables']:
            self._params[v] = model_detail['variables'][v]

    def predict_prob(self, x):
        self.check_x(x)
        probs = []
        for idx, r in x.iterrows():
            point = self.calculate_points(r)
            probs.append(self.get_predict_prob_by_point(point))
        return probs

    def calculate_points(self, r):
        points = 0
        for v in self._params:
            if v not in r:
                raise Exception('variable [{0}] not found in {1}'.format(v, r))
            val = r[v]
            cal = self._params[v]
            if cal['type'] == 'discrete':
                matched = False
                last_point = 0
                for t in cal['map']:
                    if t['range'][0] <= val <= t['range'][1]:
                        points += t['point']
                        matched = True
                        break
                    else:
                        last_point = t['point']
                if not matched:
                    logging.info('{0} value {1} not matched, using nearest range point'.format(v, val))
                    points += last_point
            else:
                point_range = t['point']
                val_range = t['variable']
                if val > val_range[0]:
                    ratio = 1.0 * (val - val_range[0]) / (val_range[1] - val_range[0])
                    points += ratio * (point_range[1] - point_range[0]) + point_range[0]
        return points

    def get_predict_prob_by_point(self, point):
        prev = None
        next = None
        for t in self._pp_mappings:
            if point <= t['point']:
                next = t
                break
            prev = t
        if prev is None:
            if point == next['point']:
                return t['predict']
            else:
                return 0
        else:
            if next is not None:
                ratio = (point - prev['point']) / (next['point'] - prev['point'])
                return (next['predict'] - prev['predict']) * ratio  + prev['predict']
            else:
                return 1


class NOCOS(PredictionModel):
    """
    a function reimplemented from https://cbmi.northwell.edu/nocos/
    """
    def __init__(self, model_data):
        self._detail = {}
        super(NOCOS, self).__init__(model_data)

    def predict_prob(self, x):
        self.check_x(x)
        probs = []
        for idx, r in x.iterrows():
            probs.append(1 - NOCOS.covid19SurvivalProbabilityFormula_v4(r, self._detail['calculationData']))
        return probs

    def load_model_detail(self, model_detail):
        self._detail = model_detail

    @staticmethod
    def covid19SurvivalProbabilityFormula_v4(factorInput, calculationData):
        predictorData = calculationData['predictorData']
        bayesData = calculationData['bayesData']

        x = 0.0
        for pd in predictorData:
            if pd['id'] in factorInput:
                factorValue = float(factorInput[pd['id']])
            else:
                factorValue = 0.0
            ## ignore validate not convertable values as an error would be raised anyway
            # if factorValue is None:
            #     factorValue = 0.0

            sigma = 1.0 * pd['sigma']
            mu = 1.0 * pd['mu']
            coefficient = 1.0 * pd['coefficient']
            x = x + coefficient * ((factorValue - mu) / sigma)

        LPos = 0.0
        posteriorData = calculationData['paretoData']['posteriorPos']
        if x < posteriorData['threshold1']:
            LPos = NOCOS.lowerParetoTail(posteriorData['p1'], posteriorData['sigma1'], posteriorData['k1'], posteriorData['threshold1'], x)
        elif x > posteriorData['threshold2']:
            LPos = NOCOS.upperParetoTail(posteriorData['p2'], posteriorData['sigma2'], posteriorData['k2'], posteriorData['threshold2'], x)
        else:
            LPos = NOCOS.polynomial(posteriorData['coefficients'], x)

        LNeg = 0.0
        posteriorData = calculationData['paretoData']['posteriorNeg']
        if x < posteriorData['threshold1']:
            LNeg = NOCOS.lowerParetoTail(posteriorData['p1'], posteriorData['sigma1'], posteriorData['k1'], posteriorData['threshold1'], x)
        elif x > posteriorData['threshold2']:
            LNeg = NOCOS.upperParetoTail(posteriorData['p2'], posteriorData['sigma2'], posteriorData['k2'], posteriorData['threshold2'], x)
        else:
            LNeg = NOCOS.polynomial(posteriorData['coefficients'], x)
        return LPos * calculationData['paretoData']['priorPos'] / (LPos * calculationData['paretoData']['priorPos'] + LNeg * calculationData['paretoData']['priorNeg'])

    @staticmethod
    def lowerParetoTail(p, sigma, k, threshold, randomVar):
        return p * (1.0 / sigma) * math.pow((1.0 + k * (threshold - randomVar) / sigma), (-1.0 - (1.0 / k)))

    @staticmethod
    def upperParetoTail(p, sigma, k, threshold, randomVar):
        return (1.0 - p) * (1 / sigma) * math.pow((1.0 + k * (randomVar - threshold) / sigma), (-1.0 - (1.0 / k)))

    @staticmethod
    def polynomial(coefficients, randomVar):
        result = 0.0
        for i in range(len(coefficients)):
            result += coefficients[i] * (randomVar ** i)
        return result


class Imputator(object):
    """
    abstract imputation class
    """
    def __init__(self):
        pass

    def impute(self, x, variables):
        pass


class DistributionImputator(Imputator):
    """
    use median and iqr to impute data
    """
    def __init__(self, dist):
        super().__init__()
        self._dist = dist

    def impute(self, x_orig, variables):
        if not isinstance(x_orig, pd.DataFrame):
            raise Exception('the parameter x needs to be a pandas DataFrame instance')
        x = x_orig.copy()
        ignore_impute_vars = []
        for v in variables:
            if v not in x.columns:
                if v not in self._dist:
                    raise Exception('{0} does not have a distribution data'.format(v))
                d = self._dist[v]
                if 'type' in d and d['type'] == 'binary':
                    # don't do binary, they are put in for collecting model variable purpose only
                    ignore_impute_vars.append(v)
                    continue
                if 'median' not in d:
                    raise Exception('only continuous variable imputations supported [{0}]'.format(d))
                x.loc[:, v] = [DistributionImputator.iqr_random_value(d['median'], d['l25'], d['h25'])
                               for idx in range(0, x.shape[0])]
        na_cols = x.columns[x.isna().any()].tolist()
        to_impute_cols = [v for v in variables if v in na_cols and v not in ignore_impute_vars]
        if len(to_impute_cols) > 0:
            for idx, r in x.iterrows():
                for c in to_impute_cols:
                    if pd.isna(r[c]):
                        d = self._dist[c]
                        x.loc[idx, c] = \
                            DistributionImputator.iqr_random_value(d['median'], d['l25'], d['h25'])
        return x

    @staticmethod
    def iqr_random_value(median, l25, h25):
        r = random.random()
        if r <= 0.0:
            return median - (median - l25) * random.random()
        elif 0.0 < r <= 1:
            return median
        else:
            return median + (h25 - median) * random.random()