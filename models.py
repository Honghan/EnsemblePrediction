import pandas as pd
import random


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
        super(DecisionTree, self).__init__(model_data)

    def load_model_detail(self, model_detail):
        if 'nodes' not in model_detail:
            raise Exception('decision tree model does not find nodes')
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
                    probs.append(ret['outcome'])
                    break
                else:
                    n = self._nodes[ret['id']]
        return probs


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
        super(ScoringModel, self).__init__(model_data)

    def load_model_detail(self, model_detail):
        for v in model_detail:
            self._params[v] = model_detail[v]
            self._max_score += model_detail[v]['yes'] \
                if model_detail[v]['yes'] > model_detail[v]['no'] else model_detail[v]['no']

    @property
    def max_score(self):
        return self._max_score

    def predict_prob(self, x):
        self.check_x(x)
        probs = []
        for idx, r in x.iterrows():
            score = 0
            for p in self._params:
                if p not in r:
                    raise Exception('variable [{0}] not found in [{1}]'.format(p, r))
                op = self._params[p]
                score += op['yes'] if ModelUtil.binary_operator(op, r[p]) else op['no']
            probs.append(1.0 * score / self.max_score)
        return probs


class NomogramModel(PredictionModel):
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
                for t in cal['map']:
                    if t['range'][0] <= val < t['range'][1]:
                        points += t['point']
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
        self._dist = dist

    def impute(self, x, variables):
        if not isinstance(x, pd.DataFrame):
            raise Exception('the parameter x needs to be a pandas DataFrame instance')
        for v in variables:
            if v not in x.columns:
                if v not in self._dist:
                    raise Exception('{0} does not have a distribution data'.format(v))
                d = self._dist[v]
                if 'median' not in d:
                    raise Exception('only continuous variable imputations supported [{0}]'.format(d))
                x.loc[:, v] = [DistributionImputator.iqr_random_value(d['median'], d['l25'], d['h25'])
                               for idx in range(0, x.shape[0])]
        na_cols = x.columns[x.isna().any()].tolist()
        to_impute_cols = [v for v in variables if v in na_cols]
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