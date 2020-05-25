from enum import Enum
import pandas as pd
import random


class ModelType(Enum):
    logistic_regression = 1
    decision_tree = 2
    nomogram = 3


class PredictionModel(object):
    def __init__(self, model_data):
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

    def predict(self, x):
        pass

    def load_model_detail(self, model_detail):
        pass

    def get_params(self):
        pass

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
    def __init__(self, model_data):
        self._intercept = 0.0
        self._params = {}
        super(LogisticRegression, self).__init__(model_data)

    def load_model_detail(self, model_detail):
        if 'Intercept' not in model_detail:
            raise Exception('intercept not found in model detail')
        self._intercept = model_detail['Intercept']
        for k in model_detail:
            if k != 'Intercept':
                self._params[k] = model_detail[k]

    def get_params(self):
        return self._params

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

    def predict(self, x, threshold=0.5):
        probs = self.predict_prob(x)
        return [(1 if p >= threshold else 0) for p in probs]


class Imputator(object):
    def __init__(self):
        pass

    def impute(self, x, variables):
        pass


class DistributionImputator(Imputator):
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
                x.loc[:, v] = [d['median'] + DistributionImputator.iqr_random_value(d['median'], d['l25'], d['h25'])
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
        if r <= 0.5:
            return median - (median - l25) * random.random()
        else:
            return median + (h25 - median) * random.random()