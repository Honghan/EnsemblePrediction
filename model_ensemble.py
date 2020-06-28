from enum import Enum
import models


class VoteMode(Enum):
    majority = 1
    one_vote_positive = 2
    one_vote_negative = 3
    average_score = 4
    max_score = 5


class Ensembler(object):
    """
    the abstract class for ensemble learning
    """
    def __init__(self):
        self._models = []
        self._model2weights = {}

    def add_model(self, prediction_model, weight=1):
        self._models.append(prediction_model)
        self._model2weights[prediction_model.id] = weight

    def adjust_severity_weight(self, outcome_severity, severity_conf):
        for m in self._models:
            self._model2weights[m.id] = severity_conf[m.outcome] / outcome_severity * self._model2weights[m.id]
            print('weight->', m.id, self._model2weights[m.id])

    def predict(self, x):
        models.PredictionModel.check_x(x)
        pass

    def predict_probs(self, x):
        pass

    @property
    def model2weight(self):
        return self._model2weights

    @property
    def models(self):
        return self._models


class BasicEnsembler(Ensembler):
    def __init__(self):
        self._mode = VoteMode.one_vote_positive
        super(BasicEnsembler, self).__init__()

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, v):
        self._mode = v

    def predict(self, x, threshold=0.5):
        preds = []
        for m in self.models:
            dist = m.model_data['cohort_variable_distribution']
            di = models.DistributionImputator(dist)
            imputed_x = di.impute(x, variables=[k for k in dist])
            if self.mode in [VoteMode.average_score, VoteMode.max_score]:
                preds.append(m.predict_prob(imputed_x))
            else:
                preds.append(m.predict(imputed_x, threshold=threshold))
        if self.mode == VoteMode.majority:
            return BasicEnsembler.majority(preds)
        elif self.mode == VoteMode.one_vote_positive:
            return BasicEnsembler.one_vote(preds, positive=True)
        elif self.mode == VoteMode.average_score:
            return BasicEnsembler.score_fuse(preds, threshold=threshold)
        elif self.mode == VoteMode.max_score:
            return BasicEnsembler.score_fuse(preds, max_fuse=True, threshold=threshold)
        else:
            return BasicEnsembler.one_vote(preds, positive=False)

    def predict_probs(self, x):
        """
        score
        :param x:
        :return:
        """
        if self.mode not in [VoteMode.average_score, VoteMode.max_score]:
            raise Exception('only average_score/max_score modes can predict probs')
        preds = []
        weights = []
        for m in self.models:
            dist = m.model_data['cohort_variable_distribution']
            di = models.DistributionImputator(dist)
            imputed_x = di.impute(x, variables=[k for k in dist])
            preds.append(m.predict_prob(imputed_x))
            weights.append(self.model2weight[m.id])
        use_max = self.mode == VoteMode.max_score
        return BasicEnsembler.score_fuse(preds, max_fuse=use_max, use_score=True, weights=weights)

    @staticmethod
    def majority(preds):
        """
        majority fusion
        :param preds:
        :return:
        """
        fused = []
        n = len(preds)
        for idx in range(len(preds[0])):
            voted = 0
            for pds in preds:
                if pds[idx] == 1:
                   voted += 1
            if voted > n/2:
                fused.append(1)
            else:
                fused.append(0)
        return fused

    @staticmethod
    def one_vote(preds, positive=True):
        """
        one vote fusion
        :param preds:
        :param positive:
        :return:
        """
        fused = []
        for idx in range(len(preds[0])):
            p = 0 if positive else 1
            for pds in preds:
                v = 1 if positive else 0
                if pds[idx] == v:
                    p = v
                    break
            fused.append(p)
        return fused

    @staticmethod
    def score_fuse(preds, threshold=0.5, max_fuse=False, use_score=False, weights=None):
        """
        score based fusion
        :param preds:
        :param threshold:
        :param max_fuse:
        :param use_score:
        :param weights:
        :return:
        """
        fused = []
        n = len(preds)
        for idx in range(len(preds[0])):
            s_sum = 0
            scores = []
            for j in range(len(preds)):
                pds = preds[j]
                score = pds[idx] * (1 if weights is None else weights[j])
                s_sum += score
                scores.append(score)
            s = s_sum * 1.0 / (n if weights is None else sum(weights))
            if max_fuse:
                s = max(scores)
            if use_score:
                fused.append(s)
            else:
                if s >= threshold:
                    fused.append(1)
                else:
                    fused.append(0)
        return fused