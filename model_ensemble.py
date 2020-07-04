from enum import Enum
import models


class VoteMode(Enum):
    majority = 1
    one_vote_positive = 2
    one_vote_negative = 3
    average_score = 4
    max_score = 5
    competence_by_age = 6


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
        # normalise severity score
        m2s = {}
        max_score = 0
        for m in self._models:
            m2s[m.id] = severity_conf[m.outcome] / outcome_severity
            max_score = max(max_score, m2s[m.id])

        for m in self._models:
            self._model2weights[m.id] = (m2s[m.id] / max_score) * self._model2weights[m.id]
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
        self._competence_assessor = None
        super(BasicEnsembler, self).__init__()

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, v):
        self._mode = v

    def get_competence(self):
        if self._competence_assessor is None:
            self._competence_assessor = DistBasedAssessor(self.models)
        return self._competence_assessor

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
        elif self.mode == VoteMode.competence_by_age:
            return BasicEnsembler.score_fuse_by_competence(x, preds, self.models,
                                                           self.get_competence(), threshold=threshold)
        else:
            return BasicEnsembler.one_vote(preds, positive=False)

    def predict_probs(self, x):
        """
        score
        :param x:
        :return:
        """
        if self.mode not in [VoteMode.average_score, VoteMode.max_score, VoteMode.competence_by_age]:
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
        if self.mode == VoteMode.competence_by_age:
            return BasicEnsembler.score_fuse_by_competence(x, preds, self.models,
                                                           self.get_competence(),
                                                           weight_by_competence=True,
                                                           default_weights=weights)
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

    @staticmethod
    def score_fuse_by_competence(X, preds, models, competence_assessor, threshold=None,
                                 weight_by_competence=False, default_weights=None):
        new_preds = []
        for idx, r in X.iterrows():
            # competence_list = [(i, competence_assessor.evaluate(models[i], r)) for i in range(len(models))]
            # competence_list = sorted(competence_list, key=lambda cl: -cl[1])
            competence_list = BasicEnsembler.do_combined_competence_score(r, competence_assessor, models)
            if weight_by_competence:
                new_preds.append(BasicEnsembler.competence_weighted_fuse(preds, idx, competence_list,
                                                                         default_weights, threshold=threshold))
            else:
                new_preds.append(BasicEnsembler.select_most_competence(preds, idx, competence_list,
                                                                       competence_assessor,
                                                                       threshold=threshold))
        return new_preds

    @staticmethod
    def do_combined_competence_score(r, competence_assessor, models, largest_N=15000):
        max_dist_vars_num = 0
        model_scores = []
        for i in range(len(models)):
            m = models[i]
            vars = [v for v in m.model_data['cohort_variable_distribution']
                    if 'type' not in m.model_data['cohort_variable_distribution'][v]]
            max_dist_vars_num = max(max_dist_vars_num, len(vars))
            score = m.model_data['provenance']['derivation_cohort']['N'] / largest_N
            for v in vars:
                score += competence_assessor.evaluate(models[i], r, var=v)
            model_scores.append((i, score))
        for idx in range(len(model_scores)):
            model_scores[idx] = (model_scores[idx][0], model_scores[idx][1] / (max_dist_vars_num + 1))
        return sorted(model_scores, key=lambda cl: -cl[1])

    @staticmethod
    def competence_weighted_fuse(preds, index, competence_list, default_weights, threshold=None):
        use_default = True if competence_list[0][1] == 0 else False
        total_weight = 0
        total_pred = 0
        for cl in competence_list:
            w = (1.5 * cl[1] + default_weights[cl[0]]) if not use_default else default_weights[cl[0]]
            total_weight += w
            total_pred += w * preds[cl[0]][index]
        pred = total_pred / total_weight
        if threshold is not None:
            return 1 if pred >= threshold else 0
        else:
            return pred

    @staticmethod
    def select_most_competence(preds, index, competence_list, competence_assessor, threshold=None):
        model_idx = competence_list[0][0]
        if competence_list[0][1] == 0:
            model_idx = competence_assessor.default_selection()
        if threshold is not None:
            return 1 if preds[model_idx][index] >= threshold else 0
        else:
            return preds[model_idx][index]


class PredictorCompetenceAssessor(object):
    def __init__(self, models):
        self._models = models
        self._default_model_index = None

    def default_selection(self):
        if self._default_model_index is None:
            lst = [(idx, self._models[idx].model_data['provenance']['derivation_cohort']['N'])
                   for idx in range(len(self._models))]
            lst = sorted(lst, key=lambda t:-t[1])
            self._default_model_index = lst[0][0]
        return self._default_model_index

    def evaluate(self, model, x):
        pass


class DistBasedAssessor(PredictorCompetenceAssessor):
    def __init__(self, models):
        super().__init__(models=models)

    def evaluate(self, model, x, var='age'):
        val = x[var]
        dist = model.model_data['cohort_variable_distribution']
        m_var = dist[var]['median']
        var_range = dist[var]['h25'] - dist[var]['l25']
        delta = 0 if dist[var]['l25'] <= val <= dist[var]['h25'] else max( abs(val - m_var) / var_range, 1)
        # delta = abs(val - m_var) / var_range
        competence = (1 - delta) if delta <= 1 else 0
        return competence
