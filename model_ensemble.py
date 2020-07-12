from enum import Enum
import models
from fusion_strategies import BasicFusion, CompetenceFusion


class VoteMode(Enum):
    """
    vote mode enumerate
    """
    majority = 1
    one_vote_positive = 2
    one_vote_negative = 3
    average_score = 4
    max_score = 5
    competence_fusion = 6
    most_competence = 7
    highest_in_top_competences = 8


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
    """
    a basic ensembler implemented a set of fusion strategies
    """
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

    def get_competence_assessor(self):
        """
        get competence assessor
        :return:
        """
        if self._competence_assessor is None:
            self._competence_assessor = DistBasedAssessor(self.models)
        return self._competence_assessor

    def predict(self, x, threshold=0.5):
        """
        predict binary output
        :param x: the data frame
        :param threshold: threshold for prob cut-off in models predicting probabilities
        :return:
        """
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
            return BasicFusion.majority(preds)
        elif self.mode == VoteMode.one_vote_positive:
            return BasicFusion.one_vote(preds, positive=True)
        elif self.mode == VoteMode.average_score:
            return BasicFusion.score_fuse(preds, threshold=threshold)
        elif self.mode == VoteMode.max_score:
            return BasicFusion.score_fuse(preds, max_fuse=True, threshold=threshold)
        elif self.mode == VoteMode.competence_fusion:
            return CompetenceFusion.score_fuse_by_competence(x, preds, self.models,
                                                             self.get_competence_assessor(), threshold=threshold)
        else:
            return BasicFusion.one_vote(preds, positive=False)

    def predict_probs(self, x):
        """
        predict probabilities
        :param x: the data frame of data to be predicted
        :return:
        """
        if self.mode not in [VoteMode.average_score, VoteMode.max_score, VoteMode.most_competence,
                             VoteMode.competence_fusion, VoteMode.highest_in_top_competences]:
            raise Exception('only certain modes can predict probs')
        preds = []
        weights = []
        for m in self.models:
            dist = m.model_data['cohort_variable_distribution']
            di = models.DistributionImputator(dist)
            imputed_x = di.impute(x, variables=[k for k in dist])
            preds.append(m.predict_prob(imputed_x))
            weights.append(self.model2weight[m.id])
        use_max = self.mode == VoteMode.max_score
        if self.mode == VoteMode.competence_fusion:
            return CompetenceFusion.score_fuse_by_competence(x, preds, self.models,
                                                             self.get_competence_assessor(),
                                                             weight_by_competence=True,
                                                             default_weights=weights)
        elif self.mode == VoteMode.most_competence:
            return CompetenceFusion.predict_by_most_competent(x, preds, self.models,
                                                              self.get_competence_assessor())
        elif self.mode == VoteMode.highest_in_top_competences:
            return CompetenceFusion.predict_by_highest_risk(x, preds, self.models,
                                                            self.get_competence_assessor())
        return BasicFusion.score_fuse(preds, max_fuse=use_max, use_score=True, weights=weights)


class PredictorCompetenceAssessor(object):
    """
    an abstract competence assessor class
    """
    def __init__(self, models):
        self._models = models
        self._default_model_index = None

    def default_selection(self):
        if self._default_model_index is None:
            lst = [(idx, self._models[idx].model_data['provenance']['derivation_cohort']['N'])
                   for idx in range(len(self._models))]
            lst = sorted(lst, key=lambda t:-t[1])
            self._default_model_index = lst[0][0]
        return self._default_model_index, lst

    def evaluate(self, model, x):
        pass


class DistBasedAssessor(PredictorCompetenceAssessor):
    """
    a competence assessor using distribution of deriviation cohort
    """
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
