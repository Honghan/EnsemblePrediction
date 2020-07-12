

class BasicFusion(object):
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


class CompetenceFusion(object):
    @staticmethod
    def do_competence_fusion(X, preds, models, competence_assessor, threshold=None,
                             fusion_func=None, kargs=None):
        if kargs is None:
            kargs = {}
        new_preds = []
        for idx, r in X.iterrows():
            # competence_list = [(i, competence_assessor.evaluate(models[i], r)) for i in range(len(models))]
            # competence_list = sorted(competence_list, key=lambda cl: -cl[1])
            competence_list = CompetenceFusion.do_combined_competence_score(r, competence_assessor, models)
            if fusion_func is not None:
                kargs.update({'preds': preds, 'index': idx, 'competence_list': competence_list,
                              'competence_assessor': competence_assessor, 'threshold': threshold})
                new_preds.append(fusion_func(**kargs))
        return new_preds

    @staticmethod
    def score_fuse_by_competence(X, preds, models, competence_assessor, threshold=None,
                                 weight_by_competence=False, default_weights=None):
        return CompetenceFusion.do_competence_fusion(X, preds, models, competence_assessor,
                                                     threshold=threshold,
                                                     fusion_func=CompetenceFusion.competence_weighted_fuse,
                                                     kargs={'default_weights': default_weights})

    @staticmethod
    def predict_by_most_competent(X, preds, models, competence_assessor, threshold=None):
        return CompetenceFusion.do_competence_fusion(X, preds, models, competence_assessor,
                                                     threshold=threshold,
                                                     fusion_func=CompetenceFusion.select_most_competence)

    @staticmethod
    def predict_by_highest_risk(X, preds, models, competence_assessor, threshold=None, top_k=5):
        return CompetenceFusion.do_competence_fusion(X, preds, models, competence_assessor,
                                                     threshold=threshold,
                                                     fusion_func=CompetenceFusion.use_the_highest_risk,
                                                     kargs={'top_k': top_k})

    @staticmethod
    def do_combined_competence_score(r, competence_assessor, models, largest_N=15000):
        """
        calculate competence of a model on all numeric variables. For each variable, the competence is calculated
         based on a positively correlated function based on the following.
        - the distance between the value of the variable to the distribution (median/IQR) of the variable
        in the derivation cohort.
        :param r: data row - a pandas series object
        :param competence_assessor: the competence assessor instance
        :param models: the list of models to be assessed
        :param largest_N: the number to be used as denominator in calculate the cohort size based competence of
        a model.
        :return: the ordered list (from largest to smallest competences) of tuples:
        [(model_index, competence_value)]
        """
        max_dist_vars_num = 0
        model_scores = []
        for i in range(len(models)):
            m = models[i]
            vars = [v for v in m.model_data['cohort_variable_distribution']
                    if 'type' not in m.model_data['cohort_variable_distribution'][v]]
            max_dist_vars_num = max(max_dist_vars_num, len(vars))
            score = 0
            for v in vars:
                score += competence_assessor.evaluate(models[i], r, var=v)
            model_scores.append((i, (score, m.model_data['provenance']['derivation_cohort']['N'] / largest_N)))
        for idx in range(len(model_scores)):
            scores = model_scores[idx][1]
            model_scores[idx] = (model_scores[idx][0], (scores[0] / max_dist_vars_num + scores[1]) / 2)
        return sorted(model_scores, key=lambda cl: -cl[1])

    @staticmethod
    def competence_weighted_fuse(preds, index, competence_list, default_weights, threshold=None,
                                 competence_assessor=None):
        """
        competence weighted fusion - weighted average using competence values of each model
        :param competence_assessor: not in use
        :param preds: predictions by models
        :param index: row index of the data item to be predicted in the X data frame
        :param competence_list: competence list - a list of tuple: [(model_index, competence_value)]
        :param default_weights: default weights of each model
        :param threshold: the threshold to predict 0/1
        :return: return the fused score or prediction (if threshold is not None)
        """
        use_default = True if competence_list[0][1] == 0 else False
        total_weight = 0
        total_pred = 0
        for cl in competence_list:
            if cl[1] == 0:
                continue
            w = (2 * cl[1] + default_weights[cl[0]]) if not use_default else default_weights[cl[0]]
            total_weight += w
            total_pred += w * preds[cl[0]][index]
        pred = total_pred / total_weight
        if threshold is not None:
            return 1 if pred >= threshold else 0
        else:
            return pred

    @staticmethod
    def select_most_competence(preds, index, competence_list, competence_assessor, threshold=None):
        """
        use the most competent predictor
        :param preds:
        :param index:
        :param competence_list: ordered list of competences: [(model_index, competence_value),...]
        :param competence_assessor:
        :param threshold:
        :return:
        """
        model_idx = competence_list[0][0]
        if competence_list[0][1] == 0:
            model_idx, _ = competence_assessor.default_selection()
        if threshold is not None:
            return 1 if preds[model_idx][index] >= threshold else 0
        else:
            return preds[model_idx][index]

    @staticmethod
    def use_the_highest_risk(preds, index, competence_list, competence_assessor, top_k=3, threshold=None):
        working_list = competence_list
        if competence_list[0][1] == 0:
            _, working_list = competence_assessor.default_selection()
        highest = 0
        for i in range(min(top_k, len(working_list))):
            highest = max(preds[i][index], highest)
        if threshold is not None:
            return 1 if highest >= threshold else 0
        return highest
