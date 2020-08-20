#
# Copyright (C) 2015 Ilya Markov
#
# Full copyright notice can be found in LICENSE.
#
from __future__ import division
from abc import abstractmethod
#from sklearn.metrics import roc_auc_score
#from scipy.stats import pearsonr
#import numpy as np
import math
import collections
import logging
import pprint
import sys

__author__ = 'Luka Stout, Finde Xumara, Ilya Markov'


RANK_MAX = 10


class Evaluation(object):
    """An abstract evaluation method for click models."""

    @abstractmethod
    def evaluate(self, click_model, search_sessions):
        """
        Evaluates the quality of the given click model using the given list of search sessions.

        This method must be implemented by subclasses.

        :param click_model: The click model to evaluate.
        :param search_sessions: The list of search sessions (also known as test set).
        :return: The quality of the click model, given the set of test search sessions.
        """
        pass


class LogLikelihood(Evaluation):
    """
    The log-likelihood evaluation of click models.
    """
    #TODO: according to what paper?

    def evaluate(self, click_model, search_sessions):
        """
        Returns the log-likelihood of search sessions, given a click model.
        LL(Sessions | Model) = sum_{session in Sessions} P(clicks in the session | Model)
        """
        loglikelihood = 0

        for session in search_sessions:
            click_probs = click_model.get_conditional_click_probs(session)
            log_click_probs = [math.log(prob) for prob in click_probs]
            loglikelihood += sum(log_click_probs) / len(log_click_probs)
        loglikelihood /= len(search_sessions)

        return loglikelihood 


class Perplexity(Evaluation):
    """
    The perplexity evaluation of click models.

    Dupret, Georges E. and Piwowarski, Benjamin.
    A user browsing model to predict search engine click data from past observations.
    Proceedings of SIGIR, pages 331-338, 2008.
    """

    def evaluate(self, click_model, search_sessions):
        """
        Given a click model and search sessions, returns the following:
        perplexity, [perplexity_at_1, ..., perplexity_at_N]
        """
        perplexity_at_rank = [0.0] * RANK_MAX

        for session in search_sessions:
            click_probs = click_model.get_full_click_probs(session)

            for rank, click_prob in enumerate(click_probs):
                if session.web_results[rank].click:
                    p = click_prob
                else:
                    p = 1 - click_prob

                if p > 0:
                    perplexity_at_rank[rank] += math.log(p, 2)
                else:
                    print >>sys.stderr, 'Click probability is not positive: %f' % p

        perplexity_at_rank = [2 ** (-x / len(search_sessions)) for x in perplexity_at_rank]
        perplexity = sum(perplexity_at_rank) / len(perplexity_at_rank)
        return perplexity, perplexity_at_rank


class PerplexityCond(Evaluation):
    """
    The conditional perplexity evaluation of click models.

    Dupret, Georges E. and Piwowarski, Benjamin.
    A user browsing model to predict search engine click data from past observations.
    Proceedings of SIGIR, pages 331-338, 2008.
    """

    def evaluate(self, click_model, search_sessions):
        """
        Given a click model and search sessions, returns the following:
        perplexity, [perplexity_at_1, ..., perplexity_at_N]
        """
        perplexity_at_rank = [0.0] * RANK_MAX

        for session in search_sessions:
            click_probs = click_model.get_conditional_click_probs(session)

            for rank, click_prob in enumerate(click_probs):
                perplexity_at_rank[rank] += math.log(click_prob, 2)

        perplexity_at_rank = [2 ** (-x / len(search_sessions)) for x in perplexity_at_rank]
        perplexity = sum(perplexity_at_rank) / len(perplexity_at_rank)
        return perplexity, perplexity_at_rank


class CTRPrediction(Evaluation):

    CLICK_TRESHOLD = .75

    def _group_sessions(self, sessions):
        """
            Group sessions based on query
        """
        session_dict = collections.defaultdict(list)
        for session in sessions:
            session_dict[session.query].append(session)
        return session_dict
    
    def _split_train_test_sets(self, sessions):
        """
            Splits the sessions in a set of train sets and a set of test sets.
            test_sets[i] belongs to train_sets[i]
        """
        # For every session find out whether there is a document at the first position that also occurs in other positions
        test_sets = []
        train_sets = []
        for s_idx, session in enumerate(sessions):
            pos_1 = session.web_results[0].id
            found_in_other_test_set = False

            #Check whether session is already in a test set.
            for test in test_sets:
                if pos_1 == test[0].web_results[0].id:
                    found_in_other_test_set = True
            if found_in_other_test_set:
                break
                    
            #If not already in a test set create a new test/train pair
            test = [session]
            train = []
            
            for session_2 in sessions[:s_idx] + sessions[s_idx+1:]:

                #Add session to test set if they have same doc in pos 1
                if pos_1 == session_2.web_results[0].id:
                    test.append(session_2)
                #Add session to train set if it is in another place than the first
                elif pos_1 in [result.id for result in session_2.web_results[1:]]:
                    train.append(session_2)
            
            #Only add if there is both a test and train set.
            if test and train:
                test_sets.append(test)
                train_sets.append(train)
        return train_sets, test_sets
    
    def evaluate(self, click_model, search_sessions):
        """
            Returns the RMSE of the CTR wrt the given sessions.
            Calculated according section 4.2 of:
            "A Dynamic Bayesian Network Click Model for Web Search Ranking" Chappele and Zhang, 2009
        """

        session_dict = self._group_sessions(search_sessions)
        MSEs, weights = [], []

        for query_id, search_sessions in session_dict.items():
            
            train_sets, test_sets = self._split_train_test_sets(search_sessions)
            
            # Train the model on the train set and get the predicted clicks of the test set.
            for test, train in zip(test_sets,train_sets):
                click_model.train(train)
                pred_clicks, true_clicks = [], []
                pred_click_prob =  click_model.get_full_click_probs(test[0])[0]
                for t in test:
                    true_clicks.append( t.web_results[0].click )

                true_ctr = sum(1 for true_click in true_clicks if true_click)/len(true_clicks)
                MSE = (pred_click_prob - true_ctr) ** 2
                MSEs.append(MSE)

                weights.append(len(test))
            
        # Average MSE over all queries
        return math.sqrt(np.average(MSEs,weights=weights))


class RelevancePrediction(Evaluation):

    def __init__(self, true_relevances):
        # relevances should be a dict with structure relevances[query_id][url] -> relevance
        self.relevances = true_relevances
        super(self.__class__, self).__init__()

    def evaluate(self, click_model, search_sessions):
        """
            Returns the AUC of the true relevances and the predicted relevances by the model and the Pearson correlation between the two.
            AUC: a statistically consistent and more discriminating measure than accuracy. Charles X. Ling and Jin Huang and Harry Zhang. 2003
        """

        pred_relevances = []
        true_relevances = []
        current_i = 0

        for session in search_sessions:
            for rank, result in enumerate(session.web_results):
                if session.query in self.relevances:
                    if result.id in self.relevances[session.query]:
                        true_rel = self.relevances[session.query][result.id]
                        true_relevances.append(true_rel)
                        pred_relevances.append(click_model.predict_relevance(session.query, result.id))
                
        auc = roc_auc_score(true_relevances, pred_relevances)
        cor, p = pearsonr(true_relevances, pred_relevances)
        return auc, cor, p

class RankingPerformance():

    def __init__(self, args):
        self.logger = logging.getLogger(args.model)

    def evaluate(self, model, relevance_queries, k):
        """
        Return the NDCG@k of the rankings given by the model for the given sessions.
        """
        # For every useful query get the predicted relevance and compute NDCG
        total_ndcg = 0
        total_query = 0
        not_useful = 0
        for info_per_query in relevance_queries:
            total_query += 1
            id = info_per_query['id']
            sid = info_per_query['sid']
            qid = info_per_query['qid']
            uids = info_per_query['uids']
            relevances = info_per_query['relevances']
            ideal_ranking_relevances = sorted(relevances, reverse=True)[:k]
            
            # Only use query if there is a document with a positive ranking. (Otherwise IDCG will be 0 -> NDCG undetermined.)
            if not any(ideal_ranking_relevances):
                not_useful += 1
                continue
            
            # Get the relevances computed by the model
            pred_rels = dict()
            for uid in uids:
                pred_rels[uid] = model.predict_relevance(qid, uid)
            ranking = sorted([uid for uid in pred_rels], key = lambda uid : pred_rels[uid], reverse=True)
            ranking_relevances = [relevances[uids.index(uid)] for uid in ranking[:k]]
            
            # Compute ndcg
            dcg = self.dcg(ranking_relevances)
            idcg = self.dcg(ideal_ranking_relevances)
            ndcg = dcg / idcg
            assert ndcg <= 1
            total_ndcg += ndcg

        # Checks
        assert total_query == len(relevance_queries)
        assert len(relevance_queries) - not_useful > 0

        # Average NDCG over all queries
        ndcg_version1 = total_ndcg / (len(relevance_queries) - not_useful)
        ndcg_version2 = (total_ndcg + not_useful) / len(relevance_queries)
        return ndcg_version1, ndcg_version2

    def dcg(self, ranking_relevances):
        """
        Computes the DCG for a given ranking_relevances
        """
        return sum([(2 ** relevance - 1) / math.log(rank + 2, 2) for rank, relevance in enumerate(ranking_relevances)])