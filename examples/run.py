import sys, os
import time, random
import argparse, logging
from pyclick.click_models.Evaluation import LogLikelihood, Perplexity, RankingPerformance
from pyclick.click_models.UBM import UBM
from pyclick.click_models.DBN import DBN
from pyclick.click_models.SDBN import SDBN
from pyclick.click_models.DCM import DCM
from pyclick.click_models.CCM import CCM
from pyclick.click_models.CTR import DCTR, RCTR, GCTR
from pyclick.click_models.CM import CM
from pyclick.click_models.PBM import PBM
from pyclick.utils.Utils import Utils
from pyclick.utils.TianGongParser import TianGongParser
from pyclick.utils.TianGongLabelParser import TianGong_HumanLabel_Parser

__author__ = 'Jianghao Lin'

def check_path(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

def is_file(file_path):
    return os.path.isfile(file_path)

def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser('PyClick')

    model_settings = parser.add_argument_group('model settings')
    model_settings.add_argument('--model', type=str, default='DCM',
                                help='choose the model to use. E.g. DCM/UBM...')
    
    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--train_dir', default='./data/train_per_query.txt',
                                help='train dir that contain the preprocessed train data')
    path_settings.add_argument('--dev_dir', default='./data/dev_per_query.txt',
                                help='dev dirs that contain the preprocessed dev data')
    path_settings.add_argument('--test_dir', default='./data/test_per_query.txt',
                                help='test dirs that contain the preprocessed test data')
    path_settings.add_argument('--human_label_dir', default='./data/human_label.txt',
                                help='the dir to Human Label txt file')
    path_settings.add_argument('--log_dir', default='./logs/',
                                help='path of the log file. If not set, logs are printed to console')

    return parser.parse_args()

if __name__ == "__main__":
    """
    Train a click model and evaluate it under traditional metrics: LL, PPL, NDCG
    """
    # get arguments
    args = parse_args()

    # Change and check dirs in args
    assert is_file(args.train_dir)
    assert is_file(args.dev_dir)
    assert is_file(args.test_dir)
    assert is_file(args.human_label_dir)
    check_path(args.log_dir)
    
    # Create a logger
    logger = logging.getLogger(args.model)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(args.log_dir, (time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime(time.time())) + '.txt')))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Log basic information
    logger.info('Running with args : {}'.format(args))
    logger.info('Model: {}'.format(args.model))

    # Initialize click model
    logger.info('Initializing the click model: {}'.format(args.model))
    click_model = globals()[args.model]()
    
    # Load train & dev & test dataset and human_label
    logger.info('Loading train dataset from {}'.format(args.train_dir))
    train_dataset = TianGongParser().parse(args.train_dir)
    logger.info('Loading dev dataset from {}'.format(args.dev_dir))
    dev_dataset = TianGongParser().parse(args.dev_dir)
    logger.info('Loading test dataset from {}'.format(args.test_dir))
    test_dataset = TianGongParser().parse(args.test_dir)
    logger.info('Loading human label from {}'.format(args.human_label_dir))
    relevance_queries = TianGong_HumanLabel_Parser().parse(args.human_label_dir)
    logger.info('train query num: {}'.format(len(train_dataset)))
    logger.info('dev query num: {}'.format(len(dev_dataset)))
    logger.info('test query num: {}'.format(len(test_dataset)))
    logger.info('human label has {} queries'.format(len(relevance_queries)))
    
    # Train
    logger.info('Start training')
    start = time.time()
    click_model.train(train_dataset)
    end = time.time()
    logger.info('Finish training. Time consumed: {} seconds'.format(end - start))
    
    # Log likelihood
    logger.info('Computing log likelihood')
    loglikelihood = LogLikelihood()
    start = time.time()
    ll_value = loglikelihood.evaluate(click_model, test_dataset)
    end = time.time()
    logger.info('Log likelihood: {}. Time consumed: {} seconds'.format(ll_value, end - start))

    # Perplexity
    logger.info('Computing perplexity')
    perplexity = Perplexity()
    start = time.time()
    perp_value = perplexity.evaluate(click_model, test_dataset)[0]
    end = time.time()
    logger.info('Perplexity: {}. Time consumed: {} seconds'.format(perp_value, end - start))

    # NDCG
    logger.info('Computing NDCG@k')
    RelevanceEstimation = RankingPerformance(args)
    start = time.time()
    ndcg_version1 = {}
    ndcg_version2 = {}
    ks = [1, 3, 5, 10]
    for k in ks:
        ndcg_version1[k], ndcg_version2[k] = RelevanceEstimation.evaluate(click_model, relevance_queries, k)
    end = time.time()
    for k in ks:
        logger.info('NDCG@{}: {}, {}'.format(k, ndcg_version1[k], ndcg_version2[k]))
    logger.info('Time consumed: {} seconds'.format(end - start))
