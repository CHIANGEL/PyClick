'''
@Author: Jianghao Lin
@Date: 2020-07-09 16:23:01
@LastEditTime: 2020-07-09 16:45:59
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /PyClick/pyclick/utils/TianGongParser.py
'''
from pyclick.click_models.task_centric.TaskCentricSearchSession import TaskCentricSearchSession
from pyclick.search_session.SearchResult import SearchResult

__author__ = 'Jianghao Lin'

class TianGongParser:
    '''
    A parser for the publicly available dataset TianGong-ST, released by Sougou
    '''
    @staticmethod
    def parse(file_path):
        sessions = [] # Named as session, but is a list of queries
        lines = open(file_path).readlines()
        for line in lines:
            attr = line.strip().split('\t')

            sid = int(attr[0])
            qid = int(attr[1].strip())
            uids = [int(uid) for uid in eval(attr[4])]
            if len(uids) < 10:
                continue
            uids = uids[:10]
            clicks = eval(attr[6])[:10]
            
            session = TaskCentricSearchSession(sid, qid)
            for uid, click in zip(uids, clicks):
                result = SearchResult(uid, click)
                session.web_results.append(result)
            sessions.append(session)
            
        return sessions
