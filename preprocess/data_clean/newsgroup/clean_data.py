# coding=gbk
import cPickle as Pickle
import nltk
import string
import re
"""
    将每条数据进行清理
"""
#先进行分词，传入一个line
def getTokenWords(line):
    return nltk.word_tokenize(line)


def clean_data(pd_data):
    def f(line):
        # 先进行分词
        line = getTokenWords(line)
        return line
    pd_data['content'] = pd_data['content'].apply(f)
