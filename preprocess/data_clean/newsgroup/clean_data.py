# coding=gbk
import cPickle as Pickle
import nltk
import string
import re
"""
    ��ÿ�����ݽ�������
"""
#�Ƚ��зִʣ�����һ��line
def getTokenWords(line):
    return nltk.word_tokenize(line)


def clean_data(pd_data):
    def f(line):
        # �Ƚ��зִ�
        line = getTokenWords(line)
        return line
    pd_data['content'] = pd_data['content'].apply(f)
