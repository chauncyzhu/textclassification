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

