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

