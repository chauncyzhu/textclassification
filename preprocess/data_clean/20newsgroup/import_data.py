# coding=gbk
"""
    将20newsgroup text转为pandas dataframe，py2.7
"""
#读取text文件
def readfile(filename):
    lines = open(filename).readlines()
    for line in lines:
        line = line.strip()
        line = line.split()