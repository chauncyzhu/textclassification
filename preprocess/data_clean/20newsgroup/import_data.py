# coding=gbk
"""
    ��20newsgroup textתΪpandas dataframe��py2.7
"""
#��ȡtext�ļ�
def readfile(filename):
    lines = open(filename).readlines()
    for line in lines:
        line = line.strip()
        line = line.split()