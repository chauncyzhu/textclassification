# coding=gbk
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer
from sklearn import metrics
from sklearn.naive_bayes import BernoulliNB
import numpy as np

import preprocess.data_clean.reuters.import_data as rid
import preprocess.data_clean.newsgroup.import_data as nid  #���ݻ�ȡ
import utils.newsgroup_path as npath
import utils.reuters_path as rpath
"""
    ��Ҷ˹��ʵ���㷨����Ҫ������sklearn����ʵ��
"""
def naive_bayes(train_data,train_target,test_data,test_target):
    nbc = Pipeline([
        ('vect', TfidfVectorizer(

        )),
        ('clf', MultinomialNB(alpha=1.0)),
    ])
    nbc.fit(train_data, train_target)  # ѵ�����ǵĶ���ʽģ�ͱ�Ҷ˹������
    predict = nbc.predict(test_data)  # �ڲ��Լ���Ԥ����
    count_eaq = 0  # ͳ��Ԥ����ȷ�Ľ������
    count_not_eaq = 0
    for left, right in zip(predict, test_target):
        print(left, right)
        if left == right:
            count_eaq += 1
        else:
            count_not_eaq += 1
    print(float(count_eaq) / len(test_target))
    print(count_not_eaq)


def newsgroup():
    CONFIRM_POS_CLASS = 0
    # ��ȡ���ݲ�תΪdataframe
    pd_train, pd_test = nid.getTrainAndTest(npath.SOURCEFILE)  #ע�������Ѿ������2����

    # Ӧ�ø���ָ��������ı�������е�class
    def f(x):
        if x[CONFIRM_POS_CLASS] == 1:
            return 1
        else:
            return 0

    pd_train['class'] = pd_train['class'].apply(f)
    pd_test['class'] = pd_test['class'].apply(f)
    print(pd_train)
    print(pd_test)

    train_data = list(pd_train['content'])
    train_target = list(pd_train['class'])
    test_data = list(pd_test['content'])
    test_target = list(pd_test['class'])

    naive_bayes(train_data, train_target, test_data, test_target)

def reuters():
    TOP_CLASS_NUM = 8
    CONFIRM_POS_CLASS = 0
    pd_train, pd_test = rid.getTrainAndTest(rpath.SOURCEFILE, TOP_CLASS_NUM)

    # Ӧ�ø���ָ��������ı�������е�class
    def f(x):
        if x[CONFIRM_POS_CLASS] == 1:
            return 1
        else:
            return 0

    pd_train['class'] = pd_train['class'].apply(f)
    pd_test['class'] = pd_test['class'].apply(f)
    print(pd_train)
    print(pd_test)

    train_data = list(pd_train['content'])
    train_target = list(pd_train['class'])
    test_data = list(pd_test['content'])
    test_target = list(pd_test['class'])

    naive_bayes(train_data, train_target, test_data, test_target)

if __name__ == '__main__':
    #reuters()
    newsgroup()

