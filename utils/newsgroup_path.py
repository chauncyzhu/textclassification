# coding=gbk
"""
    ��Ÿ��������ļ����ļ�·������Ϊ�ļ����󲻺��ϴ���github��
"""
ROOT_PATH = "D:/Coding/pycharm-professional/pycharm-file/pythondata/"

NEWSGROUP = ROOT_PATH + "newsgroup/textclassification"  #newsgroup

SOURCEFILE = ROOT_PATH + "newsgroup/6ng.txt"  # ·͸�����Ͽ�
TRAIN_TEXT = ROOT_PATH + "newsgroup/20ng-train-all-terms.txt"  # ѵ����
TEST_TEXT = ROOT_PATH + "newsgroup/20ng-test-all-terms.txt"  # ѵ����

VOCA_MULTI_CSV = NEWSGROUP + "/multiclass/voca_dict_multiclass.csv"  # �ֵ�
VOCA_BINARY_CSV = NEWSGROUP + "/binaryclass/voca_dict_binaryclass.csv"  # �ֵ�

TRAIN_MULTI_BDC_CSV = NEWSGROUP + "/multiclass/newsgroup_train_multiclass_bdc.csv"  # �ֿ���ѵ����
TEST_MULTI_BDC_CSV = NEWSGROUP + "/multiclass/newsgroup_test_multiclass_bdc.csv"  # �ֿ��Ĳ��Լ�
TRAIN_MULTI_DF_BDC_CSV = NEWSGROUP + "/multiclass/newsgroup_train_multiclass_df_bdc.csv"  # �ֿ���ѵ����
TEST_MULTI_DF_BDC_CSV = NEWSGROUP + "/multiclass/newsgroup_test_multiclass_df_bdc.csv"  # �ֿ��Ĳ��Լ�

TRAIN_BINARY_BDC_CSV = NEWSGROUP + "/binaryclass/newsgroup_train_binaryclass_bdc.csv"  # �ֿ���ѵ����
TEST_BINARY_BDC_CSV = NEWSGROUP + "/binaryclass/newsgroup_test_binaryclass_bdc.csv"  # �ֿ��Ĳ��Լ�
TRAIN_BINARY_DF_BDC_CSV = NEWSGROUP + "/binaryclass/newsgroup_train_binaryclass_df_bdc.csv"  # �ֿ���ѵ����
TEST_BINARY_DF_BDC_CSV = NEWSGROUP + "/binaryclass/newsgroup_test_binaryclass_df_bdc.csv"  # �ֿ��Ĳ��Լ�

EVALUATION_BIANRY_BDC_CSV = NEWSGROUP + "/evaluation/evaluation_binary_bdc_csv.csv"  # ���������
EVALUATION_BINARY_DF_BDC_CSV = NEWSGROUP + "/evaluation/evaluation_binary_df_bdc_csv.csv"  # ����������

EVALUATION_MULTI_BDC_CSV = NEWSGROUP + "/evaluation/evaluation_multi_bdc_csv.csv"  # ���������
EVALUATION_MULTI_DF_BDC_CSV = NEWSGROUP + "/evaluation/evaluation_multi_df_bdc_csv.csv"  # ���������