# coding=gbk
"""
    存放各个数据文件的文件路径，因为文件过大不好上传到github上
"""
ROOT_PATH = "D:/Coding/pycharm-professional/pycharm-file/pythondata/"

NEWSGROUP = ROOT_PATH + "newsgroup/textclassification"  #newsgroup

SOURCEFILE = ROOT_PATH + "newsgroup/6ng.txt"  # 路透社语料库
TRAIN_TEXT = ROOT_PATH + "newsgroup/20ng-train-all-terms.txt"  # 训练集
TEST_TEXT = ROOT_PATH + "newsgroup/20ng-test-all-terms.txt"  # 训练集

VOCA_MULTI_CSV = NEWSGROUP + "/multiclass/voca_dict_multiclass.csv"  # 字典
VOCA_BINARY_CSV = NEWSGROUP + "/binaryclass/voca_dict_binaryclass.csv"  # 字典

TRAIN_MULTI_BDC_CSV = NEWSGROUP + "/multiclass/newsgroup_train_multiclass_bdc.csv"  # 分开的训练集
TEST_MULTI_BDC_CSV = NEWSGROUP + "/multiclass/newsgroup_test_multiclass_bdc.csv"  # 分开的测试集
TRAIN_MULTI_DF_BDC_CSV = NEWSGROUP + "/multiclass/newsgroup_train_multiclass_df_bdc.csv"  # 分开的训练集
TEST_MULTI_DF_BDC_CSV = NEWSGROUP + "/multiclass/newsgroup_test_multiclass_df_bdc.csv"  # 分开的测试集

TRAIN_BINARY_BDC_CSV = NEWSGROUP + "/binaryclass/newsgroup_train_binaryclass_bdc.csv"  # 分开的训练集
TEST_BINARY_BDC_CSV = NEWSGROUP + "/binaryclass/newsgroup_test_binaryclass_bdc.csv"  # 分开的测试集
TRAIN_BINARY_DF_BDC_CSV = NEWSGROUP + "/binaryclass/newsgroup_train_binaryclass_df_bdc.csv"  # 分开的训练集
TEST_BINARY_DF_BDC_CSV = NEWSGROUP + "/binaryclass/newsgroup_test_binaryclass_df_bdc.csv"  # 分开的测试集

EVALUATION_BIANRY_BDC_CSV = NEWSGROUP + "/evaluation/evaluation_binary_bdc_csv.csv"  # 多分类结果集
EVALUATION_BINARY_DF_BDC_CSV = NEWSGROUP + "/evaluation/evaluation_binary_df_bdc_csv.csv"  # 二分类结果集

EVALUATION_MULTI_BDC_CSV = NEWSGROUP + "/evaluation/evaluation_multi_bdc_csv.csv"  # 多分类结果集
EVALUATION_MULTI_DF_BDC_CSV = NEWSGROUP + "/evaluation/evaluation_multi_df_bdc_csv.csv"  # 多分类结果集