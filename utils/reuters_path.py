# coding=gbk
"""
    存放各个数据文件的文件路径，因为文件过大不好上传到github上
"""
ROOT_PATH = "D:/Coding/pycharm-professional/pycharm-file/pythondata/"

REUTERS = ROOT_PATH+"reuters/textclassification"  #路透社，注意这里没有s

SOURCEFILE = ROOT_PATH+"reuters/reuter_all.xml"  # 路透社语料库

VOCA_MULTI_CSV = REUTERS + "/multiclass/voca_dict_multiclass.csv"  # 字典
VOCA_BINARY_CSV = REUTERS + "/binaryclass/voca_dict_binaryclass.csv"  # 字典

TRAIN_MULTI_BDC_CSV = REUTERS + "/multiclass/reuter_train_multiclass_bdc.csv"  # 分开的训练集
TEST_MULTI_BDC_CSV = REUTERS + "/multiclass/reuter_test_multiclass_bdc.csv"  # 分开的测试集
TRAIN_MULTI_DF_BDC_CSV = REUTERS + "/multiclass/reuter_train_multiclass_df_bdc.csv"  # 分开的训练集
TEST_MULTI_DF_BDC_CSV = REUTERS + "/multiclass/reuter_test_multiclass_df_bdc.csv"  # 分开的测试集

TRAIN_BINARY_BDC_CSV = REUTERS + "/binaryclass/reuter_train_binaryclass_bdc.csv"  # 分开的训练集
TEST_BINARY_BDC_CSV = REUTERS + "/binaryclass/reuter_test_binaryclass_bdc.csv"  # 分开的测试集
TRAIN_BINARY_DF_BDC_CSV = REUTERS + "/binaryclass/reuter_train_binaryclass_df_bdc.csv"  # 分开的训练集
TEST_BINARY_DF_BDC_CSV = REUTERS + "/binaryclass/reuter_test_binaryclass_df_bdc.csv"  # 分开的测试集

EVALUATION_BIANRY_BDC_CSV = REUTERS + "/evaluation/evaluation_binary_bdc_csv.csv"  # 二分类结果集
EVALUATION_BINARY_DF_BDC_CSV = REUTERS + "/evaluation/evaluation_binary_df_bdc_csv.csv"  # 二分类结果集

EVALUATION_MULTI_BDC_CSV = REUTERS + "/evaluation/evaluation_multi_bdc_csv.csv"  # 多分类结果集
EVALUATION_MULTI_DF_BDC_CSV = REUTERS + "/evaluation/evaluation_multi_df_bdc_csv.csv"  # 多分类结果集

MULTI_BDC_CSV = REUTERS + "/evaluation/multi_bdc_csv.csv"
