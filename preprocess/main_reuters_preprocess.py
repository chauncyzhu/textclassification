# coding=gbk
import preprocess.data_clean.reuters.clean_data as cd
import preprocess.data_clean.reuters.import_data as id
import preprocess.transfer_vector.voca_dict.voca_data as vd
import preprocess.transfer_vector.generate_vector.feature as feature
import preprocess.transfer_vector.generate_vector.transfer_vector as tv
import pandas as pd
"""
   对函数进行调用，下面部分主要是对路透社语料库进行处理
"""
SOURCEFILE = "../data/reuters/reuter_all.xml"   #路透社语料库
TOP_CLASS_NUM = 8

#数据清理和字典获取，多分类和二分类的class_num不一样
def __voca_dict(class_num,voca_csv=None):
    # 数据获取和清理
    pd_train, pd_test = id.getTrainAndTest(SOURCEFILE, TOP_CLASS_NUM)
    # 处理训练集
    cd.clean_data(pd_train)
    # 处理测试集
    cd.clean_data(pd_test)

    #获取分类字典，如果是多分类class_num>2，如果是二分类，class_num=2
    #pd_train = pd_train.head(100)  #控制训练集个数
    voca_dict = vd.getRelativeValue(pd_train,vd.getUniqueVocabulary(pd_train),class_num)   #getUniqueVocabulary比较耗时，存储在csv中

    # 如需增加更多term weighting schema，在这里添加
    feature.getBDCVector(voca_dict, class_num,"bdc")  # 根据字典计算BDC值，需要指定index
    feature.getDFBDCVector(voca_dict, class_num,"df_bdc")  # 根据字典计算DF_BDC值，需要指定index
    feature.getTotalVoca(pd_test, voca_dict)  # 将测试集中的特征加入到词典中

    if voca_csv:   #如果存在则写入文件中
        voca_dict.to_csv(voca_csv)
    print(voca_dict)
    return pd_train,pd_test,voca_dict

#转化为不同的向量
def __generate_vector(pd_train,pd_test,voca_dict,feature_name,train_csv=None,test_csv=None):
    pd_train_copy = pd_train.copy()  #防止数据干扰
    pd_test_copy = pd_test.copy()

    # 测试集和训练集转为向量
    tv.changeToFeatureVector(pd_train_copy, voca_dict, feature_name)
    tv.changeToFeatureVector(pd_test_copy, voca_dict, feature_name)
    if train_csv:
        pd_train_copy.to_csv(train_csv)  # 写入训练文件中
    if test_csv:
        pd_test_copy.to_csv(test_csv)  # 写入测试文件中


#多分类的数据处理操作
def multi_class_data():
    voca_csv = "../data/reuters/multiclass/voca_dict_multiclass.csv"  # 字典

    train_bdc_csv = "../data/reuters/multiclass/reuter_train_multiclass_bdc.csv"  # 分开的训练集
    test_bdc_csv = "../data/reuters/multiclass/reuter_test_multiclass_bdc.csv"  # 分开的测试集
    train_df_bdc_csv = "../data/reuters/multiclass/reuter_train_multiclass_df_bdc.csv"  # 分开的训练集
    test_df_bdc_csv = "../data/reuters/multiclass/reuter_test_multiclass_df_bdc.csv"  # 分开的测试集

    class_num = 8  # 多分类的类别个数

    pd_train, pd_test,voca_dict = __voca_dict(class_num, voca_csv)  #获取多分类的字典，包括
    __generate_vector(pd_train, pd_test, voca_dict,"bdc", train_bdc_csv, test_bdc_csv)
    __generate_vector(pd_train, pd_test, voca_dict,"df_bdc", train_df_bdc_csv, test_df_bdc_csv)


#获得多分类数据
def binary_class_data():
    voca_csv = "../data/reuters/binaryclass/voca_dict_binaryclass.csv"  # 字典

    train_bdc_csv = "../data/reuters/binaryclass/reuter_train_binaryclass_bdc.csv"  # 分开的训练集
    test_bdc_csv = "../data/reuters/binaryclass/reuter_test_binaryclass_bdc.csv"  # 分开的测试集
    train_df_bdc_csv = "../data/reuters/binaryclass/reuter_train_binaryclass_df_bdc.csv"  # 分开的训练集
    test_df_bdc_csv = "../data/reuters/binaryclass/reuter_test_binaryclass_df_bdc.csv"  # 分开的测试集

    class_num = 2  # 二分类的类别个数

    pd_train, pd_test,voca_dict = __voca_dict(class_num, voca_csv)  #获取多分类的字典，包括
    __generate_vector(pd_train, pd_test, voca_dict,"bdc", train_bdc_csv, test_bdc_csv)
    __generate_vector(pd_train, pd_test, voca_dict,"df_bdc", train_df_bdc_csv, test_df_bdc_csv)



if __name__ == '__main__':
    #multi_class_data()
    binary_class_data()
