# coding=gbk
import preprocess.data_clean.reuters.clean_data as cd
import preprocess.data_clean.reuters.import_data as id
import preprocess.transfer_vector.voca_dict.voca_data as vd
import preprocess.transfer_vector.generate_vector.feature as feature
import preprocess.transfer_vector.generate_vector.transfer_vector as tv
import pandas as pd
"""
   �Ժ������е��ã����沿����Ҫ�Ƕ�·͸�����Ͽ���д���
"""
SOURCEFILE = "../data/reuters/reuter_all.xml"   #·͸�����Ͽ�
TOP_CLASS_NUM = 8

#����������ֵ��ȡ�������Ͷ������class_num��һ��
def __voca_dict(class_num,voca_csv=None):
    # ���ݻ�ȡ������
    pd_train, pd_test = id.getTrainAndTest(SOURCEFILE, TOP_CLASS_NUM)
    # ����ѵ����
    cd.clean_data(pd_train)
    # ������Լ�
    cd.clean_data(pd_test)

    #��ȡ�����ֵ䣬����Ƕ����class_num>2������Ƕ����࣬class_num=2
    #pd_train = pd_train.head(100)  #����ѵ��������
    voca_dict = vd.getRelativeValue(pd_train,vd.getUniqueVocabulary(pd_train),class_num)   #getUniqueVocabulary�ȽϺ�ʱ���洢��csv��

    # �������Ӹ���term weighting schema�����������
    feature.getBDCVector(voca_dict, class_num,"bdc")  # �����ֵ����BDCֵ����Ҫָ��index
    feature.getDFBDCVector(voca_dict, class_num,"df_bdc")  # �����ֵ����DF_BDCֵ����Ҫָ��index
    feature.getTotalVoca(pd_test, voca_dict)  # �����Լ��е��������뵽�ʵ���

    if voca_csv:   #���������д���ļ���
        voca_dict.to_csv(voca_csv)
    print(voca_dict)
    return pd_train,pd_test,voca_dict

#ת��Ϊ��ͬ������
def __generate_vector(pd_train,pd_test,voca_dict,feature_name,train_csv=None,test_csv=None):
    pd_train_copy = pd_train.copy()  #��ֹ���ݸ���
    pd_test_copy = pd_test.copy()

    # ���Լ���ѵ����תΪ����
    tv.changeToFeatureVector(pd_train_copy, voca_dict, feature_name)
    tv.changeToFeatureVector(pd_test_copy, voca_dict, feature_name)
    if train_csv:
        pd_train_copy.to_csv(train_csv)  # д��ѵ���ļ���
    if test_csv:
        pd_test_copy.to_csv(test_csv)  # д������ļ���


#���������ݴ������
def multi_class_data():
    voca_csv = "../data/reuters/multiclass/voca_dict_multiclass.csv"  # �ֵ�

    train_bdc_csv = "../data/reuters/multiclass/reuter_train_multiclass_bdc.csv"  # �ֿ���ѵ����
    test_bdc_csv = "../data/reuters/multiclass/reuter_test_multiclass_bdc.csv"  # �ֿ��Ĳ��Լ�
    train_df_bdc_csv = "../data/reuters/multiclass/reuter_train_multiclass_df_bdc.csv"  # �ֿ���ѵ����
    test_df_bdc_csv = "../data/reuters/multiclass/reuter_test_multiclass_df_bdc.csv"  # �ֿ��Ĳ��Լ�

    class_num = 8  # ������������

    pd_train, pd_test,voca_dict = __voca_dict(class_num, voca_csv)  #��ȡ�������ֵ䣬����
    __generate_vector(pd_train, pd_test, voca_dict,"bdc", train_bdc_csv, test_bdc_csv)
    __generate_vector(pd_train, pd_test, voca_dict,"df_bdc", train_df_bdc_csv, test_df_bdc_csv)


#��ö��������
def binary_class_data():
    voca_csv = "../data/reuters/binaryclass/voca_dict_binaryclass.csv"  # �ֵ�

    train_bdc_csv = "../data/reuters/binaryclass/reuter_train_binaryclass_bdc.csv"  # �ֿ���ѵ����
    test_bdc_csv = "../data/reuters/binaryclass/reuter_test_binaryclass_bdc.csv"  # �ֿ��Ĳ��Լ�
    train_df_bdc_csv = "../data/reuters/binaryclass/reuter_train_binaryclass_df_bdc.csv"  # �ֿ���ѵ����
    test_df_bdc_csv = "../data/reuters/binaryclass/reuter_test_binaryclass_df_bdc.csv"  # �ֿ��Ĳ��Լ�

    class_num = 2  # �������������

    pd_train, pd_test,voca_dict = __voca_dict(class_num, voca_csv)  #��ȡ�������ֵ䣬����
    __generate_vector(pd_train, pd_test, voca_dict,"bdc", train_bdc_csv, test_bdc_csv)
    __generate_vector(pd_train, pd_test, voca_dict,"df_bdc", train_df_bdc_csv, test_df_bdc_csv)



if __name__ == '__main__':
    #multi_class_data()
    binary_class_data()
