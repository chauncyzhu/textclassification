# coding=gbk
import preprocess.data_clean.reuters.clean_data as cd
import preprocess.data_clean.reuters.import_data as id
import preprocess.transfer_vector.voca_dict.voca_data as vd
import preprocess.transfer_vector.generate_vector.feature as feature
import preprocess.transfer_vector.generate_vector.transfer_vector as tv
import utils.reuters_path as path
"""
   �Ժ������е��ã����沿����Ҫ�Ƕ�·͸�����Ͽ���д���
"""
TOP_CLASS_NUM = 8
CONFIRM_POS_CLASS = 0  #ָ���������������

#����������ֵ��ȡ�������Ͷ������class_num��һ��
def __voca_dict(class_num,voca_csv=None):
    # ���ݻ�ȡ������
    pd_train, pd_test = id.getTrainAndTest(path.SOURCEFILE, TOP_CLASS_NUM)
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
    class_num = 8  # ������������

    pd_train, pd_test,voca_dict = __voca_dict(class_num, voca_csv=path.VOCA_MULTI_CSV)  #��ȡ�������ֵ䣬����
    __generate_vector(pd_train, pd_test, voca_dict,"bdc", train_csv=path.TRAIN_MULTI_BDC_CSV, test_csv=path.TEST_MULTI_BDC_CSV)
    __generate_vector(pd_train, pd_test, voca_dict,"df_bdc", train_csv=path.TRAIN_MULTI_DF_BDC_CSV, test_csv=path.TEST_MULTI_DF_BDC_CSV)


#��ö��������
def binary_class_data():
    class_num = 2  # �������������

    pd_train, pd_test,voca_dict = __voca_dict(class_num, voca_csv=path.VOCA_BINARY_CSV)  #��ȡ�������ֵ䣬����

    #Ӧ�ø���ָ��������ı�������е�class
    def f(x):
        if x[CONFIRM_POS_CLASS] == 1:
            return [1,0]
        else:
            return [0,1]
    pd_train['class'] = pd_train['class'].apply(f)
    pd_test['class'] = pd_test['class'].apply(f)

    __generate_vector(pd_train, pd_test, voca_dict,"bdc", train_csv=path.TRAIN_BINARY_BDC_CSV, test_csv=path.TEST_BINARY_BDC_CSV)
    __generate_vector(pd_train, pd_test, voca_dict,"df_bdc", train_csv=path.TRAIN_BINARY_DF_BDC_CSV, test_csv=path.TEST_BINARY_DF_BDC_CSV)

if __name__ == '__main__':
    #multi_class_data()
    binary_class_data()
