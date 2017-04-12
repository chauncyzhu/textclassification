# coding=gbk
import preprocess.data_clean.reuters.clean_data as cd
import preprocess.data_clean.reuters.import_data as id
import preprocess.transfer_vector.voca_dict.voca_data as vd
import preprocess.transfer_vector.generate_vector.feature as feature
import preprocess.transfer_vector.generate_vector.transfer_vector as tv
import utils.reuters_path as path
"""
   对函数进行调用，下面部分主要是对路透社语料库进行处理
"""
TOP_CLASS_NUM = 8
CONFIRM_POS_CLASS = 0  #指定二分类正类序号

#数据清理和字典获取，多分类和二分类的class_num不一样
def __voca_dict(class_num,voca_csv=None):
    # 数据获取和清理
    pd_train, pd_test = id.getTrainAndTest(path.SOURCEFILE, TOP_CLASS_NUM)
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
    class_num = 8  # 多分类的类别个数

    pd_train, pd_test,voca_dict = __voca_dict(class_num, voca_csv=path.VOCA_MULTI_CSV)  #获取多分类的字典，包括
    __generate_vector(pd_train, pd_test, voca_dict,"bdc", train_csv=path.TRAIN_MULTI_BDC_CSV, test_csv=path.TEST_MULTI_BDC_CSV)
    __generate_vector(pd_train, pd_test, voca_dict,"df_bdc", train_csv=path.TRAIN_MULTI_DF_BDC_CSV, test_csv=path.TEST_MULTI_DF_BDC_CSV)


#获得多分类数据
def binary_class_data():
    class_num = 2  # 二分类的类别个数

    pd_train, pd_test,voca_dict = __voca_dict(class_num, voca_csv=path.VOCA_BINARY_CSV)  #获取多分类的字典，包括

    #应该根据指定的正类改变二分类中的class
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
