# coding=gbk
"""
    将特征权重应用到向量化中
"""
#将pd_data的content属性转为特征向量，放在"vector"这一列中，targetfile可以指定要保存在那个文件里面去
def changeToFeatureVector(pd_data,voca_dict,feature_name,target_file=None):
    #将分词序列转换为向量
    def f(x,name):
        return list(voca_dict[name][x])  # 太神奇了，DataFrame允许列表输入，会自动迭代列表的每一个元素
    pd_data['vector'] = pd_data['content'].apply((lambda x:f(x,feature_name)))  # 对于每一行words而言，但是神奇的是这个迭代居然到了每一个词？pn['words']相当于取出了一个series
    #将pn写入文件中
    if target_file:
        pd_data.to_csv(target_file)