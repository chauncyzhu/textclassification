# coding=gbk
"""
    读取数据文件，清理数据文件
    不同的语料库处理方式可能不同，但是最终都要转成dataframe数据结构
    dataframe:
    第一列class--所属的类别：列表[1,0,0,0,0,0,0]表示一共选出了8个类，该文档属于第一个类)
    第二列content--文档内容：字符串--已经处理后的内容，英文需要分词、去停用词、提取词干，中文需要分词、去停用词
    第三列mark--标签：表明是训练集还是测试集，0表示训练集，1表示测试集
"""