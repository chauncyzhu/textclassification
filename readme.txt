注意的事项：
1.对语料库进行分词、去停用词等数据清理
2.统一将语料库转为pandas dataframe格式，有词典和多分类和二分类文档向量

数据格式：
1.词典向量voca
[word_appear_set,class_word_appear_set,word_doc_set,doc_class_set,bdc,df_bdc]

2.文档向量（下面为每一行）
class:类别，格式如[1,0,0,0,0,0,0,0]，1出现的序号表示属于第几个类
content:内容，格式为[x,x,x,x,x...]，x表示word
mark:标记，表示属于训练集还是测试集，0是训练集，1为测试集
bdc:文档向量，格式为[x,x,x,x,x...]，x表示bdc值

注意的错误1：
未知原因，reuters_train.csv第4192行，即编号为4190处，在运行textclassification\classification\knn\knn.py时出现了错误
原来在这一行的vector出现了：...1.0, nan, 0.0042338727427472245...
而在前面的content中也可以发现：u'nan'
可能这是一个pandas dataframe的坑，会自动将nan转成NaN，因为reuters_train.csv只出现了一次，因此手动修改为0
