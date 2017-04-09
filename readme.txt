注意的错误1：
未知原因，reuters_train.csv第4192行，即编号为4190处，在运行textclassification\classification\knn\knn.py时出现了错误
原来在这一行的vector出现了：...1.0, nan, 0.0042338727427472245...
而在前面的content中也可以发现：u'nan'
可能这是一个pandas dataframe的坑，会自动将nan转成NaN，因为reuters_train.csv只出现了一次，因此手动修改为0