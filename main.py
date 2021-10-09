from collections import OrderedDict
import pandas as pd
import matplotlib.pyplot as plt
examDict={
    '学习时间':[0.50,0.75,1.00,1.25,1.50,1.75,1.75,2.00,2.25,
            2.50,2.75,3.00,3.25,3.50,4.00,4.25,4.50,4.75,5.00,5.50],
    '分数':    [10,  22,  13,  43,  20,  22,  33,  50,  62,
              48,  55,  75,  62,  73,  81,  76,  64,  82,  90,  93]
}

examDf=pd.DataFrame(examDict)
exam_x=examDf.loc[:,'学习时间']
exam_y=examDf.loc[:,'分数']

#方法2: plt.scatter(examDf['学习时间'],examDf["分数"])


examDf.plot(x="学习时间",y="分数",color="b",kind="scatter")


#添加图标标签
plt.xlabel("Hours")
plt.ylabel("Score")
#显示图像
plt.show()

#相关矩阵表
rDf=examDf.corr()
print(rDf)

from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(exam_x,exam_y,train_size=0.8)

from sklearn.linear_model import LinearRegression
model=LinearRegression()

#reshape训练数据
train_x=train_x.values.reshape(-1,1)
test_x=test_x.values.reshape(-1,1)
#截距
model.fit(train_x , train_y)
a=model.intercept_

#回归系数
b=model.coef_


#绘制回归直线
pred_y=model.predict(train_x)
examDf.plot(x="学习时间",y="分数",color="b",kind="scatter")

plt.plot(train_x,pred_y,label="best line",color="k")

#添加图标标签
plt.legend(loc=2)
plt.xlabel("Hours")
plt.ylabel("Score")

#显示图像
plt.show()

model.score(test_x,test_y)