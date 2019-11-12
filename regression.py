# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 16:22:58 2019

@author: Lee
"""

from pyecharts import options as opts
from pyecharts.charts import Geo,Line,WordCloud,Pie,Parallel,PictorialBar,Bar,Polar
from pyecharts.globals import ChartType, SymbolType

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
import math
from sklearn.metrics import r2_score


bar1=(
        Bar()
        .add_xaxis([str(i)+'年' for i in range(2009,2019)])
        .add_yaxis("销售额（亿元）",[0.5,9,52,191,350,571,912,1207,1682,2135], category_gap="50%")
        .set_global_opts(title_opts=opts.TitleOpts(title="天猫“双十一”成交额变化趋势图"))
    )
#bar1.render('成交额.html')

x=[i for i in range(1,11)]
#x=[math.log(j) for j in x]
#print(x)
#print(sum(x))
x=np.reshape(x,(-1,1))

y=[0.5,9,52,191,350,571,912,1207,1682,2135]
#y=[math.log(i) for i in y]
#print(y)
#print(sum(y))

#x=[i for i in range(1,11)]
#x=[math.log(j) for j in x]
#y=[0.5,9,52,191,350,571,912,1207,1682,2135]
#y=[math.log(i) for i in y]
#total=[]
#for i in range(10):
#    total.append(x[i]*y[i])
#print(total)    
#print(sum(total))
#print(np.sqrt(4096.3434/8))
#print(math.exp(math.log(11)*3.623-0.279))
#print(math.exp(-0.279)*(11**3.623))
plt.scatter(x,y,color='red')
plt.xlabel('transform year')
plt.ylabel('transform total GMV')

#regr=linear_model.LinearRegression()
#regr.fit(x,y)
#plt.plot(x,regr.predict(x),color='black')
#print('b:{}'.format(regr.intercept_))
#print('w:{}'.format(regr.coef_))
#predict=regr.coef_[0]*math.log(11)+regr.intercept_
#print(math.exp(predict))

pf = PolynomialFeatures(degree=2)
x_2_fit = pf.fit_transform(x)
print(x_2_fit)
lrModel = linear_model.LinearRegression()
lrModel.fit(x_2_fit,y)

x_2_predict = pf.fit_transform(np.reshape([11],(-1,1)))
print("多项式方程为:{:.3f}x²{:.3f}x+{:.3f}".format(lrModel.coef_[2],lrModel.coef_[1],lrModel.intercept_))
print(lrModel.coef_[2])
print(lrModel.intercept_)
lrModel.predict(x_2_predict)
print(lrModel.predict(x_2_predict))

y_predicted=[lrModel.predict(pf.fit_transform(np.reshape([i],(-1,1))))[0] for i in range(1,11)]
score = r2_score(y, y_predicted, multioutput='raw_values')
print(score)
plt.plot(x,y_predicted,color='black')
