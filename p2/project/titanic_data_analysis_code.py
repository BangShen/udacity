# -*- coding: utf-8 -*-
import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
titanic_data=pd.read_csv(r'D:\self-development\data science\udacity\p2\project\titanic_data.csv')
#titanic_data.head(10)

#这段代码希望能够对数据有个整体的认识，and I pay more attention to if there are NaN in special columns
columns_name=titanic_data.columns
NaN_count = titanic_data.ix[:,:].isnull().sum()
NaN_count_df = pd.DataFrame(NaN_count,columns=['NaN numbers'])
print 'there are %d rows,\nand %d columns in this data set ' % (len(titanic_data),titanic_data.shape[1])
datatype_df = pd.DataFrame(titanic_data.dtypes,columns=['Datatype'])
data_description = pd.merge(NaN_count_df,datatype_df,left_index =True,right_index = True)
data_description


def distribution_plot(data,key='PassengerId',groupname= '',**kwargs):
    '''this fuction aims at drawing the distribtion of folks on the titanic board, function takes a lot of parameters for plot data on various
    variables
    data- means the data set that you want to draw
    key - means the columns name of the data set, this key will be a vital parameter for counting, so it must be exclusive, by defult,it is passengerID
    groupname - this function will show a distribution grouped by groupname
    plot type: this function will only provide a bar plot for each subplot,and all the obtained figures will be arranged in a figure.
    '''
    #check the key exists
    if not groupname:
        raise Exception("hi sorry,you did not provide a key to me, make sure which parameter that u wanna plot")
    if groupname not in data.columns.values:
        raise Exception("there is no '{}' in the data set,did u spell sth. wrong, be attention that all keys in the original data set with an upper case first letter".format(key))
    key_distribution = data[key].groupby(data[groupname]).count()
    fig = key_distribution.plot(kind='bar')
    fig.set_title('%s distribution on the board'%(groupname))
    fig.set_ylabel("Population")
    plt.xticks(rotation = 0)
    #autolabel()
    plt.show()





plt.subplot(121)
plt.scatter(titanic_data['Pclass'],titanic_data['Fare'])

#ticks = ax1.set_xticks([0,1,2,3,4])
plt.subplot(1,2,2)
titanic_data[['Survived','Fare']].boxplot(by = 'Survived')
#ax2.set_ylim([0,200])
plt.show()