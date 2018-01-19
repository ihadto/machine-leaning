import csv
import time
import pandas as pd
import quandl
import math
import datetime
import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib  import style # importing various librabies 
style.use('ggplot')# setting the style for the graph
dates=[]
prices=[]
x=np.linspace(1,1331,1331)
def get_data(filename):
	with open(filename,'r') as csvfile:
		csvFileReader=csv.reader(csvfile)
		next(csvFileReader)
		for  row in csvFileReader:
			dates.append(int(row[0].split('-')[0]))
			prices.append(float(row[1]))
	return

get_data('rates_data.csv')
#dates=np.reshape(dates,(len(dates),1))
#svr_lin=SVR(kernel='linear',C=1e3)
svr_poly=SVR(kernel='poly',C=1e3,degree=2)
#svr_rbf=SVR(kernel='rbf',C=1e3,gamma=0.1)
#svr_lin.fit(dates,prices)
svr_poly.fit(dates,prices)
#svr_rbf.fit(dates,prices)
plt.scatter(dates,prices,color='black',label='Date')
#plt.plot(dates,svr_rbf.predict(dates),color='red',label='RBF model')	
#plt.plot(dates,svr_lin.predict(dates),color='green',label='Linear model')
plt.plot(dates,svr_poly.predict(dates),color='blue',label='Polynomial model')
plt.scatter(dates,prices)
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('svr')
plt.legend()
plt.show()




















