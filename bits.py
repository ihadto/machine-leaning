import pandas as pd
import quandl
from sklearn.svm import SVR
import datetime
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from sklearn.cross_validation import train_test_split
from matplotlib  import style # importing various librabies 
style.use('fivethirtyeight')# setting the style for the graph
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
df=quandl.get("BITSTAMP/USD", authtoken="v-k2WeDGnhZ3ifPyK7sq")
df['dayno']=np.linspace(1,1331,1331)
x=df['dayno']
y=df['High']
plt.legend(loc =-4)
plt.title('BITCOIN ORIGINAL POSITION')
plt.xlabel('DATE')
plt.ylabel('PRICE')
plt.plot(df['High'],'k.')
plt.grid(True)
plt.show()# graph of how the data is over 
fp1, residuals, rank, sv, rcond = sp.polyfit(x, y, 1, full=True)
print("Model parameters: %s" % fp1)
f1 = sp.poly1d(fp1)
fx = sp.linspace(1,1331, 1331)
plt.plot(fx, f1(fx), linewidth=4)
plt.legend(["d=%i" % f1.order], loc="upper left")
plt.title('BITCOIN ANALYSIS BY LINEAR REGRESSION')
plt.xlabel('DAYNO')
plt.ylabel('PRICE')
plt.show()# we got a straight line
f2p = sp.polyfit(x, y, 2)
f2 = sp.poly1d(f2p)
plt.plot(fx, f2(fx), linewidth=4)
plt.legend(loc =-4)
plt.title('BITCOIN ANALYSIS FOR QUADRATIC POLYNOMIAL')
plt.xlabel('DAYNO')
plt.ylabel('PRICE')
plt.show()# by using the polynomial of degree two
fx = sp.linspace(1332,1440, 68)
plt.plot(fx, f2(fx), linewidth=4)
plt.legend(loc =-4)
plt.title('BITCOIN FUTURE ANALYSIS FOR NEXT 68 DAYS')
plt.xlabel('DAYNO')
plt.ylabel('PRICE')
plt.show()# 
def error(f, x, y):
	return sp.sum((f(x)-y)**2)
print(error(f1, x, y))
print(error(f2, x, y))
# analysis done
