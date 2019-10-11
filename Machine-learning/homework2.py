from numpy import *
import random
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR

def load_data(fileX,fileY):  
	#load training data from file and generate feature matrix X and feature matrix Y
	fx = open(fileX,'r')
	fy = open(fileY,'r')
	X=[]
	Y=[]
	n=0
	ok=1
	for l in fx.readlines():
		_X=[]
		_x=[]
		ls=l.strip().split('	')
		for i in range(len(ls)):
			a=float(ls[i])
			_X.append(a)
			_x.append(a*a)  #添加二次项
		if ok==1:
			_A=[]
			for i in range(len(ls)):
				_A.append(1)
			ok=0
			X.append(_A)
		X.append(_X)
		X.append(_x)
	fx.close()

	for l in fy.readlines():
		Y.append(float(l))
	fy.close()
	return mat(X).T,mat(Y).T

def standard_variance(_Y,Y):
	n=Y.shape[0]
	e=(Y-_Y)
	E=math.sqrt(e.T*e/n)
	return E

def Least_square_method(X,Y):
	_X=X.T*X
	_X=_X.I
	_X=(_X*(X.T))
	_X=_X*Y
	return mat(_X)


if __name__ =="__main__":
	fx='count_data_trainx.txt'
	fy='count_data_trainy.txt'
	train_X,train_Y=load_data(fx,fy)   

	fx='count_data_testx.txt'
	fy='count_data_testy.txt'
	test_X,test_Y=load_data(fx,fy)
	#print(train_X)
	#print(train_X.shape[0],train_X.shape[1])

	#最小二乘法
	W=Least_square_method(train_X,train_Y)
	#print(W)
	print("least square:")
	print(standard_variance(train_X*W,train_Y),end='\t')
	print(standard_variance(test_X*W,test_Y))
	

	


	# SGD 随机梯度下降
	sgd = SGDRegressor(loss='squared_loss', penalty=None, random_state=42)
	sgd.fit(train_X,train_Y)
	y_verify=sgd.predict(train_X)
	y_predict=sgd.predict(test_X)
	#print(mat(y_predict).T)
	print("SGD:")
	print(standard_variance(mat(y_verify).T,train_Y),end='\t')
	print(standard_variance(mat(y_predict).T,test_Y))


	#SVM 支持向量机
	svr_linear = SVR(kernel='linear')#线性核,可以选用不同的核如poly,rbf
	svr_linear.fit(train_X,train_Y)
	y_verify=svr_linear.predict(train_X)
	y_predict=svr_linear.predict(test_X)
	#print(mat(y_predict).T)
	print("SVM:")
	print(standard_variance(mat(y_verify).T,train_Y),end='\t')
	print(standard_variance(mat(y_predict).T,test_Y))
