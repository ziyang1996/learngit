from numpy import *
import math
import matplotlib.pyplot as plt 

def load_data(file_X,file_Y):
	fx=open(file_X,"r")
	fy=open(file_Y,"r")
	X=[]
	Y=[]
	for l in fx.readlines():
		l=l.strip().split('	')
		for i in range(len(l)):
			X.append(float(l[i]))

	for l in fy.readlines():
		Y.append(float(l))

	return mat(X).T,mat(Y).T

def poly(X,n):
	_X=[]
	for i in range(X.shape[0]):
		s=[]
		for j in range(n+1):
			s.append(math.pow(X[i],j))
		_X.append(s)
	return mat(_X)


def least_squares(X,Y):
	_X=X.T*X
	_X=_X.I
	_X=(_X*(X.T))
	_X=_X*Y
	return mat(_X)

def show_plot(X,Y,W,Z):
	X=array(X)
	Y=array(Y)
	_X=arange(-2,2,0.05)
	_Y=[]
	for i in range(len(_X)):
		a=_X[i]
		b=float(W[0])+a*float(W[1])+a*a*float(W[2])+a*a*a*float(W[3])+math.pow(a,4)*float(W[4])+math.pow(a,5)*float(W[5])
		_Y.append(b)

	_zx=arange(-2,2,0.05)
	_zy=[]
	for i in range(len(_zx)):
		a=_zx[i]
		b=float(Z[0])+a*float(Z[1])+a*a*float(Z[2])+a*a*a*float(Z[3])+math.pow(a,4)*float(Z[4])+math.pow(a,5)*float(Z[5])
		_zy.append(b)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(X,Y,color='r')
	ax.plot(_X,_Y,'b')
	ax.plot(_zx,_zy,'#ff00ff')
	plt.show()

def load(fileZ):
	z=open(fileZ,'r')
	Z=[]
	for l in z.readlines():
		Z.append(float(l))
	return mat(Z).T


if __name__ =="__main__":
	fx='polydata_data_sampx.txt'
	fy='polydata_data_sampy.txt'
	raw_X,train_Y=load_data(fx,fy)   
	train_X=poly(raw_X,5)
	#print(train_X.shape[0],train_X.shape[1],train_Y.shape[0],train_Y.shape[1])

	fz='polydata_data_thtrue.txt'
	Z=load(fz)

	W=least_squares(train_X,train_Y)
	print(W)

	show_plot(raw_X,train_Y,W,Z)





