# -*- coding: utf-8 -*-

import numpy as np
import regressionData as rg
import time
import pdb

#-------------------
# �N���X�̒��`�n�܂�
class linearRegression():
	#------------------------------------
	# 1) �w�K�f�[�^�����у��f���p�����[�^�̏�����
	# x: �w�K���̓f�[�^�i���̓x�N�g���̎������~�f�[�^����numpy.array�j
	# y: �w�K�o�̓f�[�^�i�f�[�^����numpy.array�j
	# kernelType: �J�[�l���̎��ށi�������Fgaussian�j
	# kernelParam: �J�[�l���̃n�C�p�[�p�����[�^�i�X�J���[�j
	def __init__(self, x, y, kernelType="linear", kernelParam=1.0):
		# �w�K�f�[�^�̐ݒ�
		self.x = x
		self.y = y
		self.xDim = x.shape[0]
		self.dNum = x.shape[1]

		# �J�[�l���̐ݒ�
		self.kernelType = kernelType
		self.kernelParam = kernelParam
	#------------------------------------

	#------------------------------------
	# 2) �ŏ������@���p���ă��f���p�����[�^���œK��
	# �i�����̌v�Z��For�����p�����ꍇ�j
	def train(self):
		self.w = np.zeros([self.xDim,1])
	#------------------------------------

	#------------------------------------
	# 2) �ŏ������@���p���ă��f���p�����[�^���œK���i�s�񉉎Z�ɂ��荂�����j
	def trainMat(self):
		one = np.ones((1,self.dNum))
		x = np.concatenate([self.x,one],0)

		a = np.linalg.inv(np.matmul(x,x.T))
		b = (self.y*x).sum(axis=1)
		self.w = np.matmul(a,b)
	#------------------------------------

	#------------------------------------
	# 3) �\��
	# x: ���̓f�[�^�i���͎��� x �f�[�^���j
	def predict(self,x):
		y = []
		return y
	#------------------------------------

	#------------------------------------
	# 4) ���摹���̌v�Z
	# x: ���̓f�[�^�i���͎��� x �f�[�^���j
	# y: �o�̓f�[�^�i�f�[�^���j
	def loss(self,x,y):
		loss = 0.0
		return loss
	#------------------------------------

	#------------------------------------
	#2つのデータ集合間の全ての組み合わせの距離の計算
	#x: 行列（次元xデータ数）
	#y: 行列（次元xデータ数）
	def calcDist(self,x,z):
		dist = np.matmul(x.T,z)
		return dist
	#------------------------------------

	#------------------------------------
	#カーネルの計算
	#x: カーネルを計算する対象の行列（次元xデータ数）
	def kernel(self,x):
		K = np.exp(-(regression.calcDist(self.x,x))/(2*(self.kernelParam**2)))
		return K
	#------------------------------------

	def trainMatKernel(self):
		K = regression.kernel(self.x)
		one = np.ones((1,self.dNum))
		Kb = np.concatenate([K,one],0)

		ad = np.matmul(Kb,Kb.T)
		#ax = np.full(self.dDim,0.00001)
		#ax = np.diag(ax)
		#ad = ad + ax
		a = np.linalg.inv(ad)
		b = (self.y*Kb).sum(axis=1)
		self.w = np.matmul(a,b)


# �N���X�̒��`�I����
#-------------------

#-------------------
# ���C���̎n�܂�
if __name__ == "__main__":

	# 1) �w�K���͎�����2�̏ꍇ�̃f�[�^�[����
	#myData = rg.artificial(200,100, dataType="1D")
	myData = rg.artificial(200,100, dataType="1D",isNonlinear=True)

	# 2) ���`���A���f��
	#regression = linearRegression(myData.xTrain,myData.yTrain)
	regression = linearRegression(myData.xTrain,myData.yTrain,kernelType="gaussian",kernelParam=1)

	# 3) �w�K�iFor���Łj
	sTime = time.time()
	regression.train()
	eTime = time.time()
	print("train with for-loop: time={0:.4} sec".format(eTime-sTime))

	# 4) �w�K�i�s���Łj
	sTime = time.time()
	regression.trainMat()
	eTime = time.time()
	print("train with matrix: time={0:.4} sec".format(eTime-sTime))

	sTime = time.time()
	regression.trainMatKernel()
	eTime = time.time()
	print("train with matrix and kernel: time={0:.4} sec".format(eTime-sTime))

	# 5) �w�K�������f�����p���ė\��
	print("loss={0:.3}".format(regression.loss(myData.xTest,myData.yTest)))

	# 6) �w�K�E�]���f�[�^�����ї\�����ʂ��v���b�g
	predict = regression.predict(myData.xTest)
	myData.plot(predict,isTrainPlot=False)

#���C���̏I����
#-------------------
