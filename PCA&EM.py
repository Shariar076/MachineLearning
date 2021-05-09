import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


class EM:
	def __init__(self,data, nCluster, nDim):
		self.data=data
		self.nCluster=nCluster
		self.P = np.ones(shape=[len(data),self.nCluster])*(1.0/self.nCluster)
		self.nDim=nDim
		self.miu = np.random.rand(self.nCluster,self.nDim)
		self.sigma=np.random.rand(self.nCluster,self.nDim, self.nDim)
		self.weights=np.ones(self.nCluster)*(1/self.nCluster)
		self.ln_p=0
		self.itr = 0

	def gaussian(self,x,k):
		t=((2*np.pi)**self.nDim)*abs(np.linalg.det(self.sigma[k]))
		a=1/np.sqrt(t)
		sigInv = np.linalg.inv(self.sigma[k])
		dev =x-self.miu[k]
		dev = np.reshape(dev,[2,1])
		t=np.dot(sigInv,dev)
		b = 0.5*np.dot(dev.T,t)[0][0]
		N_k = a * np.exp(-b)
		return N_k

	def E_step(self):
		for i in range(len(self.data)):
			for k in range(self.nCluster):
				self.P[i][k]=self.weights[k]*self.gaussian(self.data[i],k)
			self.P[i]/=np.sum(self.P[i])

	def M_step(self):
		for k in range(self.nCluster):
			miusum=0
			sigsum=0
			psum = 0
			for i in range(len(data)):
				dev = data[i]-self.miu[k]
				dev = np.reshape(dev,[2,1])
				miusum += self.P[i][k]*self.data[i]

				sigsum += self.P[i][k]*np.dot(dev,dev.T)
				psum +=self.P[i][k]
			self.miu[k] = miusum/psum
			self.sigma[k] = sigsum/psum
			self.weights[k]=psum/len(self.P)

	def evaluate(self):
		ln_p=0
		for i in range(len(self.data)):
			err=0
			for k in range(self.nCluster):
				err+=self.weights[k]*self.gaussian(self.data[i],k)
			ln_p+=np.log(err)
		return ln_p

	def print_info(self):
		n_points= np.zeros(shape=[self.nCluster])
		for p in self.P:
			n_points[np.argmax(p)]+=1
		for i in range(self.nCluster):
			print("Cluster ",i)
			print("Mean:")
			print(self.miu[i])
			print("Covariance:")
			print(self.sigma[i])
			print("Number of points:")
			print(n_points[i])
			print("weights:")
			print(self.weights[i])


	def train(self):
		while True:
			self.E_step()
			self.M_step()
			if self.itr%20==0:
				self.plot()
			ln_p=model.evaluate()
			print(ln_p)
			if abs(ln_p - self.ln_p)<0.000001:
				print("Converged")
				self.print_info()
				break
			self.ln_p=ln_p
			self.itr+=1

	def plot(self):
		colmap={0:'r',1:'g',2:'b',3:'c'}
		color=[]
		for p in self.P:
			color=np.append(color,colmap[np.argmax(p)])

		plt.scatter(self.data[:, 0], self.data[:, 1], c=color, s=5)
		plt.scatter(self.miu[0][0], self.miu[0][1], s=500, c='r', marker="d", alpha=0.2)
		plt.scatter(self.miu[1][0], self.miu[1][1], s=500, c='g', marker="d", alpha=0.2)
		plt.scatter(self.miu[2][0], self.miu[2][1], s=500, c='b', marker="d", alpha=0.2)
		plt.scatter(self.miu[3][0], self.miu[3][1], s=500, c='c', marker="d", alpha=0.2)

		for k in range(self.nCluster):
			x, y = np.mgrid[-4:10:.01, -6:6:.01]
			pos = np.empty(x.shape + (2,))
			pos[:, :, 0] = x
			pos[:, :, 1] = y
			rv = multivariate_normal(self.miu[k], self.sigma[k])
			# if (k == 0):
			# 	plt.contour(x, y, rv.pdf(pos), alpha=0.2)
			# elif (k == 1):
			# 	plt.contour(x, y, rv.pdf(pos), alpha=0.2)
			# else:
			plt.contour(x, y, rv.pdf(pos), alpha=0.2)

		filename='./results/iteration'+str(self.itr)+'.png'
		plt.savefig(filename)
		plt.gcf().clear()


orgdata = np.loadtxt('./datasets/cluster.txt')
covar = np.cov(orgdata.T)
eig_vectors= np.linalg.eig(covar)[1]
u_red= eig_vectors[:,:2]
data=np.dot(orgdata,u_red)

plt.scatter(data[:,0], data[:,1],s=5)
plt.savefig('./results/data.png')

model = EM(data, 4, 2)

model.train()

model.plot()

