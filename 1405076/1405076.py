import pandas as pd
import numpy as np
from copy import deepcopy


class ALS:
	def __init__(self, X,lamb_u,lamb_v,K):
		self.X = X
		self.K = K
		self.lamb_u = lamb_u
		self.lamb_v = lamb_v
		self.N = np.shape(self.X)[0]
		self.M = np.shape(self.X)[1]
		self.U = np.random.randn(self.N, self.K)
		self.V = np.random.randn(self.K, self.M)

	def update_V(self):
		for i in range(len(self.V[0])):
			rated = np.where(self.X[:, i] != 99)
			sum_u = self.U[rated].T.dot(self.U[rated])
			sum_x = np.zeros(self.K)
			for j in np.ravel(rated):
				sum_x += self.U[j] * self.X[j][i]
			self.V[:, i] = np.dot(np.linalg.inv(sum_u + self.lamb_u * np.identity(self.K)), sum_x)

	def update_U(self):
		for i in range(len(self.U)):
			rated = np.where(self.X[i, :] != 99)
			sum_v = self.V.T[rated].T.dot(self.V.T[rated])
			sum_x = np.zeros(self.K)
			for j in np.ravel(rated):
				sum_x += self.V[:, j] * self.X[i][j]
			self.U[i] = np.dot(np.linalg.inv(sum_v + self.lamb_v * np.identity(self.K)), sum_x)

	def calc_risk(self, X):
		cur_X = self.U.dot(self.V)
		org_X = deepcopy(X)
		unrated = np.where(X == 99)
		org_X[unrated]=cur_X[unrated]
		rms_err = np.sqrt(np.sum(np.square(org_X - cur_X))/(self.M*self.N-len(unrated)))
		return rms_err

	def train(self):
		while self.calc_risk(self.X)>1.0:
			self.update_V()
			self.update_U()
		print("converged at rmse: ",self.calc_risk(self.X))

def train_val_test(data):
	print(np.shape(np.where(data!=99)))
	train_data = np.ones(shape=data.shape)*99
	val_data = np.ones(shape=data.shape)*99
	test_data = np.ones(shape=data.shape)*99
	for i in range(len(data)):
		rated = np.ravel(np.where(data[i] != 99))
		tot_rated = len(rated)
		train_size = int(tot_rated*0.65)+1
		train_ind = np.random.choice(rated, size=train_size)
		train_data[i,train_ind] = data[i,train_ind]

		rated = np.delete(rated,train_ind)
		tot_rated = len(rated)
		val_size = int(tot_rated*0.6)+1
		val_ind = np.random.choice(rated, size=val_size)
		val_data[i,val_ind] = data[i,val_ind]

		rated = np.delete(rated,val_ind)
		tot_rated = len(rated)
		test_size = tot_rated
		test_ind = np.random.choice(rated, size=test_size)
		test_data[i, test_ind] = data[i, test_ind]

	print(np.shape(np.where(train_data!=99)))
	print(np.shape(np.where(val_data!=99)))
	print(np.shape(np.where(test_data!=99)))

	return train_data, val_data, test_data



lambda_v = [0.01, 0.1, 1.0, 10]
lambda_u = [0.01, 0.1, 1.0, 10]
K = [5, 10, 20, 40]

dataset = pd.read_csv('datasets/data.csv', header=None)
data = dataset.values
X = data[:1000,1:]

train_data, val_data, test_data = train_val_test(X)
models = []
i = 1
print("training stage")
for l_v in lambda_v:
	for l_u in lambda_u:
		for k in K:
			print("model: ", i)
			print("lambda_v: ",l_v,"lambda_u: ", l_u, "K: ",k)
			model = ALS(X=train_data, K=k, lamb_u=l_u, lamb_v=l_v)
			model.train()
			models.append(model)
			i+=1


print("\n\nvalidation stage")
model_best = models[0]
curr_risk = 10
model_id = 0
i=1
for model in models:
	print("model: ", i)
	if model.calc_risk(val_data)<curr_risk:
		model_best = model
		curr_risk = model.calc_risk(val_data)
		model_id = i
	i+=1
print("model: ", model_id, "is best")

print("\n\ntesting stage")
print("RMSE of test data: ", model_best.calc_risk(test_data))