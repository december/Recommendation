import numpy as np
import datetime
from scipy.sparse import lil_matrix

begin = datetime.datetime.now()
iddic = {}
fw = open('../Project2-data/users.txt')
data = fw.readlines()
fw.close()
cnt = 0
for line in data:
	iddic[line[:-2]] = cnt
	cnt += 1
fw = open('../Project2-data/netflix_train.txt')
data = fw.readlines()
fw.close()
train = lil_matrix((10000, 10000))
sign = lil_matrix((10000, 10000))
#print iddic
cnt = 0
for line in data:
	temp = line.split(' ')
	train[iddic[temp[0]], int(temp[1])-1] = int(temp[2])
	sign[iddic[temp[0]], int(temp[1])-1] = 1
	cnt += 1
	if cnt % 10000 == 0:
		print cnt
print 'Get train matrix.'
fw = open('../Project2-data/netflix_test.txt')
testuser = list()
testflim = list()
answer = list()
data = fw.readlines()
fw.close()
cnt = 0
for line in data:
	temp = line.split(' ')
	testuser.append(iddic[temp[0]])
	testflim.append(int(temp[1])-1)
	answer.append(int(temp[2]))
	cnt += 1
	if cnt % 10000 == 0:
		print cnt
print 'Get the test quest.'
print datetime.datetime.now() - begin
k = 50
lbd = 0.01
alpha = 0.01
J = 1000000
X = train.todense()
U = np.mat(np.random.random(size=(10000, k)))
V = np.mat(np.random.random(size=(10000, k)))
A = sign.todense()
cnt = 0
while J > 100:
	delta = U * V.T - X
	D = np.multiply(A, delta)
	du = D * V + 2 * lbd * U
	dv = D.T * U + 2 * lbd * V
	U = U - alpha * du
	V = V - alpha * dv
	J = 0.5 * (np.linalg.norm(D) ** 2) + lbd * (np.linalg.norm(U) ** 2) + lbd * (np.linalg.norm(V) ** 2)
	cnt += 1
	print cnt
	print J
	Y = U * V.T
	rmse = 0
	for i in range(n):
		rmse += (Y[testuser[i], testflim[i]] - result[i]) ** 2
	print (rmse / n) ** 0.5

