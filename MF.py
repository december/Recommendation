import numpy as np
import datetime
import matplotlib.pyplot as plt
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
for line in data:
	temp = line.split(' ')
	train[iddic[temp[0]], int(temp[1])-1] = int(temp[2])
	sign[iddic[temp[0]], int(temp[1])-1] = 1
print 'Get train matrix.'
fw = open('../Project2-data/netflix_test.txt')

data = fw.readlines()
fw.close()
test = lil_matrix((10000, 10000))
flag = lil_matrix((10000, 10000))
n = len(data)
for line in data:
	temp = line.split(' ')
	test[iddic[temp[0]], int(temp[1])-1] = int(temp[2])
	flag[iddic[temp[0]], int(temp[1])-1] = 1
print 'Get the test quest.'
print datetime.datetime.now() - begin
sn = n ** 0.5
k = 50
lbd = 0.001
alpha = 0.0001
X = train.todense()
U = np.mat(np.random.normal(size=(10000, k)))
V = np.mat(np.random.normal(size=(10000, k)))
A = sign.todense()
W = test.todense()
F = flag.todense()
cnt = 0
iterlist = list()
losslist = list()
rmselist = list()
while cnt < 1000:
	delta = U * V.T - X
	D = np.multiply(A, delta)
	du = D * V + 2 * lbd * U
	dv = D.T * U + 2 * lbd * V
	U = U - alpha * du
	V = V - alpha * dv
	J = 0.5 * (np.linalg.norm(D) ** 2) + lbd * (np.linalg.norm(U) ** 2) + lbd * (np.linalg.norm(V) ** 2)
	cnt += 1
	iterlist.append(cnt)
	print cnt
	losslist.append(J)
	print J
	'''
	loss = abs(lastJ-J)
	lastJ = J
	print loss
	'''
	Y = U * V.T
	R = np.multiply(Y, F)
	E = R - W
	rmse = np.linalg.norm(E) / sn
	rmselist.append(rmse)
	print rmse
iterlist = np.array(iterlist)
losslist = np.array(losslist)
rmselist = np.array(rmselist)
plt.plot(iterlist, losslist, 'b')
plt.savefig('J_'+str(k)+'_'+str(lbd)+'.png')
plt.cla()
plt.plot(iterlist, rmselist, 'r')
plt.savefig('RMSE_'+str(k)+'_'+str(lbd)+'.png')
plt.cla()
