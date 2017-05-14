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

data = fw.readlines()
fw.close()
test = lil_matrix((10000, 10000))
flag = lil_matrix((10000, 10000))
for line in data:
	temp = line.split(' ')
	test[iddic[temp[0]], int(temp[1])-1] = int(temp[2])
	flag[iddic[temp[0]], int(temp[1])-1] = 1
print 'Get the test quest.'
print datetime.datetime.now() - begin
n = len(answer)
sn = n ** 0.5
k = 50
lbd = 0.01
alpha = 0.01
lastJ = 0
loss = 100
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
while loss > 0.01:
	delta = U * V.T - X
	#print delta[:3,:3]
	D = np.multiply(A, delta)
	#print D[:3,:3]
	du = D * V + 2 * lbd * U
	#print du[:3,:3]
	dv = D.T * U + 2 * lbd * V
	#print dv[:3,:3]
	U = U - alpha * du
	#print U[:3,:3]
	V = V - alpha * dv
	#print V[:3,:3]
	J = 0.5 * (np.linalg.norm(D) ** 2) + lbd * (np.linalg.norm(U) ** 2) + lbd * (np.linalg.norm(V) ** 2)
	cnt += 1
	iterlist.append(cnt)
	print cnt
	losslist.append(J)
	print J
	loss = abs(lastJ-J)
	lastJ = J
	print loss
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
plt.plot(iterlist, rmselist, 'r')
plt.savefig('result_'+str(k)+'_'+str(lbd)+'.png')
plt.cla()
