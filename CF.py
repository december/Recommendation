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
sim = train * train.T
for i in range(10000):
	sim[i, i] = sim[i, i] ** 0.5
for i in range(10000):
	if i % 200 == 0:
		print i
	for j in range(i+1, 10000):
		'''
		x = train.getrow(i)
		y = train.getrow(j)
		s = x.dot(y.T)[0, 0]
		if s != 0:
			x = (x * x.T)[0, 0]
			y = (y * y.T)[0, 0]
			sim[i, j] = s * 1.0 / ((x * y) ** 0.5)
			sim[j, i] = sim[i, j]
		'''
		if sim[i, j] != 0:
			sim[i, j] = sim[i, j] / sim[i, i] / sim[j, j]
			sim[j, i] = sim[i, j]
print 'Get similarity matrix.'
print datetime.datetime.now() - begin
result = list()
n = len(answer)
print n
numerator = sim * train.T
denominator = sim * sign.T
for i in range(n):
	if i % 10000 == 0:
		print i
	'''
	s = sim.getrow(testuser[i])
	numerator = 0
	denominator = 0
	for j in range(10000):
		if train[j, testflim[i]] != 0:
			numerator += s[j] * train[j, testflim[i]]
			denominator += s[j]
	'''
	result.append(numerator[testuser[i], testflim[i]] / denominator[testuser[i], testflim[i]])
print 'Get answer.'
print datetime.datetime.now() - begin
rmse = 0
for i in range(n):
	rmse += (answer[i] - result[i]) ** 2
print (rmse / n) ** 0.5
print datetime.datetime.now() - begin


