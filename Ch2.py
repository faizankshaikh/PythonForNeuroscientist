# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 16:01:41 2015

@author: rajat
"""

#2.2.3 Matlab as Calculator
import math

print 2 + 3

print 7 - 5

print 17*4

print 24/7

print 5+3*8

print 2**3

print math.log(2.7183)

print math.log(1)

print math.exp(1)

print math.sin(0)

print math.sin(math.pi/2)

print math.sin((3*math.pi)/2)

print math.exp(math.log(5) + math.log(7))

print 2**500

print 2e3


#2.2.4 Defining Matrices
import numpy as np

print np.array([1,2,3])

print np.array([[2,2,2],[3,3,3]])

a = np.array([1,2,3,4,5])
print a

print a.size  #gives the largest dimension of matrix

print a.shape #gives the size of matrix

b = np.transpose(a)
print b
print b.shape

A = np.array([[7,5],[2,3],[1,8]])
print A
A[1][0] = 2*A[1][0]
print A[1][0]

B = np.transpose(A)
print B

v1 = np.linspace(0,1,num=7)
v2 = np.linspace(0,1,num=10)
print len(v2)
print v2[2]


#2.2.5 Basic Matrix Algebra

import numpy as np
import sympy

p = np.array([[1,2],[3,4]])
print p

p = p+2
print p

q = np.array([[2,1],[1,1]])
print q

m=p+q
print m

r = np.array([[2,1],[1,1],[1,1]])
print r

#n=p+r
#print n  #errors

print p*q  #element wise product
print np.dot(p,q) #matrix multiplication

print p**2

R = np.array([[1,1,2,9],[2,4,-3,1],[3,6,-5,0]])
print sympy.Matrix(R).rref()

C = np.eye(5)
D = np.ones(5)
E = (C+D)**2
print E

x = np.array([[2],[1]])
print x

t = np.linspace(0,100,num=20)
print t
q = []
for i in t:
    q.append(2+5*((i)**(1.7)))
print q

#2.2.6 Indexing
import numpy as np

A = np.array([[1,2,3,4],[5,6,7,8],[10,20,30,40],[50,60,70,80]])
A[0,:] = 23
print A

A[:,0] = 23                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
print A

print A[1,:]>=7

A[1,A[1,:]>=7]=57
print A

A[2,1] = 15
print A


#2.3.1 Basic Visualization


import matplotlib.pyplot as plt
import numpy as np

x = range(11)
y = np.sin(x)
plt.plot(x,y)
plt.show()

x=np.arange(0,10,0.1)
y = np.sin(x)
z = np.cos(x)
plt.plot(x,y)
plt.plot(x,z,color='k')
plt.show()

results = np.array([55,30,10,5])
x = np.arange(len(results))
plt.bar(x,results)
plt.show()

suspicious = np.random.normal(0, 0.1, 100000)
plt.hist(suspicious,100)
plt.show()


#2.4.2 Functions

def triple(i):
    return 3*i;
    
a = triple(7)
print a 

b = triple(np.array([10,20,30]))
print b


#2.4.3 Control Structures

