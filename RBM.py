#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 02:49:23 2017

@author: snigdha
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle



def sigmoid(x):
    e_x = np.exp(-x)
    return (1 / (1 + e_x))


# Loading the given train and validation data
string1=[]
string2=[]
string3 = []

f1 = open("digitstrain.txt", 'r')
for line1 in f1:
     string1.append(line1.strip().split(','))
     
f2 = open("digitsvalid.txt", 'r')
for line2 in f2:
     string2.append(line2.strip().split(','))

x_train1, y_train1 = [], []
x_train = np.zeros((3000,784))
x_valid = np.zeros((1000,784))
for subl1 in string1:
    x_train1.append(subl1[:-1])
    y_train1.append([subl1[-1]])

x_valid1, y_valid1 = [], []
for subl2 in string2:
    x_valid1.append(subl2[:-1])
    y_valid1.append([subl2[-1]])
    
x_train1 = np.float64(x_train1)
y_train1 = np.array(y_train1)
x_valid1 = np.float64(x_valid1)
y_valid1 = np.array(y_valid1)

#Shuffling the train data
x_train,y_train = shuffle(x_train1,y_train1, random_state=5)


#Initializations 
num_epochs = 150
learn_rate = 0.1
k = 1  # k is the number of steps of Gibbs steps
numhid = 100
W = np.random.normal(0, 0.1,(784,numhid))
b = 0
c = 0

cross_entr = np.zeros(3000)
cross_entr_valid = np.zeros(1000)

avg_cr_train = np.zeros(num_epochs)
avg_cr_valid = np.zeros(num_epochs)

x_actual = np.zeros((3000,784))
x_predict = np.zeros((3000,784))
x_tilda = np.zeros((784,1))

x_actualv = np.zeros((1000,784))
x_predictv = np.zeros((1000,784))
x_tildav = np.zeros((784,1))


for iter2 in range(0,num_epochs):
    for i1 in range(0,3000):
    
        x_input = x_train[i1,:]
        x_0 = x_input.reshape((784,1))
    
        a1 = np.transpose(np.dot(np.transpose(x_0),W)) + b
        a1r = a1.reshape((100,1))
        h_0 = sigmoid(a1r)
        h_0_sam = np.random.binomial(1, h_0)
    
        h_tilda = h_0
        h_tilda_sam = h_0_sam
        
        for j1 in range(0,k):
                  
                x_tilda = sigmoid(np.dot(W,h_tilda_sam) + c) 
                x_tilda_sam = np.random.binomial(1,x_tilda)
            
                h_tilda = sigmoid(np.dot(np.transpose(W),x_tilda_sam) + b)
                h_tilda_sam = np.random.binomial(1, h_tilda)  
                               
        
        x_actual[i1,:]= x_0.reshape(1,784)
        x_predict[i1,:]= x_tilda.reshape(1,784)
                           
        #updating W, b and c
    
        term1 = np.matmul(x_0,np.transpose(h_0)) - np.matmul(x_tilda_sam,np.transpose(h_tilda))
        W = W + learn_rate*(term1)
        
    
        b = b + learn_rate*(h_0 - h_tilda)
        c = c + learn_rate*(x_0 - x_tilda_sam) 
    
    for l in range(0,3000):
        
        ter1 = (-1)*np.multiply(x_actual[l,:],np.log(x_predict[l,:]))
        ter2 = (-1)*np.multiply((1-x_actual[l,:]),np.log(1-x_predict[l,:]))
        cross_entr[l] = np.sum(ter1 + ter2)
    
    avg_cr_train[iter2] = np.sum(cross_entr)/3000   
                
                
    for i2 in range(0,1000):
    
        x_inputv = x_valid1[i2,:]
        x_0v = x_inputv.reshape((784,1))
    
        a1v = np.transpose(np.dot(np.transpose(x_0v),W)) + b
        a1rv = a1v.reshape((100,1))
        h_0v = sigmoid(a1rv)
        h_0_samv = np.random.binomial(1, h_0v)
                  
        x_tildav = sigmoid(np.dot(W,h_0_samv) + c) 
        
        x_actualv[i2,:]= x_0v.reshape(1,784)
        x_predictv[i2,:]= x_tildav.reshape(1,784)
                              
        
    for l in range(0,1000):
        
        ter1v = (-1)*np.multiply(x_actualv[l,:],np.log(x_predictv[l,:]))
        ter2v = (-1)*np.multiply((1-x_actualv[l,:]),np.log(1-x_predictv[l,:]))
        cross_entr_valid[l] = np.sum(ter1v + ter2v)
    
    avg_cr_valid[iter2] = np.sum(cross_entr_valid)/1000   

numar = np.arange(0,num_epochs,1)
        
plt.figure(2)                
plt.plot(numar, avg_cr_train, 'r', label='Avg Cross Entropy Training Error')
plt.plot(numar, avg_cr_valid, 'b', label='Avg Cross Entropy Validation Error')
plt.title('Observation of average cross-entropy error of training and validation')
plt.xlabel('Number of epochs')
plt.ylabel('Prediction error')
plt.legend()
plt.show()

#W,b and c are saved and visualized with visualization.m in the code folder
