#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 10:50:27 2020

@author: haodale
"""

import os
import math
import re
import sys

X = []
Y = []
W = {}
W_new = {}

lamda = 0.01
mu = 0.01 #learning reate
iterate_num = 10
stopWords = set()
voc = set()


# read training files and prepare the data for training, denote spam Y = 0, ham Y = 1
def readFile(path, isSpam):    
    files = os.listdir(path)
    for file in files:
        f = open(os.path.join(path, file), 'r', encoding = "ISO-8859-1")
        content = f.read()

        #only count the meaningful words
        content = re.sub('[^a-zA-Z]', ' ', content)
        content = content.strip().lower()
        
        row = {}
        row["0"] = 1.0
        for word in content.split():
            # filter out the stop words
            if word in stopWords:
                continue
            
            # get distinct words, put them in voc set
            voc.add(word)
            
            # as the voc sequence to form the X and Y vectors
            if row.get(word) is None:
                row[word] = 0
            row[word] += 1
        
        X.append(row)
        if isSpam:
            Y.append(0)
        else:
            Y.append(1)

 
# calculate the prob P(Y=1|Xl,W)
def calProb(xl):
    global W
    WX = W["0"]
    for word in xl:
        if word in W:
            WX += W[word] * xl[word]

    #avoid overflow
    if WX >= 700:
        return 1
    
    exp_WX = math.exp(WX)
    prob = exp_WX / (1 + exp_WX)
    return prob


# train LR to get w vector
def trainLR():
    global W, W_new
    W_new = W.copy()
    print('running iteration: ', end=" ", flush = True)
    for i in range(iterate_num):
        print((i + 1), end=" ", flush = True)
        for word in W:
            sum = 0.0
            for l in range(len(Y)):
                yl = Y[l]
                xl = X[l]
                if word in xl:
                    prob = calProb(xl)
                    sum += xl[word] * (yl - prob)
            W_new[word] = W[word] + mu * sum - lamda * mu * W[word]
        W = W_new.copy()
    print()


# test the logistic regression with test files
def testLR(path, isSpam):
    correct = 0
    total = 0
    
    files = os.listdir(path)
    # get distinct words, put them in voc set
    for file in files:
        total += 1
        f = open(os.path.join(path, file), 'r', encoding = "ISO-8859-1")
        content = f.read()

        #only count the meaningful words
        content = re.sub('[^a-zA-Z]', ' ', content)
        content = content.strip().lower()
        
        row = {}
        for word in content.split():
            # filter out the stop words
            if word in stopWords:
                continue
            
            if word in W:
                if row.get(word) is None:
                    row[word] = 0
                row[word] += 1
        
        WX = W["0"]
        for word in row:
            WX += W[word] * row[word]
        
        if isSpam:
            if WX <= 0:
                correct += 1
        else:
            if WX > 0:
                correct += 1
    
    return correct, total

    
# load stop words 
def loadStopWords(path):
    f = open(path, 'r', encoding = "ISO-8859-1")
    content = f.read().strip().lower()
        
    for word in content.split():
        stopWords.add(word)


#main function
if __name__ == '__main__':
    #parameter 1 for lamda, parameter 2 for iteration times
    lamda = float(sys.argv[1])
    iterate_num = int(sys.argv[2])
    
    trainSpamPath = 'train/spam'
    trainHamPath = 'train/ham'
    testSpamPath = 'test/spam'
    testHamPath = 'test/ham'
    
    stopWordPath = './stopwords.txt'
    
    #load stop words into a set
    loadStopWords(stopWordPath)

    # read training files and prepare the data for training
    readFile(trainSpamPath, True)
    readFile(trainHamPath, False)
    
    # initial W vector all item to be 0
    W["0"] = 0.0
    for word in voc:
        W[word] = 0.0
    
    # train LR
    trainLR()
    
    #test for spam and ham test data seperatelly
    spam_correct, spam_total = testLR(testSpamPath, True)
    ham_correct, ham_total = testLR(testHamPath, False)
    
    print("spam accuracy:", spam_correct,"/", spam_total,"=", spam_correct/spam_total)
    print("ham accuracy:", ham_correct,"/", ham_total,"=", ham_correct/ham_total)
    
    # final result for accuracy
    accuracy = (spam_correct + ham_correct) / (spam_total + ham_total) 
    print ("total accuracy:", accuracy)