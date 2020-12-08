#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 10:50:27 2020

@author: haodale
"""

import os
import math
import re


spamDic = {}
hamDic = {}
spamCount = 0
hamCount = 0
voc = set()


# read training files
def readFile(path, isSpam):
    global spamCount, hamCount
    
    files = os.listdir(path)
    for file in files:
        f = open(os.path.join(path, file), 'r', encoding = "ISO-8859-1")
        content = f.read()

        #only count the meaningful words
        content = re.sub('[^a-zA-Z]', ' ', content)
        content = content.strip().lower()
        
        for word in content.split():
            voc.add(word)
            
            if isSpam:
                spamCount += 1
                count = spamDic.get(word)
                if count is None:
                    count = 0
                count += 1
                spamDic.update({word: count}) 
                
            else :
                hamCount += 1
                count = hamDic.get(word)
                if count is None:
                    count = 0
                count += 1
                hamDic.update({word: count})


# test the NB with test files
def testNB(path, isSpam):
    correct = 0
    total = 0
    
    files = os.listdir(path)
    for file in files:
        total += 1
        p_spam = math.log2(1/2)
        p_ham = math.log2(1/2)
        
        f = open(os.path.join(path, file), 'r', encoding = "ISO-8859-1")
        content = f.read()
        
        #only count the meaningful words
        content = re.sub('[^a-zA-Z]', ' ', content)
        content = content.strip().lower()
        
        for word in content.split():            
            if spamDic.get(word) is None:
                p_spam += math.log2(1 / (spamCount + len(voc)))
            else:
                p_spam += math.log2((spamDic.get(word) + 1) / (spamCount + len(voc)))
                
            if hamDic.get(word) is None:
                p_ham += math.log2(1 / (hamCount + len(voc)))
            else:
                p_ham += math.log2((hamDic.get(word) + 1) / (hamCount + len(voc)))
        
        if isSpam:
            if p_spam > p_ham:
                correct += 1
        else:
            if p_ham > p_spam:
                correct += 1

    return correct, total
    

#main function
if __name__ == '__main__':
    trainSpamPath = 'train/spam'
    trainHamPath = 'train/ham'
    testSpamPath = 'test/spam'
    testHamPath = 'test/ham'

    # read training files and prepare the data for test
    readFile(trainSpamPath, True)
    readFile(trainHamPath, False)
    
    #test for spam and ham test data seperatelly
    spam_correct, spam_total = testNB(testSpamPath, True)
    ham_correct, ham_total = testNB(testHamPath, False)
    
    print("spam accuracy:", spam_correct,"/", spam_total,"=", spam_correct/spam_total)
    print("ham accuracy:", ham_correct,"/", ham_total,"=", ham_correct/ham_total)

    # final result for accuracy
    accuracy = (spam_correct + ham_correct) / (spam_total + ham_total) 
    print ("total accuracy:", accuracy)