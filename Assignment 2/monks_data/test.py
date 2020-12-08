#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 16:58:25 2020

@author: haodale
"""
import numpy as np
import os
import graphviz


if __name__ == '__main__':
    # Load the training data
    #M = np.genfromtxt('./monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    #ytrn = M[:, 0]
    #Xtrn = M[:, 1:]

    #print(Xtrn)
    #print(ytrn)
    
    dict = {1:[11,12,13,14]}
    dict.get(1).append(15)
    dict.update({2:[21,22,23,24]})
    dict.get(2).append(25)
    #print(type(dict))
    
    str = "1"
    
    print("hello" + str + "world")
    

    # Load the test data
    '''M = np.genfromtxt('./monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]'''

'''
    # Learn a decision tree of depth 3
    decision_tree = id3(Xtrn, ytrn, max_depth=3)

    # Pretty print it to console
    pretty_print(decision_tree)

    # Visualize the tree and save it as a PNG image
    dot_str = to_graphviz(decision_tree)
    render_dot_file(dot_str, './my_learned_tree')

    # Compute the test error
    y_pred = [predict_example(x, decision_tree) for x in Xtst]
    tst_err = compute_error(ytst, y_pred)

    print('Test Error = {0:4.2f}%.'.format(tst_err * 100))
    
'''