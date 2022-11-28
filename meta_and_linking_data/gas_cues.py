# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 16:06:45 2022

@author: fische_r
"""

import os
import time

folder = r"U:\07_X_ray\005_TOMCAT_2022\01_gas_cues"


def gas_cue():
    t = time.time()
    number = len(os.listdir(folder))
    new_num = number+1
    new_file = os.path.join(folder, 'cue_'+str(new_num)+'.txt')
    
    print(new_file)
    print('cue number '+str(new_num))
    print('please write comment')
    
    
    
    with open(new_file, "w") as file:
        # Writing data to a file
        file.write("cue number \n")
        file.write(str(new_num)+"\n")
        file.write('time \n')
        file.write(str(t)+"\n")
        file.write("original path \n")
        file.write(new_file+"\n")
        file.write("please add comment below \n")