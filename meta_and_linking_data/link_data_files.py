# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 15:42:38 2022

@author: fische_r
"""

import os
# import h5py
import json

EC_folder = r"U:\07_X_ray\005_TOMCAT_2022\02_EC_lab_files"
gas_cue_folder = r"U:\07_X_ray\005_TOMCAT_2022\01_gas_cues"

def extract_sample_names():
    files = os.listdir(EC_folder)
    sample_names = []
    for file in files:
        nosfx = file.split('.')[0]
        name = nosfx.split('_')[1]
        if not name in sample_names:
            sample_names.append(name)
    if 'galv' in sample_names:
        sample_names.remove('galv') #this is an unsed mps file
    return sample_names


def extract_corresponding_files(sample):
    sample_files = []
    files = os.listdir(EC_folder)
    for file in files:
        if file[5:6+len(sample)] == sample+'_':
            sample_files.append(file)
    return sample_files

def extract_gas_cue(path):
    with open(path, 'r') as file:
        lines = file.readlines()
    cuenum = int(lines[1][:-1])
    timestamp = float(lines[3][:-1])
    comment = lines[7]
    return cuenum, timestamp, comment

def read_gas_cues(gas_cue_folder):
    # TODO: add header for the columns
    cues = []
    timestamps = []
    comments = []
    gasfiles = os.listdir(gas_cue_folder)
    # TODO: sort numerically
    for file in gasfiles:
        path = os.path.join(gas_cue_folder, file)
        cuenum, timestamp, comment = extract_gas_cue(path)
        cues.append(cuenum)
        timestamps.append(timestamp)
        comments.append(comment)
    return cues, timestamps, comments
    

    
sample_names = extract_sample_names()

file_dict = {}
cues, timestamps, comments = read_gas_cues(gas_cue_folder)
file_dict['cues'] = {'cue #': cues, 'timestamp': timestamps, 'comment': comments}
file_dict['EC_folder'] = EC_folder
file_dict['samples'] = {}
file_dict['test_samples'] = {'1': {'comment': 'testing setup and without echem, e.g. operando with gas (see logbook)'},'2': {'comment': 'tests, unfortunately gas not connected'}, '3': {'comment': 'beam damage tests'}}

for sample in sample_names:
    if sample == '2': continue
    correct_name = sample
    if sample == '3b':
        correct_name = '3II'
    if sample == '3iii':
        correct_name = '3III'
    if sample == '4ii':
        correct_name = '4II'
    if sample == '5ii':
        correct_name = '5II'
        
    if not sample == '4iib' and not sample == '6b':
        file_dict['samples'][correct_name] = {}
        file_dict['samples'][correct_name]['ec_lab_names'] = [sample]
        file_dict['samples'][correct_name]['ec_lab_files'] = extract_corresponding_files(sample)
    elif sample == '4iib':
        sample_files = extract_corresponding_files(sample)
        file_dict['samples']['4II']['ec_lab_names'] = file_dict['samples']['4II']['ec_lab_names'] + [sample]
        file_dict['samples']['4II']['ec_lab_files'] = file_dict['samples']['4II']['ec_lab_files'] + sample_files
        
    elif sample == '6b':
        sample_files = extract_corresponding_files(sample)
        file_dict['samples']['6']['ec_lab_names'] = file_dict['samples']['6']['ec_lab_names'] + [sample]
        file_dict['samples']['6']['ec_lab_files'] = file_dict['samples']['6']['ec_lab_files'] + sample_files
        
        
# TODO: start and end of each sample

# usually (all?) 5min 2V + 5min 2.5V as conditioning

# cell 4, 3mA constant current
cell = '4'
# pre-operation scans without electrochemistry
file_dict['samples'][cell]['no_gas_no_water'] = 'n.a.'
file_dict['samples'][cell]['gas_no_water'] = 'n.a.'
file_dict['samples'][cell]['gas_water'] = 'T_4_scan_01'
file_dict['samples'][cell]['operando'] = ['T_4_scan_02', 'T_4_scan_03']
file_dict['samples'][cell]['main EC condition'] = 'galv. stat. 3mA'

#cell 5, initially 3mA then ramping up to 20mA
cell = '5'
file_dict['samples'][cell]['no_gas_no_water'] = 'T_5_scan_02'
file_dict['samples'][cell]['gas_no_water'] = 'T_5_scan_03'
file_dict['samples'][cell]['gas_water'] = 'T_5_scan_04'
file_dict['samples'][cell]['operando'] = ['T_5_scan_05','T_5_scan_06', 'T_5_scan_07']
file_dict['samples'][cell]['main EC condition'] = 'galv. stat. initially 3mA, ramped up to 20mA'

#cell 6 constant current 5 mA
cell = '6'
file_dict['samples'][cell]['no_gas_no_water'] = 'T_6_scan_01'
file_dict['samples'][cell]['gas_no_water'] = 'T_6_scan_02'
file_dict['samples'][cell]['gas_water'] = 'T_6_scan_03'
file_dict['samples'][cell]['operando'] = ['T_6_scan_04', 'T_6_scan_05', 'T_6_scan_06']
file_dict['samples'][cell]['main EC condition'] = 'galv. stat. 5mA'


#cell 3II, constant potential 3.5V with 5min 3.9V after about 45min
cell = '3II'
file_dict['samples'][cell]['no_gas_no_water'] = 'T_3b_scan_02'
file_dict['samples'][cell]['gas_no_water'] = 'T_3b_scan_03'
file_dict['samples'][cell]['gas_water'] = 'T_3b_scan_04'
file_dict['samples'][cell]['operando'] = ['T_3b_scan_05', 'T_3b_scan_06', 'T_3b_scan_07']
file_dict['samples'][cell]['main EC condition'] = 'pot. stat. 3.5V'

#cell 5II constant potential 3.5V
cell = '5II'
file_dict['samples'][cell]['no_gas_no_water'] = 'T_5II_scan_01'
file_dict['samples'][cell]['gas_no_water'] = 'T_5II_scan_02'
file_dict['samples'][cell]['gas_water'] = 'T_5II_scan_03'
file_dict['samples'][cell]['operando'] = ['T_5II_scan_04']
file_dict['samples'][cell]['comment'] = 'flooded rotation stage and probably unreliable EC-measurements'
file_dict['samples'][cell]['main EC condition'] = 'pot. stat. 3.5V'

#cell 4II current ramp up, pot overflow at 20mA, added 2 more current values after
cell = '4II'
file_dict['samples'][cell]['no_gas_no_water'] = 'T_4_scan_01'
file_dict['samples'][cell]['gas_no_water'] = 'T_4_scan_02'
file_dict['samples'][cell]['gas_water'] = 'T_4_scan_03'
file_dict['samples'][cell]['operando'] = ['T_4_scan_05']
file_dict['samples'][cell]['comment'] = 'operando scan 4 cancelled within first minute to fix dangerously moving tubing'
file_dict['samples'][cell]['main EC condition'] = 'current ramp up'

#cell 8x IV curve ramping up current to 20mA
cell = '8x'
file_dict['samples'][cell]['no_gas_no_water'] = 'T_8x_scan_01'
file_dict['samples'][cell]['gas_no_water'] = 'T_8x_scan_02'
file_dict['samples'][cell]['gas_water'] = 'T_8x_scan_03'
file_dict['samples'][cell]['operando'] = ['T_8x_scan_04']
file_dict['samples'][cell]['comment'] = 'pinched GDL'
file_dict['samples'][cell]['main EC condition'] = 'current ramp up'

#cell 3III constant potential 3V
cell = '3III'
file_dict['samples'][cell]['no_gas_no_water'] = 'T_3III_01'
file_dict['samples'][cell]['gas_no_water'] = 'T_3III_scan_02'
file_dict['samples'][cell]['gas_water'] = 'T_3III_scan_03'
file_dict['samples'][cell]['operando'] = ['T_3III_scan_04', 'T_3III_scan_05']
file_dict['samples'][cell]['comment'] = 'cathode flow field from former cell 6 due to capillaries ripped off'
file_dict['samples'][cell]['main EC condition'] = 'pot. stat. 3V'


#cell 7x constant current 5mA
cell = '7x'
file_dict['samples'][cell]['no_gas_no_water'] = 'T_7x_scan_01'
file_dict['samples'][cell]['gas_no_water'] = 'T_7x_scan_02'
file_dict['samples'][cell]['gas_water'] = 'T_7x_scan_03'
file_dict['samples'][cell]['operando'] = ['T_7x_scan_04']
file_dict['samples'][cell]['comment'] = 'pinched GDL, gas flow at 10ml/min'
file_dict['samples'][cell]['main EC condition'] = 'galv. stat. 5mA'

cell = '1'
file_dict['samples'][cell] = {}
file_dict['samples'][cell]['test 1'] = 'T_1_scan_01'
file_dict['samples'][cell]['test 2'] = 'T_1_scan_02'
file_dict['samples'][cell]['dry reference'] = 'T_1_scan_03'
file_dict['samples'][cell]['dynamic scan running water'] = 'T_1_scan_04'
file_dict['samples'][cell]['dynamic scan gas+water'] = 'T_1_scan_05'
file_dict['samples'][cell]['comment'] = 'test sample: beam setup and water transport without electrochemistry'

cell = '3'
file_dict['samples'][cell] = {}
file_dict['samples'][cell]['pre-damage'] = 'T_3_scan_02'
file_dict['samples'][cell]['post-damage'] = 'T_3_scan_03'
file_dict['samples'][cell]['comment'] = 'test sample: beam damage tests. Series of 10 s exposures (7) + 1 60s exposure + 10 s before seeing decrease of EC-performance'

# create json file
jsonpath = r"C:\git_repos\01_Python\co2ely-tomcat\meta_and_linking_data\data_overview.json"
with open(jsonpath, "w") as outfile:
    json.dump(file_dict, outfile,indent = 4)
      