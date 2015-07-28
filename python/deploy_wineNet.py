"""
Test the wine classifier (post training)
"""

import pprint
import numpy as np
import caffe
import os.path
import termcolor
import random
import csv

CSV_FILE = '../data/wine.csv'
TRAIN_LMDB_FILE = '../data/wine_train_lmdb'
TEST_LMDB_FILE = '../data/wine_test_lmdb'
NET_DEPLOY_FILE = '../net/wineNet_deploy.prototxt'
NET_WEIGHTS_FILE = '../net/_iter_350000.caffemodel'

caffe.set_mode_cpu()
net = caffe.Net(NET_DEPLOY_FILE, NET_WEIGHTS_FILE, caffe.TEST)
wrong = []

#Reimport for comparison
with open(CSV_FILE) as csvfile:
    csv_reader = csv.reader(csvfile)

    num_rows = sum(1 for row in csv_reader)
    csvfile.seek(0)
    num_training = 70
    num_test = num_rows - num_training


    import_labels = np.zeros(num_rows, dtype=np.int64) 
    import_data = np.ones((num_rows, 13, 1,1), dtype = np.float)
    
    i = 0
    j = 0
    k = 0
    for row in csv_reader:
        
        import_labels[i] = row[0]
        for j in range(12):
            import_data[i][j] = row[j+1]
        j=0
        i = i + 1

#Loop over import data and compare
for i in range(len(import_data)):
	datum = import_data[i]
	datum = np.asarray([datum.reshape(datum.shape[0], 1, 1)])
	out = net.forward(**{'data': datum})
	prob_distr = out['ip2'][0]
#	if (prob_distr[0] > prob_distr[1] and import_labels[i] == 1) or (prob_distr[0] < prob_distr[1] and import_labels[i] == 0):
#		color = "red"
#	else:
#		color = "green"
	color = "yellow"
	termcolor.cprint("Results: " + str(prob_distr) + ", Label: " + str(import_labels[i]) + ", Confidence: " + str(max(prob_distr[0], prob_distr[1], prob_distr[2])), color)








