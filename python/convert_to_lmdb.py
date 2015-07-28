
"""
A few test manipulation methods:
csv to lmdb
"""

import os.path
import csv
import lmdb
import numpy as np
import caffe
import deepdish as dd

CSV_FILE = '../data/wine.csv'
LMDB_FILE_TRAIN = '../data/wine_train_lmdb'
LMDB_FILE_TEST = '../data/wine_test_lmdb'
LMDB_FILE_DEPLOY = '../data/wine_deploy_lmdb'

#Pythony way to check if files exist
#Since python <3 exceptions
try:
    os.remove(LMDB_FILE_TRAIN)
except OSError:
    pass

try:
    os.remove(LMDB_FILE_TEST)
except OSError:
    pass

try:
    os.remove(LMDB_FILE_DEPLOY)
except OSError:
    pass

#Record CSV
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
        
print "labels[1]", import_labels[1]
train_labels, test_labels, deploy_labels = np.array_split(import_labels, [50, 70])
train_data, test_data, deploy_data = np.array_split(import_data, [50, 70])

map_size = 2**31 #Setting to the max seems to be the only way this doens't throw errors. 

train_blob = train_data.reshape(train_data.shape[0], train_data.shape[1], 1, 1).astype(np.float)
test_blob = test_data.reshape(test_data.shape[0], test_data.shape[1], 1, 1).astype(np.float)
deploy_blob = deploy_data.reshape(deploy_data.shape[0], deploy_data.shape[1], 1, 1).astype(np.float)

k = 0
train_env = lmdb.open(LMDB_FILE_TRAIN, map_size=map_size)
with train_env.begin(write=True) as train_transaction:
    for k in range(len(train_blob)):
        datum = caffe.io.array_to_datum(train_blob[k], int(train_labels[k]))
        str_id = '{:08}'.format(k)
        train_transaction.put(str_id.encode('ascii'), datum.SerializeToString())

k = 0
test_env = lmdb.open(LMDB_FILE_TEST, map_size=map_size)
with test_env.begin(write=True) as test_transaction:
    for k in range(len(test_blob)):
        datum = caffe.io.array_to_datum(test_blob[k], int(test_labels[k]))
        str_id = '{:08}'.format(k)
        test_transaction.put(str_id.encode('ascii'), datum.SerializeToString())

k = 0
deploy_env = lmdb.open(LMDB_FILE_DEPLOY, map_size=map_size)
with deploy_env.begin(write=True) as deploy_transaction:
    for k in range(len(deploy_blob)):
        datum = caffe.io.array_to_datum(deploy_blob[k], int(deploy_labels[k]))
        str_id = '{:08}'.format(k)
        deploy_transaction.put(str_id.encode('ascii'), datum.SerializeToString())

print "Done!"