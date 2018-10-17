import numpy as np
from settings import *

'''       
def read_csv(filename):
    with open(filename, 'r') as f:
        lines = [ln.strip().split(",")[-7:-3] for ln in f.readlines()]
        lines = map(lambda x: (x[0], np.float32(x[1:])), lines) # imagefile, outputs
        return lines
'''

def read_csv(filename):
    with open(filename, 'r') as f:
        lines = []
        for ln in f.readlines():
            x = ln.strip().split(",")[-7:-3]
            if x[0].startswith('center'):
                lines.append((x[0],np.float32(x[1:])))
        return lines


def process_csv(filename, val=5):
    sum_f = np.float128([0.0] * OUTPUT_DIM)
    sum_sq_f = np.float128([0.0] * OUTPUT_DIM)
    lines = read_csv(filename)
    # leave val% for validation
    train_seq = []
    valid_seq = []
    cnt = 0
    for ln in lines:
        if cnt < SEQ_LEN * BATCH_SIZE * (100 - val): 
            train_seq.append(ln)
            sum_f += ln[1]
            sum_sq_f += ln[1] * ln[1]
        else:
            valid_seq.append(ln)
        cnt += 1
        cnt %= SEQ_LEN * BATCH_SIZE * 100
    mean = sum_f / len(train_seq)
    var = sum_sq_f / len(train_seq) - mean * mean
    std = np.sqrt(var)
    #print(len(train_seq), len(valid_seq))
    #print(mean, std) # we will need these statistics to normalize the outputs (and ground truth inputs)
    return (train_seq, valid_seq), (mean, std)
