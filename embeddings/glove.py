import os
import numpy as np

root_folder = os.path.realpath('.')

def get_glove_vectors():
    embeddings_index = dict()
    f = open(root_folder+'/embeddings/glove.6B/glove.6B.300d.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index
