# import library
import string
import binascii
import random
import numpy as np
import time
import scipy.optimize as opt
import math
from collections import defaultdict
from itertools import combinations
import sys
from multiprocessing import Pool
import multiprocessing
from functools import partial

# Declare gobal variable
DATA_SIZE = [100, 1000, 2500, 10000]
NUM_HASH = 100
THRESHOLD = 0.7

NUM_PROCESS = 1

# jarccard similarity
def jarccard_sim(s1, s2):
    '''
    calculate jarccard similarity
    > input: 
        > s1: set
        > s2: set
    > output:
        > j_sim: jarccard similarity
    '''
    j_sim = (len(s1.intersection(s2)) / len(s1.union(s2)))
    return j_sim
# ---------------------------------------------------------------------------------------

# data parsing
def parse_data(filename):
    '''
    read articles data and return an array of tuples
    > input:
        > filename: file stored articles data
    > output:
        > doc_list: an array include tuples with id and processed string
    '''
    doc_list = []
    # read 1 article per line
    with open(filename, "r") as f:
        for line in f:
            # get id and text
            id, text = line.split(" ",1)
            # remove all pynctuation in text
            text = text.translate(str.maketrans('', '', string.punctuation))
            # change all letters to lowercase
            text = text.lower()
            # remove all white space
            text = text.strip("\t\b\n")
            text = "".join(text.split(" "))
            # add to document list
            doc_list.append((id,text))
    f.close()
    return doc_list
# ---------------------------------------------------------------------------------------

# shingling
def shingle_document(document, k = 9):
    '''
    Converts a document into a set representation
    > input:
        > document: document string
        > k: length of k-shingle in set representation
    > output:
        > a set of shingles
    '''
    # init set
    shingles = set()
    # for each position in string
    for index in range(0, len(document) - k):
        # create shingle with length k
        k_shingle = document[index: index + k]
        # hash k_shingle by CRC32 hash function
        crc = binascii.crc32(k_shingle.encode('utf8')) & 0xffffffff
        # insert into set
        shingles.add(crc)
    # return set
    return shingles
# ---------------------------------------------------------------------------------------

# read all article and shingling it into shingles
def read_document_and_shingling(num_doc):
    '''
    read document and shingling it into shingles
    > input:
        > num_doc: number of document.
    > output:
        > shingled_document: a dictionary include doc_id's shingles set
        > doc_id: list of document id
    '''
    # read documents
    documents = []
    data_filename = "./data/articles_" + str(num_doc) + ".txt"
    documents = parse_data(data_filename)
    # init
    shingled_document = {}
    doc_id = []
    doc_shingles = set()
    # for each document
    for doc in documents:
        # shingling it
        doc_shingles = shingle_document(doc[1])
        # save 
        doc_id.append(doc[0])
        shingled_document[doc[0]] = doc_shingles
    # return 
    return shingled_document, doc_id
# ---------------------------------------------------------------------------------------

# generate hash function
def make_random_hash_fn(m = 2**32):
    a = random.randint(1, m - 1)
    b = random.randint(0, m - 1)
    return [a,b]

def hash(row,hash_fn):
    m = 2**32
    result = (hash_fn[0] * row + hash_fn[1]) % m
    return result
# ---------------------------------------------------------------------------------------

def get_signature(shingled_data, doc_id, num_hash, hash_fn, doc_index_start, doc_index_end, signature_matrix):
    '''
    '''
    for doc_index in range(doc_index_start, doc_index_end):
        # get document id
        id = doc_id[doc_index]
        # init sinatures with infinite number (number that > shingle for all shingle in shingled_data)
        inf = 2**32 + 1
        signatures = np.full([num_hash], inf)
        # for each row in shingles of each document
        for row in shingled_data[id]:
            # with each hash function
            for hash_index in range(num_hash):
                # calculate permutation
                hash_code = hash(row, hash_fn[hash_index])
                # update signature matrix with min value
                signatures[hash_index] = min(signatures[hash_index], hash_code)
        # add signatures into matrix
        for index, sign in enumerate(signatures):
            signature_matrix[doc_index*num_hash + index] = sign

# ---------------------------------------------------------------------------------------

# minhash
def minhash(shingled_data, doc_id , num_hash):
    '''
    minhash : hashing shingles set of documents into singnature matrix
    > input:
        > shingled_data: shingles set of documents
        > doc_id: list of document id
        > num_hash: number of hash function used
    > output:
        > signature_matrix: Minhash signature matrix as a numpy 2d array
    '''
    # init signature matrrix
    num_doc = len(doc_id)
    signature_matrix = multiprocessing.Array('Q', num_doc * num_hash)

    hash_fn = [make_random_hash_fn() for i in range(0, num_hash)]

    # minhasing
    
    num_work = int(num_doc / NUM_PROCESS)
    extra_work = int(num_doc % NUM_PROCESS)
    cur = 0
    p = []

    for i in range(NUM_PROCESS):
        if (i < extra_work):
            next = cur + num_work + 1
        else:
            next = cur + num_work
        p.append(multiprocessing.Process(target=get_signature,args=(shingled_data,doc_id,num_hash,hash_fn, cur, next, signature_matrix)))
        cur = next


    for i in range(NUM_PROCESS):
        p[i].start()

    for i in range(NUM_PROCESS):
        p[i].join()

    signature_matrix = np.array(signature_matrix)
    signature_matrix = np.reshape(signature_matrix, (-1, num_hash))

    # return signature matrix
    return np.array(signature_matrix)
# ---------------------------------------------------------------------------------------

# choose band
def choose_nbands(t, n):
    '''
    Choose b and r to get the best S-curve i.e
    minimum false negative and false positive rate
    > input:
        > t: threshold
        > n: signature length
    > output:
        > b: band
        > final_t: reality threshold
    '''
    def error_fun(x):
        cur_t = (1/x[0])**(x[0]/n)
        return (t-cur_t)**2

    opt_res = opt.minimize(error_fun, x0=(10), method='Nelder-Mead')
    b = int(math.ceil(opt_res['x'][0]))  
    r = int(n / b)
    final_t = (1/b)**(1/r)
    return b, final_t
# ---------------------------------------------------------------------------------------

# generate hash vector function
def make_vector_hash():
    def f(vec):
      return ''.join(map(str, vec))
    return f
# ---------------------------------------------------------------------------------------

# locality sensitive hashing
def lsh(minhash_sign_mt, doc_id, num_hash, threshold):
    '''
    Choose b and r to get the best S-curve i.e 
    minimum false negative and false positive rate
    '''
    # choose band and row
    b, _ = choose_nbands(threshold, num_hash)
    r = int(num_hash / b)
    # generate a ramdom hash function that takes vector of length r
    hash_func = make_vector_hash()
    # init candidates pair
    candidates = set()
    # hashing on each band
    for band in range(0, b):
        # init bucket for each band
        bucket = defaultdict(list)
        # calculate position of vector 
        start_index = int(band * r)
        end_index = min(start_index + r, num_hash)
        # hash into bucket and add candidate pairs
        for index, id in enumerate(doc_id):
            # get small signature vector of this band
            hash_vector = minhash_sign_mt[index, start_index : end_index]
            # hash vector above
            hash_value = hash_func(hash_vector)
            # insert document id into bucket
            bucket[hash_value].append(id)
        # add any two doc in the same bucket into candidate pairs
        for values in bucket.values():
            candidates.update(combinations(values, 2))
    # return candidate pairs 
    return candidates
# ---------------------------------------------------------------------------------------

# finding similar document
def similar_document_searching(document_size, num_hash, threshold):
    '''
    finding similar document
    > input:
        > document_size: document data size file input
        > num_hash: number of hash function used in minhash and lsh
        > threshold: similarity threshold
    > output:
        > candidate_pairs: a set of pair of similar document
    '''
    # read documents and shingle it 
    t0 = time.time()
    print("\nRead Documents and Shingling ...")
    shingled_docs, doc_id = read_document_and_shingling(document_size)
    elapsed = (time.time() - t0)
    print ("Read Documents and Shingling time: %.2fsec" % elapsed)

    # minhash shinles of document into signatures
    t0 = time.time()
    print("\nMinHashing ...")
    signatures = minhash(shingled_docs, doc_id, num_hash)
    elapsed = (time.time() - t0)
    print ("MinHashing Time: %.2fsec" % elapsed)

    # generate candidate pairs by locality sensitive hash
    t0 = time.time()
    print("\nLocality Sensitive Hashing ...")
    candidate_pairs = lsh(signatures, doc_id, num_hash, threshold)
    elapsed = (time.time() - t0)
    print ("Locality Sensitive Hashing Time: %.2fsec" % elapsed)

    # calulate similarity
    result = {}
    t0 = time.time()
    print ("\nCalculating Jarccard Similarity ...")
    for pair in candidate_pairs:
        j_sim = jarccard_sim(shingled_docs[pair[0]], shingled_docs[pair[1]])
        if j_sim >= threshold:
            result[pair] = j_sim
    elapsed = (time.time() - t0)
    print ("Calculating Jarccard Similarity Time: %.2fsec" % elapsed)
    # return set of pair of similar document
    return result
# ---------------------------------------------------------------------------------------   

# main function
if __name__ == "__main__":    
    # check input value
    if len(sys.argv) != 3:
        print("Invalid input\ntry: $python3 minhash_lsh_p_pool.py <number of documents> <number of pool>")
        exit()
    if int(sys.argv[1]) not in DATA_SIZE:
        print("Document's size must be one of following value: 100\t1000\t2500\t10000")
        exit()
    
    if int(sys.argv[2]) not in range(1, 48):
        print("Invalid number of pool\nNumber of pool must be integer in range 1..48")

    # init value
    document_size = int(sys.argv[1])
    num_hash = NUM_HASH
    threshold = THRESHOLD

    NUM_PROCESS = int(sys.argv[2])

    print("Prossing %d document with %d processor:" % (document_size, NUM_PROCESS))

    # find similar document
    t0 = time.time()
    result = similar_document_searching(document_size, num_hash, threshold)
    elapsed = (time.time() - t0)

    print ("\nTotal Processing Time: %.2fsec" % elapsed)
    print("Check ./data/aritcles_%d.truth.txt for the result!" % document_size)

    # print result
    
    print("\nResuslt:\nEach pair of document are similar aproximately at least %.2f" % threshold)
    print("\tDoc ID\t<-->\tDoc ID\t\tJarccard Sim")


    for pair in result:
        print("\t%s\t<-->\t%s\t\t%.3f"  % (pair[0], pair[1], result[pair]))
