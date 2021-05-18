import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.notebook import tqdm
from sklearn.utils import shuffle
from scipy.stats import rv_discrete


import scipy
from scipy import sparse
from itertools import chain, combinations, product
from scipy.special import comb
import itertools


############################################################# ML UTILS ###############################################################

def sklearn_tokenize(seqs, AAs='ACDEFGHIKLMNPQRSTVWY*'):
    """
    Takes an iterable of sequences provided as one amino acid strings and returns
    an array of their tokenized form.

    TODO: Vectorize the operation

    Parameters
    ----------
    seqs : iterable of strings

        Iterable containing all strings

    AAs  : str, default="ACDEFGHIKLMNPQRSTVWY"

        The alphabet of permitted characters to tokenize. Single amino acid codes
        for all 20 naturally occurring amino acids is provided as default.

    returns : np.array(tokenized_seqs)

        Returns a tokenized NP array for the sequences.
    """
    tokens = {x:y for x,y in zip(AAs, list(range(1, len(AAs)+1)))}
    return np.array([[tokens[aa] for aa in seq] for seq in seqs])




def one_hot(sequence_set, AAs = 'ACDEFGHIKLMNPQRSTVWY'): 
    """Returns one-hot encoding matrix for an iterable of sequences of amino acids.
    
    sequence_set:             iterable of sequences
    AAs:                      accessible amino acid alphabet (iterable)"""
    
    encoding_matrix = np.diag(np.ones(len(AAs)))
    encoding_dict   = {x:y for x,y in zip(AAs, encoding_matrix)}
    
    all_seqs = []
    
    for seq_index, sequence in tqdm(enumerate(sequence_set)): 
        seq = []
        for index, amino in enumerate(sequence):
            seq.append(encoding_dict[amino])
        all_seqs.append(np.array(seq).astype('int8')) # we use  int8 dtype to reduce memory usage
    return np.array(all_seqs).astype('int8')



def preprocess(pathtofile, train_split): 
    data      = pd.read_csv(pathtofile)
    data      = data.dropna() #remove rows (sequence reads) with any empty cell in that row (usually 
                    # a lack of fitness reading)
    
    seq_list1 = data['Sequence'].tolist()
    #print(seq_list1)
    fit_list1 = data['Fitness'].tolist()
    sequences = []

    
    sequences = [seq_list1[i] for i in range(len(seq_list1)) if ('*' not in seq_list1[i]) and (len(seq_list1[i])==len(seq_list1[0]))]   #assuming first seq has correct reference len
    fitnesses = [fit_list1[i] for i in range(len(seq_list1)) if ('*' not in seq_list1[i]) and (len(seq_list1[i])==len(seq_list1[0]))]
    new_df    = pd.DataFrame(sequences, fitnesses)
    print('{} sequences with stop codon substitutions removed.'.format(len(seq_list1)-len(sequences)))
    
    sequences_onehot = one_hot(sequences)
    all_data         = [(seq,fitness) for seq, fitness in zip(sequences_onehot, fitnesses)
                       if '*' not in seq]
    
    train_test_cutoff = int((len(all_data)/100)*train_split)
    shuffled   = shuffle(all_data)
    train_data = shuffled[:train_test_cutoff]
    test_data  = shuffled[train_test_cutoff:]
    train_X    = np.array([i for i,j in train_data])
    train_Y    = np.array([j for i,j in train_data])
    test_X     = np.array([i for i,j in test_data])
    test_Y     = np.array([j for i,j in test_data])
    
    return train_X, train_Y, test_X, test_Y



def power_2(start, stop): 
    """Returns uniform distriution over powers of 2 from start to stop. Ensure stop is a power of 2 of start."""
    powers = []
    powers.append(start)
    a = start
    while stop not in powers: 
        powers.append(powers[-1]*2)
        
    prs  = [1/len(powers) for _ in powers]
    custm = rv_discrete(name='custm', values=(powers, prs))
    return custm   



################################################################## LANDSCAPE/GRAPH UTILS ##################################################################



def hamming_circle(s, n, alphabet):
    """Generate strings over alphabet whose Hamming distance from s is
    exactly n. 
    (Function taken direct from StackExchange -- https://codereview.stackexchange.com/questions/88912/create-a-list-of-all-strings-within-hamming-distance-of-a-reference-string-with)
    >>> sorted(hamming_circle('abc', 0, 'abc'))
    ['abc']
    >>> sorted(hamming_circle('abc', 1, 'abc'))
    ['aac', 'aba', 'abb', 'acc', 'bbc', 'cbc']
    >>> sorted(hamming_circle('aaa', 2, 'ab'))
    ['abb', 'bab', 'bba']
    """
    for positions in combinations(range(len(s)), n):
        for replacements in product(range(len(alphabet) - 1), repeat=n):
            cousin = list(s)
            for p, r in zip(positions, replacements):
                if cousin[p] == alphabet[r]:
                    cousin[p] = alphabet[-1]
                else:
                    cousin[p] = alphabet[r]
            yield ''.join(cousin)




def all_genotypes(N, AAs):
    """Generates all possible genotypes of length N over alphabet AAs ."""
    return np.array(list(itertools.product(AAs, repeat=N)))

def custom_neighbors(sequence, sequence_space, d):
    """Search algorithm for finding sequences in sequence_space that are exactly Hamming distance d from sequence.
    This is a possibly a slow implementation -- it might be possible to obtain speed-ups."""
    hammings = []
    for i in sequence_space:
        hammings.append((i, hamming_distance(sequence,i)))
    return [sequence  for  sequence, distance in hammings if distance==d]



def get_graph(sequenceSpace, AAs):
    """Get adjacency and degree matrices for a sequence space. This creates a Hamming graph by connecting all sequences 
    in sequence space that are 1 Hamming distance apart. Returns a sparse adjacency and degree matrix (which can be used
    for downstream applications e.g. Laplacian construction).
    sequenceSpace:      iterable of sequences
    returns:            tuple(adjacency matrix, degree matrix), where each matrix is a scipy sparse matrix """
  
    seq_space  = [''.join(i) for i in sequenceSpace]
    nodes      = {x:y for x,y in zip(seq_space, range(len(seq_space)))}
    adjacency  = sparse.lil_matrix((len(seq_space), len(seq_space)), dtype='int8') 
    
    for ind in tqdm(range(len(seq_space))):        
        seq = seq_space[ind]     

        for neighbor in hamming_circle(seq, 1,AAs): 
            adjacency[ind,nodes[neighbor]]=1 #array indexing and replacing can be quite slow; consider using lists

       # degree_matrix = (l*(a-1))*sparse.eye(len(seq_space)) #this definition comes from Zhou & McCandlish 2020, pp. 11
    return adjacency #returns adjacency







def csvDataLoader(csvfile,x_data="Sequence",y_data="Fitness"):
    """Simple helper function to load NK landscape data from CSV files into numpy arrays.
    Supply outputs to sklearn_split to tokenise and split into train/test split.

    Parameters
    ----------

    csvfile : str

        Path to CSV file that will be loaded

    x_data : str, default="Sequence"

        String key used to extract relevant x_data column from pandas dataframe of
        imported csv file

    y_data : str, default="Fitness"

        String key used to extract relevant y_data column from pandas dataframe  of
        imported csv file

    returns np.array (Nx2), where N is the number of rows in the csv file

        Returns an Nx2 array with the first column being x_data (sequences), and the second being
        y_data (fitnesses)
    """

    data      = pd.read_csv(csvfile)
    sequences = data[x_data].to_numpy()
    fitnesses = data[y_data].to_numpy()
    fitnesses = fitnesses.reshape(fitnesses.shape[0],1)
    sequences = sequences.reshape(sequences.shape[0],1)

    return np.concatenate((sequences, fitnesses), axis=1)

def collapse_concat(arrays,dim=0):
    """
    Takes an iterable of arrays and recursively concatenates them. Functions similarly
    to the reduce operation from python's functools library.

    Parameters
    ----------
    arrays : iterable(np.array)

        Arrays contains an iterable of np.arrays

    dim : int, default=0

        The dimension on which to concatenate the arrays.

    returns : np.array

        Returns a single np array representing the concatenation of all arrays
        provided.
    """
    if len(arrays) == 1:
        return arrays[0]
    else:
        return np.concatenate((arrays[0],collapse_concat(arrays[1:])))

def sklearn_tokenize(seqs, AAs='ACDEFGHIKLMNPQRSTVWY'):
    """
    Takes an iterable of sequences provided as one amino acid strings and returns
    an array of their tokenized form.

    TODO: Vectorize the operation

    Parameters
    ----------
    seqs : iterable of strings

        Iterable containing all strings

    AAs  : str, default="ACDEFGHIKLMNPQRSTVWY"

        The alphabet of permitted characters to tokenize. Single amino acid codes
        for all 20 naturally occurring amino acids is provided as default.

    returns : np.array(tokenized_seqs)

        Returns a tokenized NP array for the sequences.
    """
    tokens = {x:y for x,y in zip(AAs, list(range(1, len(AAs)+1)))}
    return np.array([[tokens[aa] for aa in seq] for seq in seqs])


def sklearn_split(data, split=0.8):
    """
    Takes a dataset array of two layers, sequences as the [:,0] dimension and fitnesses
    as the [:,1] dimension, shuffles, and returns the tokenized sequences arrays
    and retyped fitness arraysself.

    Parameters
    ----------
    data : np.array (N x 2)

        The sequence and fitness data with sequences provided as single amino acid strings

    split : float, default=0.8, range (0-1)

        The split point for the training - validation data

    returns : x_train, y_train, x_test, y_test

        All Nx1 arrays with train as the first 80% of the shuffled data and test
        as the latter 20% of the shuffled data.
    """

    assert (0 < split < 1), "Split must be between 0 and 1"

    np.random.shuffle(data)

    split_point = int(len(data)*split)

    train = data[:split_point]
    test  = data[split_point:]

    x_train = train[:,0]
    y_train = train[:,1]
    x_test  = test[:,0]
    y_test  = test[:,1]

    return sklearn_tokenize(x_train).astype("int"), y_train.astype("float"), \
           sklearn_tokenize(x_test).astype("int"), y_test.astype("float")


#----------------------------------------------------------------------------------------------------------------
@staticmethod
def position_interactions(seq_len, order): 
    """Returns pairwise diagonal epistasis matrix."""
    f = list(itertools.combinations(range(seq_len), order)) #excludes the diagonal
    return f

@staticmethod
def predict_mem_usage(seq_len, orders, alphabet='EDRKHQNSTPGCAVILMFYW'): 
    """assumes dtype int8, one-hot encoding"""
    total_val = 0
    for i in range(orders): 
        ordern     = i+1         
        num_values = len(position_interactions(seq_len, ordern))*(ordern**len(alphabet))
        total_val+=num_values
    return total_val

@staticmethod
def get_encoding(order, alphabet='EDRKHQNSTPGCAVILMFYW'): 
    """Gets encoding of the n-th order of epistasis"""
    seq_space = [''.join(i) for i in list(itertools.product(alphabet, repeat=order))]
    encoding  = {x:y for x,y in zip(seq_space, range(len(seq_space)))}
    return encoding

@staticmethod
def vector_encoding(sequence, order, encoding, positions): 
    """Create one-hot vector representation of a sequence's interaction at positions, according to encoding"""
    encoding_len = len(list(encoding.keys())[0])
    assert encoding_len==len(positions), f'Encoding length {encoding_len} not equal to number of positions {len(positions)}'
    
    position_aas = []    
    
    for i in positions: 
        position_aas.append(sequence[i])
    
    search_key   = ''.join(position_aas)   
    print(search_key)
    
    vector = [0 for _ in range(len(encoding))]
    vector[encoding[search_key]]=1
    
    return np.array(vector, dtype="int8")
    
@staticmethod
def encode_epistasis(sequences, seq_len, orders, available_memory=5, alphabet='EDRKHQNSTPGCAVILMFYW'):
    """Encodes one-hot matrix for epistasis of n-th order"""
    mem_predict = predict_mem_usage(seq_len=seq_len, orders=orders, alphabet=alphabet)/10**8
    print(f'The encoding matrix will use ~ {mem_predict} GiB of memory')
    
    assert available_memory>mem_predict, f'Required memory {mem_predict} GiB exceeds available memory {available_memory} GiB.'
    
    
    
    encodings      = {x+1:0 for x in range(orders)}    
    encoded_arrays = {x+1:0 for x in range(orders)}
    ord_interact   = {x+1:0 for x in range(orders)}
    
    encoded_arrays[1] = one_hot(sequences, AAs=alphabet)
    
    if orders>1:
        
        for i in range(orders-1):  
            
            order               = i+2 #to escpape python 0 indexing and get into 1 indexing
            encodings[order]    = get_encoding(order, alphabet=alphabet) #get the encoding 
            ord_interact[order] = position_interactions(seq_len=seq_len, order=order ) #we get the possible interactions for this order
            print(encodings[order])
            all_encodings = [] #init list for collecting encoded vectors as they are made
            
            for seq in sequences:
                at_position = []
                for inter in ord_interact[order]: 
                    vec_encode = vector_encoding(sequence=seq, order=order, encoding=encodings[order], positions=inter)
                    at_position.append(vec_encode)
                all_encodings.append(np.array(at_position, dtype='int8'))
            encoded_arrays[order]=np.array(all_encodings, dtype='int8')
            
    return encoded_arrays, encodings, ord_interact 
    
@staticmethod
def flatten_concat_encoded_arrays(encoded_arrays):
    collection = [i.reshape(i.shape[0], i.shape[1]*i.shape[2]) for i in encoded_arrays.values()]
    concat     = np.concatenate(collection, axis=1) 
    return concat

@staticmethod
def interpret_epistasis(model, seq_len, encodings, ord_interact, alphabet='EDRKHQNSTPGCAVILMFYW'): 

    ord_coeffs = {x:0 for x in encodings.keys()}

    all_coeffs     = model.coef_.reshape(-1)

    orders = list(encodings.keys())

    first_order_coeffs = all_coeffs[0:seq_len*len(alphabet)].reshape(seq_len,len(alphabet))
    ord_coeffs[1]=first_order_coeffs

    prev=seq_len*len(alphabet)
    for order in orders[1:]: 
        encoding_size    = len(encodings[order])
        num_interacts    = len(ord_interact[order])
        
        num_coefficients = encoding_size*num_interacts
        
        rt_encoding = int(encoding_size**(1/order))
        r = tuple(rt_encoding for _ in range(order))
        new_shape   = tuple((num_interacts,)) + r
        print(new_shape)
        order_coefficients  = all_coeffs[prev:prev+num_coefficients].reshape(new_shape)
        
        ord_coeffs[order] = order_coefficients
        
        prev+=num_coefficients
        
    return ord_coeffs  









#---------------------------------------------------------

#WT-based encoding

def generate_WT_encodings(WT, AAs='ACDEFGHIKLMNPQRSTVWY'): 
    """For each amino acid position in WT, generate a one-hot dummy encoding system that encodes the other 19 
    amino acids possible at that position"""
    code = {}
    for index, residue in enumerate(WT): 
        aa_sets = {x:y for x,y in zip([i for i in AAs if i!=residue], range(len(AAs)-1))}
        code[index]=aa_sets
    return code


def mutation_dummy_encoding(WT, sequences,code, AAs='ACDEFGHIKLMNPQRSTVWY'):
    """Returns dummy-encoded sequences."""
    
    
    encodings = []
    for sequence in sequences: 
        sequence_encoding = []
        for index, residue in enumerate(sequence):
            arr    = np.zeros(len(AAs)-1)
            if residue!=WT[index]: 
                number = code[index][residue]
                arr[number]=1
                sequence_encoding.append(arr)
            else: 
                sequence_encoding.append(arr)
        encodings.append(np.array(sequence_encoding))
    return np.array(encodings)

