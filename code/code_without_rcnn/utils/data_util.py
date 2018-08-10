import numpy as np
import pandas as pd
from collections import Counter
from nltk.tokenize import word_tokenize
import pickle
import os
import nltk

max_len = 20
word_threshold = 2
counter = None


def preprocess_captions(filenames, captions):
    global max_len
    print "Preprocessing Captions"
    df = pd.DataFrame()
    df['FileNames'] = filenames
    df['caption'] = captions
    df.caption = df.caption.str.decode('utf')          # converting to hexadecimal
    # making tokens of words , that is , keepinf frequencies of each word
    df['caption'] = df['caption'].apply(word_tokenize).apply(lambda x: x[:max_len]).apply(" ".join).str.lower() 
    #df = df[:158900] #uncomment if flickr
    return df


def generate_vocab(df):
    global max_len, word_threshold, counter
    print "Generating Vocabulary"

    vocab = dict([w for w in counter.items() if w[1] >= word_threshold])   # vocab contains all words in a dictionary, with each words associated to its frequency
    vocab["<UNK>"] = len(counter) - len(vocab)
    vocab["<PAD>"] = df.caption.str.count("<PAD>").sum()
    vocab["<S>"] = df.caption.str.count("<S>").sum()
    vocab["</S>"] = df.caption.str.count("</S>").sum()
    wtoidx = {}
    wtoidx["<S>"] = 1
    wtoidx["</S>"] = 2
    wtoidx["<PAD>"] = 0
    wtoidx["<UNK>"] = 3                         # wtoidx stores self genersted indices for all words
    print "Generating Word to Index and Index to Word"
    i = 4
    for word in vocab.keys():
        if word not in ["<S>", "</S>", "<PAD>", "<UNK>"]:
            wtoidx[word] = i
            i += 1
    print "Size of Vocabulary", len(vocab)
    return vocab, wtoidx


def pad_captions(df):
    global max_len
    print "Padding Caption <PAD> to Max Length", max_len, "+ 2 for <S> and </S>"
    dfPadded = df.copy()
    dfPadded['caption'] = "<S> " + dfPadded['caption'] + " </S>"
    max_len = max_len + 2
    for i, row in dfPadded.iterrows():
        cap = row['caption']
        cap_len = len(cap.split())
        if(cap_len < max_len):
            pad_len = max_len - cap_len
            pad_buf = "<PAD> " * pad_len
            pad_buf = pad_buf.strip()    # remove whitespaces from both ends
            dfPadded.set_value(i, 'caption', cap + " " + pad_buf)
    return dfPadded


def load_features(feature_path, filenames):

    print("Loading features")
    
    features = np.load(feature_path)
    
    
    ####################### Change this ############################
    ### Not necessarily 5 captions anymore ####
    
    print(features.shape)
    fin_features = np.zeros((len(filenames), features.shape[1]));
    
    set_filename = set(filenames)
    print(len(set_filename))
    
    print(fin_features.shape)
    
    fin_features[0] = features[0]
    j=0
    for i in range(1, fin_features.shape[0]):
        
        if (filenames[i]!=filenames[i-1]):
            #print(filenames[i])
            j += 1
            #print(j)
        fin_features[i] = features[j]   
    
    if (i == fin_features.shape[0]):
        print ("All features loaded properly")
    
    print "Features Loaded", feature_path
    return fin_features



def get_data(required_files):
    ret = []
    for fil in required_files:
        ret.append(np.load("Data/" + fil + ".npy"))
    return ret


def generate_captions(
        wt=2,
        ml=20,
        cap_path='Data/training.txt',
        feat_path='Data/training_features.npy'):
    required_files = ["vocab", "wordmap", "Training_Data"]
    generate = False
    for fil in required_files:
        if not os.path.isfile('Data/' + fil + ".npy"):
            generate = True
            print "Required Files not present. Regenerating Data."
            break
    if not generate:
        print "Dataset Present; Skipping Generation."
        return get_data(required_files)
    global max_len, word_threshold, counter
    max_len = ml
    word_threshold = wt
    print "Loading Caption Data", cap_path
    with open(cap_path, 'r') as f:
        data = f.readlines()
    filenames = [caps.split('\t')[0].split('#')[0] for caps in data]
    captions = [caps.replace('\n', '').split('\t')[1] for caps in data]
    df = preprocess_captions(filenames, captions)
    features = load_features(feat_path, filenames)
    
    idx = np.random.permutation(features.shape[0])
    df = df.iloc[idx]
    features = features[idx]
    # df, features = split_dataset(df, features) #use flickr8k for
    # validationSSS
    counter = Counter()
    for i, row in df.iterrows():
        counter.update(row["caption"].lower().split())   # counting occurence of each word in the captions, split() splits on space
    df = pad_captions(df)
    vocab, wtoidx = generate_vocab(df)
    captions = np.array(df.caption)
    np.save("Data/Training_Data", zip(features, captions))
    np.save("Data/wordmap", wtoidx)
    np.save("Data/vocab", vocab)

    print "Preprocessing Complete"
    return get_data(required_files)
