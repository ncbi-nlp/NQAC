from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dropout, TimeDistributed, Dense
from keras.optimizers import Adam
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import os
import gensim
import pickle
from beamsearch import BeamSearch
from keras.models import load_model
import dateutil.parser
import utils
import json

config = json.load(open("config_learn.json"))
os.environ['CUDA_VISIBLE_DEVICES'] = config["gpuID"]

#### PARAMS
studyID = config["studyID"]
maxlen = config["maxlen"]
update = config["update"]
lr = config["lr"]
batch_size = config["batch_size"]
epochs = config["epochs"]

# Some various setups
use_w2v = config["use_w2v"]
w2v_size = config["w2v_size"]
use_u2v = config["use_u2v"]
u2v_size = config["u2v_size"]
use_timestamps = config["use_timestamps"]
timestamp_size = config["timestamp_size"]
############

path = "queries/train.txt"
U_file = "pkl/u.pkl"
UNK_file = "pkl/unk.pkl"
INC_file = "pkl/inc.pkl"
ci = "pkl/char_indices"
ic = "pkl/indices_char"
run_id = "run_"+studyID
if not os.path.exists("models/"+run_id):
    os.makedirs("models/"+run_id)

print("Loading embeddings")
if use_w2v:
    w2v = gensim.models.KeyedVectors.load_word2vec_format('vectors/GoogleNews-vectors-negative300.bin', binary=True)
else: w2v = {}
if use_u2v:
    u2v = gensim.models.Doc2Vec.load('vectors/user2vec_d30_2.model')
else: u2v = {}
print("Done.")

def createVocabulary(string):
    # Create query vocabulary to filter out words that appear less than 5 tiems in logs
    qvocab = {}
    words = string.split()
    for word in words:
        if word not in qvocab:
            qvocab[word] = 0
        qvocab[word] += 1
    return qvocab

def string_to_sequences(string, qvocab, seq_maxlen, char_idx, features):
    len_chars = len(char_idx)
    w2vs = 0
    if "w2v" in features:
        w2vs = features["w2v"]
    u2vs = 0
    if "u2v" in features:
        u2vs = features["u2v"]
    ts = 0
    if "timestamp" in features:
        ts = features["timestamp"]

    sequences = []
    next_chars = []
    print("    sequencing")
    ses = string.split("\n")
    users = []
    sequences = []
    timestamps = []
    for s in ses:
        sp = s.split("\t")
        if len(sp) == 3:
            sequences.append(sp[1])
            if u2vs > 0:
                users.append(sp[0])
            if ts > 0:
                unformatted_date = sp[2]
                date_pieces = unformatted_date.split(" ")
                formatted_date = date_pieces[0]+"-"+date_pieces[1]+"-"+date_pieces[2]+" "+date_pieces[3]+":"+date_pieces[4]+":"+date_pieces[5]
                weekday = dateutil.parser.parse(formatted_date).weekday()
                cos_s, sin_s = utils.time_to_real(int(date_pieces[3]), int(date_pieces[4]), int(date_pieces[5]))
                cos_d, sin_d = utils.weekday_to_real(weekday)
                timestamps.append([cos_s, sin_s, cos_d, sin_d])

    print("    io init")
    X = np.zeros((len(sequences), seq_maxlen, sum(features.values())), dtype=np.float32)
    Y = np.zeros((len(sequences), seq_maxlen, len_chars), dtype=np.float32)
    for i, seq in enumerate(sequences):
        last_word = ""
        last_word_t = 0
        for t, char in enumerate(seq):
            X[i, t, char_idx[char]] = 1
            if u2vs > 0:
                try:
                    X[i, t, len_chars+w2vs:len_chars+w2vs+u2vs] = u2v[users[i]]
                except: pass
                    # X[i, t, len_chars+w2vs:len_chars+w2vs+u2vs] = U
            if ts > 0:
                X[i, t, len_chars+w2vs+u2vs:len_chars+w2vs+u2vs+ts] = timestamps[i]
            if char == " ":
                last_word = seq[last_word_t:t]
                last_word_t = t+1
                try:
                    embed = UNK
                    if qvocab[last_word] >= 5:
                        embed = w2v[last_word]
                    X[i, t, len_chars:len_chars+w2vs] = embed
                except:
                    X[i, t, len_chars:len_chars+w2vs] = UNK
            else:
                X[i, t, len_chars:len_chars+w2vs] = INC
            try:
                Y[i, t, char_idx[seq[t+1]]] = 1
            except:
                Y[i, t, char_idx["\n"]] = 1
    return X, Y, qvocab

queries = ""
with open(path) as f:
    queries = f.read()#[:10000000]

# Calculate static vectors
if not update:
    print('Loading previous char_indices')
    char_indices = pickle.load(open(ci,'rb'))
    indices_char = pickle.load(open(ic,'rb'))
    print('Loading UNK and INC and U')
    U = pickle.load(open(U_file, 'rb'))
    UNK = pickle.load(open(UNK_file, 'rb'))
    INC = pickle.load(open(INC_file, 'rb'))
else:
    print("Calculating char_indices")
    chars = sorted(list(set(queries)))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))
    print('Generating UNK and INC and U')
    UNK = np.random.uniform(-0.25, 0.25, 300)
    INC = np.random.uniform(-0.25, 0.25, 300)
    U = np.random.uniform(-0.25, 0.25, 30)
    pickle.dump(U, open(U_file,'wb'))
    pickle.dump(UNK, open(UNK_file,'wb'))
    pickle.dump(INC, open(INC_file,'wb'))
    pickle.dump(char_indices, open(ci,'wb'))
    pickle.dump(indices_char, open(ic,'wb'))
print("Done.")

print('Done.')

features = {"chars": len(char_indices)}
if use_w2v:
    features["w2v"] = w2v_size
if use_u2v:
    features["u2v"] = u2v_size
if use_timestamps:
    features["timestamp"] = timestamp_size

# Net
print("Building network")
model = Sequential()
model.add(GRU(1024, return_sequences=True, input_shape=(maxlen, sum(features.values())), activation="relu"))
model.add(Dropout(0.5))
model.add(GRU(1024, return_sequences=True, input_shape=(maxlen, sum(features.values())), activation="relu"))
model.add(Dropout(0.5))
model.add(TimeDistributed(Dense(len(char_indices), activation='softmax')))
optimizer = Adam(lr=lr, clipnorm=0.5)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
print(model.summary())
print(run_id)
print("Done.")

iterations = 30 #
split_size = int(len(queries.split("\n"))/iterations)
print("Init vacabulary")
qvocab = createVocabulary(queries)
print("Done.")

data = {
        "ci": char_indices,
        "ic": indices_char,
        "INC": INC,
        "UNK": UNK,
        "U": U,
        "w2v": w2v,
        "u2v": u2v,
        "qvocab": qvocab
       }

def on_epoch_end(epoch, logs):
    print()
    bs = BeamSearch(model, 10, maxlen, data, features)
    timestamp = None
    user = None
    if use_timestamps:
        timestamp = 1148693365.0
    if use_u2v:
        user = "9032971"
    suggestions = bs.search(["www "], user, timestamp)
    for s,p in suggestions[0].items():
        print(s.strip()+":"+str(p))
    print()
print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

for i in range(epochs):
    for j in range(iterations):
        print("Preparing data for next sample - sample "+str(j)+" of epoch "+str(i))
        start = j*split_size
        end = min((j+1)*split_size, len(queries))
        qsplit = "\n".join(queries.split("\n")[start:end])
        X, Y, _ = string_to_sequences(qsplit, qvocab, maxlen, char_indices, features)
        print("Done.")
        model.fit(X, Y, validation_split=0.2, batch_size=batch_size, epochs=1, callbacks=[print_callback])
        model.save("models/"+run_id+"/epoch"+str(i)+"_sample"+str(j)+".h5")
    model.save("models/"+run_id+"/epoch"+str(i)+".h5")
model.save("models/"+run_id+"/final.h5")
