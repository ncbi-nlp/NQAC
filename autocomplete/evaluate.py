from keras.models import load_model
import pickle
import gensim
from beamsearch import BeamSearch
import os
from random import randint
import time
import json
import sys

def createExamples(text):
    examples = []
    lines = text.split("\n")

    for line in lines:
        linesp = line.split("\t")
        if len(linesp) == 3:
            u = linesp[0]
            q = linesp[1]
            t = float(linesp[2].strip())
            sp = q.split(" ")
            if len(sp) > 1:
                start = q.find(" ")+1 # +1 bc we incorporate the space character in the first input: the user wants completion
                for j in range(start,len(q)):
                    # Generate all prefixes
                    x = q[:j]
                    examples.append([x,q,u,t])
    return examples

def loadData(generate):
    bg = {}
    qvocab = {}
    users = {}
    print("    BG data")
    with open("./queries/queries_"+study+".txt") as f:
        for line in f:
            sp = line.split("\t")
            query = sp[1]
            user = sp[0]
            words = query.split(" ")
            for word in words:
                if word not in qvocab:
                    qvocab[word] = 0
                qvocab[word] += 1
            if query not in bg:
                bg[query] = 0
            if user not in users:
                users[user] = 0
            bg[query] += 1
    if generate:
        print("    prefix generation")
        examples = createExamples(open("./queries/queries_"+study+"t_sample.txt",'r').read())
        with open("./queries/queries_"+study+"_prefixes.txt", "w") as fw:
            for quad in examples:
                fw.write(quad[0]+"\t"+quad[1]+"\t"+quad[2]+"\t"+str(quad[3])+"\n")
    else:
        print("    prefix loading")
        examples = []
        with open("./queries/queries_"+study+"_prefixes.txt") as f:
            for line in f:
                examples.append(line.strip().split("\t"))
    # with open("results/seen-unseen_users_"+study+".txt", "w") as fw:
    #     with open("results/seen-unseen_queries_"+study+".txt", "w") as f:
    #         for quad in examples:
    #             if quad[1] in bg:
    #                 f.write("1\n")
    #             else:
    #                 f.write("0\n")
    #             if quad[2] in users:
    #                 fw.write("1\n")
    #             else:
    #                 fw.write("0\n")
    return examples, qvocab, bg

def RR(prefix, solution, raw_results, diverse):
    inverse_rp = 0
    inverse_rp_partial = 0
    found = ""
    try:
        # Make a list, strip \n and order according to output probability
        candidates = [k for k in sorted(raw_results, key=raw_results.get, reverse=True)]
    except:
        # Or just take the list if that's the input
        candidates = raw_results
    for i,c in enumerate(candidates):
        stripped = c.strip()
        if solution == stripped:
            found = stripped
            inverse_rp = 1/(i+1)
            if inverse_rp_partial == 0:
                inverse_rp_partial = 1/(i+1)
            break
        if solution.startswith(stripped+" ") and found == "":
            found = stripped
            inverse_rp_partial = 1/(i+1)
    d = ""
    if diverse:
        d = "_d"
    with open("results/"+modelname+"/epoch"+epoch+"_sample"+sample+d+"_scores.txt", "a") as f:
        f.write(prefix+"\t"+str(inverse_rp)+"\t"+str(inverse_rp_partial)+"\n")
    return inverse_rp, inverse_rp_partial

def MRR(prefixes, bs, batchlen, diverse=False):
    total = 0
    total_partial = 0
    iteration = 0
    batch = []
    solutions_recorded = []
    t = 0
    # For each prefix-solution pair
    for ps in prefixes:
        prefix = ps[0]
        solution = ps[1]
        user = None
        if use_u2v:
            user = ps[2]
        timestamp = None
        if use_timestamps:
            timestamp = ps[3]
        # Run the net and beam search
        start_time = time.time()
        raw_results = bs.search(prefix, user, timestamp, diverse)
        t = time.time() - start_time
        d = ""
        if diverse:
            d = "_d"
        with open("results/"+modelname+"/epoch"+epoch+"_sample"+sample+d+".txt", "a") as f:
            for c,s in raw_results.items():
                f.write(c.strip()+"\t"+str(s)+"\t"+prefix+"\t"+solution+"\n")
        addtotal, addpartial = RR(prefix, solution, raw_results, diverse)
        total += addtotal
        total_partial += addpartial
        iteration += 1
        print(str(iteration)+" ~"+str(t)+" seconds per beam search         ", end="\r")
    return total/len(prefixes), total_partial/len(prefixes)

def MPC(prefix, bg):
    # Limit BG queries to the given prefix
    candidates = {}
    for q,v in bg.items():
        if q.startswith(prefix) and q != "prefix":
            candidates[q] = v

    # Convert occurences to probability (estimated by frequency)
    total = len(candidates.keys())
    for q,v in candidates.items():
        candidates[q] = v/total
    return sorted(candidates, key=candidates.get, reverse=True)[:10]

def MPCEval(prefixes, bg):
    total = 0
    total_partial = 0
    iteration = 0
    for ps in prefixes:
        prefix = ps[0]
        solution = ps[1]
        candidates = MPC(prefix, bg)
        addtotal, addpartial = RR(prefix, solution, candidates)
        total += addtotal
        total_partial += addpartial
        iteration += 1
        print(str(iteration), end="\r")
    total /= len(prefixes)
    total_partial /= len(prefixes)
    print("MRR: "+str(total)+", PMRR: "+str(total_partial))

config = json.load(open("config_evaluate.json"))
os.environ['CUDA_VISIBLE_DEVICES'] = config["CUDA_VISIBLE_DEVICES"]

#### PARAMS
print("Loading params")
modelname = config["modelname"]
study = config["study"]
maxlen = config["maxlen"]
epoch = config["epoch"]
sample = config["sample"]

# Some various setups
use_w2v = config["use_w2v"]
w2v_size = config["w2v_size"]
use_u2v = config["use_u2v"]
u2v_size = config["u2v_size"]
use_timestamps = config["use_timestamps"]
timestamp_size = config["timestamp_size"]
print("Done")
############

###### Do not touch unless you want to resample the test set
generatePrefixes = False

print("Loading data")
examples, qvocab, bg = loadData(generatePrefixes)
if modelname == "MPC":
    print(MPCEval(examples, bg))
    sys.exit()
fname = "models/"+modelname+"/epoch"+epoch+"_sample"+sample+".h5"
print("    model "+modelname+" "+epoch+"s"+sample)
model = load_model(fname)
print(model.summary())
print("Done.")

print("Loading embeddings")
U = pickle.load(open("./pkl/u_"+study+".pkl", 'rb'))
UNK = pickle.load(open("./pkl/unk_"+study+".pkl", 'rb'))
INC = pickle.load(open("./pkl/inc_"+study+".pkl", 'rb'))
char_indices = pickle.load(open("./pkl/char_indices_"+study,'rb'))
indices_char = pickle.load(open("./pkl/indices_char_"+study,'rb'))
if use_w2v:
    if study == "aol":
        w2v = gensim.models.KeyedVectors.load_word2vec_format('vectors/GoogleNews-vectors-negative300.bin', binary=True)
    else:
        w2v = gensim.models.KeyedVectors.load_word2vec_format('vectors/PubMed-w2v.bin', binary=True)
else: w2v = {}
if use_u2v:
    if study == "aol":
        u2v = gensim.models.Doc2Vec.load('vectors/user2vec_d30_2.model')
    else:
        u2v = gensim.models.Doc2Vec.load('vectors/user2vec_pubmed2.model')
else: u2v = {}
print("Done.")

print("Initializing beam searcher")
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
features = {"chars": len(char_indices)}
if use_w2v:
    features["w2v"] = w2v_size
if use_u2v:
    features["u2v"] = u2v_size
if use_timestamps:
    features["timestamp"] = timestamp_size
bs = BeamSearch(model, 10, maxlen, data, features)
print("Done")

diverse = False
d = ""
if diverse:
    d = "_d"
suggestions = bs.search("www ", diverse=diverse)
for s,p in suggestions.items():
    print(s.strip()+":"+str(p))
print()
if not os.path.exists("results/"+modelname):
    os.makedirs("results/"+modelname)
try:
    os.remove("results/"+modelname+"/epoch"+epoch+"_sample"+sample+d+"_scores.txt")
except OSError:
    pass
try:
    os.remove("results/"+modelname+"/epoch"+epoch+"_sample"+sample+d+".txt")
except OSError:
    pass
print("Number of tests: "+str(len(examples)))
print()
print(MRR(examples, bs, 1, diverse))
