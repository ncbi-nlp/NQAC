import numpy as np
import sys
from collections import Counter
import utils
import operator

class BeamSearch():
    def __init__(self, model, beam, maxlen, data, features):
        self.m = model
        self.beam = beam
        self.maxlen = maxlen
        self.char_indices = data["ci"]
        self.indices_char = data["ic"]
        self.INC = data["INC"]
        self.UNK = data["UNK"]
        self.U = data["U"]
        self.w2v = data["w2v"]
        self.u2v = data["u2v"]
        self.qvocab = data["qvocab"]
        self.features = features

    def search(self, sequence, user=None, timestamp=None, diverse=False):
        suggestions = {sequence:0}

        unfinished = True
        while unfinished:
            unfinished = False
            new_suggestions = {}
            for seq,prob in suggestions.items():
                if seq[-1] == "\n" or len(seq) == self.maxlen:
                    new_suggestions[seq] = prob
                else:
                    unfinished = True
                    predictions = self.predictCharacters(seq, user, timestamp, diverse)
                    # Add diversity, ref: Vijayakumar et al., 2016
                    if diverse:
                        initlen = len(sequence)
                        # key-val tuples sorted by value
                        sorted_predictions = sorted(predictions.items(), key=operator.itemgetter(1), reverse=True)
                        # First group is argmax of beam search
                        new_suggestions[seq + sorted_predictions[0][0]] = suggestions[seq] + np.log(sorted_predictions[0][1])
                        avgWeight = 0
                        for g in range(1, len(sorted_predictions)):
                            # For each remaining, we minimize the similarity with previous suggestions
                            # To this end, we use a normalized levenshtein distance
                            current_prob = suggestions[seq] + np.log(sorted_predictions[g][1])
                            newseq = seq + sorted_predictions[g][0]
                            avgDistance = 0
                            for previous in new_suggestions.keys():
                                avgDistance += utils.levenshtein_distance(newseq[initlen:], previous[initlen:])/max(len(newseq[initlen:]), len(previous[initlen:]))
                            avgDistance /= len(new_suggestions.keys())
                            to_add = 0.26*np.log(avgDistance)
                            avgWeight += to_add
                            new_suggestions[newseq] = current_prob + to_add
                        # Rebalance
                        new_suggestions[seq + sorted_predictions[0][0]] += avgWeight/(self.beam-1)
                    else:
                        for c,p in predictions.items():
                            new_suggestions[seq + str(c)] = suggestions[seq] + np.log(p)
            new_suggestions = dict(Counter(new_suggestions).most_common(self.beam))
            suggestions = new_suggestions
        return suggestions

    def predictCharacters(self, sequence, user=None, timestamp=None, diverse=False):
        len_chars = len(self.char_indices)

        # Features
        w2vs = 0
        if "w2v" in self.features:
            w2vs = self.features["w2v"]
        u2vs = 0
        if "u2v" in self.features:
            u2vs = self.features["u2v"]
        ts = 0
        if "timestamp" in self.features:
            ts = self.features["timestamp"]

        # Init input
        x = np.zeros((1, self.maxlen, sum(self.features.values())))
        last_word = ""
        last_word_i = 0
        for t, char in enumerate(sequence):
            if u2vs > 0:
                try:
                    x[0, t, len_chars+w2vs:len_chars+w2vs+u2vs] = self.u2v[user]
                except: pass
                    # x[i, t, len_chars+w2vs:len_chars+w2vs+u2vs] = self.U
            if ts > 0:
                try:
                    cos_s, sin_s, cos_d, sin_d = utils.timestamp_to_features(timestamp)
                    time_features = [cos_s, sin_s, cos_d, sin_d]
                    x[0, t, len_chars+w2vs+u2vs:len_chars+w2vs+u2vs+ts] = time_features
                except: pass
            x[0, t, self.char_indices[char]] = 1
            if char == " ":
                last_word = sequence[last_word_i:t]
                last_word_i = t+1
                try:
                    embed = self.UNK
                    if last_word in self.qvocab and self.qvocab[last_word] >= 5:
                        embed = self.w2v[last_word]
                    x[0, t, len_chars:len_chars+w2vs] = embed
                except:
                    x[0, t, len_chars:len_chars+w2vs] = self.UNK
            else:
                x[0, t, len_chars:len_chars+w2vs] = self.INC
        chars = {}
        predictions = self.m.predict(x)[0]
        preds = predictions[len(sequence)-1]
        indices = np.argpartition(preds, -self.beam)[-self.beam:]
        for i in indices:
            chars[self.indices_char[i]] = preds[i]
        return chars
