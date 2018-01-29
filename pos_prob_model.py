# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 19:29:04 2018

@author: ethan
"""

from __future__ import division
import pandas as pd
from collections import defaultdict, Counter
from numpy.random import choice

posdata = pd.read_csv('/home/ethan/Downloads/ner0.csv')
posdata = posdata[['word', 'pos', 'prev-prev-pos', 
                   'prev-pos', 'next-pos', 
                   'next-next-pos']]
                   
# posdata['suffix'] = [w[-3:] for w in posdata.word]
                   
posdata = [tuple(x) for x in posdata.values]
                  
def normalize(dic):
    for c in dic.values():
        total = float(sum(c.values()))
        for key in c:
            c[key] /= total
    return dic
        
def preprocess(word):
    if '-' in word and word[0] != '-':
            return '<HYPHEN>'
    elif word.isdigit() and len(word) == 4:
        return '<YEAR>'
    elif word[0].isdigit():
        return '<DIGITS>'
    else:
        return word
        
START = ['__START1__']
END = ['__END1__']

classes = set()                   
context = defaultdict(Counter)
suffixes = defaultdict(Counter)
suff_emission = defaultdict(Counter)
words = defaultdict(Counter)
emission = defaultdict(Counter)

for row in posdata:
    cont = row[3:5]
    tag = row[1]
    word = preprocess(row[0])
    suffix = word[-3:]
    
    context[cont][tag] += 1
    suff_emission[tag][suffix] += 1
    suffixes[suffix][tag] += 1
    emission[tag][word] += 1
    words[word][tag] += 1
    classes.add(tag)
    
context = normalize(context)
emission = normalize(emission)
suffixes = normalize(suffixes)
suff_emmission = normalize(suff_emission)

freq_thresh = 5
ambiguity_thresh = 0.98

tagdict = {}
for word, tag_freqs in words.items():
    tag, mode = tag_freqs.most_common(1)[0]
    n = sum(tag_freqs.values())
    if n >= freq_thresh and (mode/n) >= ambiguity_thresh:
        tagdict[word] = tag
        
words = normalize(words)

def tag(sentence, maxiter = 150):
    def best_guess(word):
        guess = tagdict.get(word)
        if not guess:
            if words.get(word) is not None:
                guess, _ = words.get(word).most_common(1)[0]
            elif suffixes.get(word[-3:]) is not None:
                guess, _ = suffixes.get(word[-3:]).most_common(1)[0]
            else:
                guess = choice(['NN', 'NNP', 'JJ', 'VBZ'])
        return guess
    
    sent = [preprocess(w) for w in sentence]
    n = len(sent)
    init = []
    idx = []
    for i, w in enumerate(sent):
        init.append(best_guess(w))
        if not tagdict.get(w):
            idx.append(i+1)
    state = START + init + END
    samples = set()
    samples.add(tuple(state))
    for k in xrange(maxiter):
        for i in idx:
            cont = (state[i-1], state[i+1])
            tags = context.get(cont)
            if not tags:
                tags = suffixes[sent[i-1][-3:]]
            draw = choice(tags.keys(), p=tags.values())
            state[i] = draw
        samples.add(tuple(state))
    best_prob = None
    best = None
    for s in samples:
        probs = []
        for i in xrange(n):
            j = i+1
            word = sent[i]
            tag = s[j]
            p = emission[tag][word]
            if not p: 
                emm = suff_emission[tag][word[-3:]]
                p = emm if emm > 0 else 0.00000001
            cont = (s[j-1], s[j+1])
            trans = context[cont][tag]
            q = trans if trans > 0 else 0.00000001
            probs.extend([p, q])
        prob = reduce(lambda x, y: x*y, probs)
        if prob > best_prob:
            best_prob = prob
            best = s
    return zip(sentence, best[1:(n+1)])
            
    
    
                  

                   



