import argparse
import collections
import math
import operator
import random
import pandas as pd
import numpy as np
import re
import utils

import time

# your code here    
# =============================================================================
# Model with emisison probabilities as likelihoods
# transition probabilities as priors <s> </s> removed
# =============================================================================
def create_model(sentences):
    
    
    prior_counts = collections.defaultdict(lambda: collections.defaultdict(int))
    priors = collections.defaultdict(lambda: collections.defaultdict(float))
    likelihood_counts = collections.defaultdict(lambda: collections.defaultdict(int))
    likelihoods = collections.defaultdict(lambda: collections.defaultdict(float))

    majority_tag_counts = collections.defaultdict(lambda: collections.defaultdict(int))
    majority_baseline = collections.defaultdict(lambda: "NN")
    tag_counts = collections.defaultdict(int)
    
    
    for sentence in sentences:
        for i in range(len(sentence)):
            majority_tag_counts[sentence[i].word][sentence[i].tag]+=1
        for val in range(1,len(sentence)-1):
            likelihood_counts[sentence[val].word][sentence[val].tag]+=1
        for i in range(len(sentence)):
            tag_counts[sentence[i].tag]+=1
            
   
    for x in tag_counts:
        if x!='</s>':
            for y in tag_counts:
                if y!='<s>':
                    prior_counts[x][y]=0
    for sentence in sentences:
        for i in range(len(sentence)-1):
            prior_counts[sentence[i].tag][sentence[i+1].tag]+=1
            
    for tag1 in prior_counts:
        for y in prior_counts[tag1]:
            priors[tag1][y]= (prior_counts[tag1][y]+1)/(tag_counts[tag1]+len(tag_counts)-1)
            
   
    for word in likelihood_counts:
        for tag in tag_counts:
            likelihoods[word][tag] = (likelihood_counts[word][tag]+1)/(tag_counts[tag]+len(sentences))

        
    for dic in majority_tag_counts:
        majority_baseline[dic] = sorted(majority_tag_counts[dic].items(), key=lambda x: x[1], reverse=True)[0][0]

   

    return priors, likelihoods, majority_baseline, tag_counts

# =============================================================================
# function to calculate unseen words
# =============================================================================
def morphology(tokens,tags,tag_counts,V):
    pos_tags = []
    token = tokens.word
    if re.fullmatch(r'([Tt]h[i|s|a|e][s|t]+|a|an|all)', token):  pos_tags.append("DT")
    elif re.fullmatch(r'^[0-9]+.[0-9]*$',token): pos_tags.append("CD")
    elif re.fullmatch(r'([a-zA-Z][a-z]|with)',token): pos_tags.append("IN")
    elif re.search(r'\'',token) :pos_tags.append("RB")
    elif re.search(r'ed$',token): pos_tags.append("VBD")
    elif re.search(r'[A-Z][a-zA-Z]*',token):pos_tags.append("PRP")
    elif re.search(r's$',token):pos_tags.append("NNS")
    else:
        pos_tags.append("NN")
    
    return 1/(tag_counts[pos_tags[0]]+V)

# =============================================================================
# Uncomment the line 190 to test this
# =============================================================================
def capital(sentence):
   
    words = ""
    inputt = ""
    for i in range(1,len(sentence)-1):
       inputt+=sentence[i].word+" "
       if i==1 or i==len(sentence)-1:
           word = sentence[i].word
           word = word[0].upper()+word[1:]
           words+=word+" "
       else:
           if re.fullmatch(r'[N|V|P|A|J]+',sentence[i].tag): 
               word = sentence[i].word
               word = word[0].upper()+word[1:]
               words+=word+" "
           elif len(sentence[i].word)>3:
               word = sentence[i].word
               word = word[0].upper()+word[1:]
               words+=word+" "
           else:
                words+= sentence[i].word +" "
    print(inputt[:-1])
    print(words[:-1])
               
# =============================================================================
# For Hmm to calculate the tags using viterbi I used the the forward filling and backward tags tracing        
# =============================================================================
         
def predict_tags(sentences, model, mode='always_NN'):
    priors, likelihoods, majority_baseline, tag_counts = model
    ap = 0
    for sentence in sentences:
        
        if mode == 'always_NN':
            # Do NOT change this one... it is a baseline
            for token in sentence:
                token.tag = "NN"
        elif mode == 'majority':
            # Do NOT change this one... it is a (smarter) baseline
            for token in sentence:
                token.tag = majority_baseline[token.word]
        elif mode == 'hmm':
            tags = list(tag_counts.keys()) 
            tags.remove('<s>')
            tags.remove('</s>')
            forward_matrix = np.zeros((len(tags), len(sentence)-2))
            backward_matrix=np.zeros((len(tags), len(sentence)-2),dtype=int)
                      
            for tag_id in range(len(tags)):
                if priors['<s>'][tags[tag_id]] == 0:
                    forward_matrix[tag_id][0] = float("-inf")
                else:
                    if likelihoods[sentence[1].word][tags[tag_id]] == 0.0:
                        likelihoods[sentence[1].word][tags[tag_id]]=morphology(sentence[1],tags,tag_counts,len(sentences))
                    forward_matrix[tag_id][0]=math.log(priors['<s>'][tags[tag_id]]) +math.log(likelihoods[sentence[1].word][tags[tag_id]]) 
# =============================================================================
#             Forward
# =============================================================================
            for word_idx in range(2,len(sentence)-1):              
                for tag_id in range(len(tags)):
                    max_prob = float('-inf')
                    previous_tag_max  = None
                    for itera in range(len(tags)):
                        if likelihoods[sentence[word_idx].word][tags[tag_id]] == 0.0:
                            likelihoods[sentence[word_idx].word][tags[tag_id]]=morphology(sentence[word_idx],tags,tag_counts,len(sentences))
                        value =forward_matrix[itera][word_idx-2]+math.log(priors[tags[itera]][tags[tag_id]])+math.log(likelihoods[sentence[word_idx].word][tags[tag_id]])
                      
                        if(max_prob<value) :
                            max_prob = value
                            previous_tag_max = itera
                            
                    
                    forward_matrix[tag_id][word_idx-1]=max_prob
                    backward_matrix[tag_id][word_idx-1]=previous_tag_max
# =============================================================================
#             Backward tracing
# =============================================================================
            end = len(sentence)-3
            tag_idx = None
            output = ['</s>']
            max_prob = float("-inf")
            for i in range(len(tags)):
                if(max_prob<forward_matrix[i][end]):
                    max_prob = forward_matrix[i][end]
                    tag_idx = i
                   
            output.append(tags[tag_idx])
            for idx in range(len(sentence)-3,0,-1):
                tag_idx=backward_matrix[tag_idx][idx]
                output.append(tags[tag_idx])
                
            output.append('<s>')
            output =  output[::-1]
        
            i=0
            for token in sentence:
                token.tag = output[i]
                i=i+1;
        
        else:
            assert False
        print(ap)
        ap=ap+1
        #capital(sentence)
        
    return sentences


if __name__ == "__main__":
    # Do NOT change this code (the main method)
    start = time.process_time()
    parser = argparse.ArgumentParser()
    parser.add_argument("PATH_TR",
                        help="Path to train file with POS annotations")
    parser.add_argument("PATH_TE",
                        help="Path to test file (POS tags only used for evaluation)")
    parser.add_argument("--mode", choices=['always_NN', 'majority', 'hmm'], default='always_NN')
    args = parser.parse_args()

    tr_sents = utils.read_tokens(args.PATH_TR) #, max_sents=1)
    # test=True ensures that you do not have access to the gold tags (and inadvertently use them)
    te_sents = utils.read_tokens(args.PATH_TE, test=False)
    

    model = create_model(tr_sents)
    print("model creatinion", time.process_time() - start)
    print("** Testing the model with the training instances (boring, this is just a sanity check)")
    gold_sents = utils.read_tokens(args.PATH_TR)
    predictions = predict_tags(utils.read_tokens(args.PATH_TR, test=True), model, mode=args.mode)
    accuracy = utils.calc_accuracy(gold_sents, predictions)
    print(f"[{args.mode:11}] Accuracy "
          f"[{len(list(gold_sents))} sentences]: {accuracy:6.2f} [not that useful, mostly a sanity check]")
    print()
    print("verify creatinion", time.process_time() - start)
    print("** Testing the model with the test instances (interesting, these are the numbres that matter)")
    # read sentences again because predict_tags(...) rewrites the tags
    gold_sents = utils.read_tokens(args.PATH_TE)
    predictions= predict_tags(te_sents, model, mode=args.mode)
    accuracy = utils.calc_accuracy(gold_sents, predictions)
    print(f"[{args.mode}:11] Accuracy "
          f"[{len(list(gold_sents))} sentences]: {accuracy:6.2f}")
    print((time.process_time() - start))
