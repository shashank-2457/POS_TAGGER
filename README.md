Implemented a bigram HMM POS tagger. During training, I have to calculate two kinds of probabilities (store them in two large dictionaries):
• P(ti|ti−1) (similar to the bigram LM you have already implemented, use Laplace smoothing).
These are the prior probabilities.
• P(wi|ti), the probability of observing word wi given POS tag ti
. You don’t need to smooth this
probability distribution. These are the likelihood probabilities.
Using  Viterbi decoding during testing (you have to update token.tag with the right tag).

Your can run the starter code with the following command:

$ python3 code/pos_tagger.py data/train data/heldout --mode always_NN
** Testing the model with the training instances (boring, this is just a sanity check)
[always_NN ] Accuracy [35000 sentences]: 12.90 [not that useful, mostly a sanity check]
** Testing the model with the test instances (interesting, these are the numbres that matter)
[always_NN:11] Accuracy [13928 sentences]: 12.84

$ python3 code/pos_tagger.py data/train data/heldout --mode majority
** Testing the model with the training instances (boring, this is just a sanity check)
[majority ] Accuracy [35000 sentences]: 12.90 [not that useful, mostly a sanity check]
** Testing the model with the test instances (interesting, these are the numbres that matter)
[majority:11] Accuracy [13928 sentences]: 12.84

$ python3 code/pos_tagger.py data/train data/heldout --mode hmm
** Testing the model with the training instances (boring, this is just a sanity check)
[hmm ] Accuracy [35000 sentences]: 12.90 [not that useful, mostly a sanity check]
** Testing the model with the test instances (interesting, these are the numbres that matter)
[hmm:11] Accuracy [13928 sentences]: 12.84

There are three part-of-speech taggers. The first one always predicts NN and is already implemented.
The second one will be the majority tag per word baseline and you have to implement it. The third
one will be the HMM tagger and you have to implement it.
