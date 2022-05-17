import argparse
import collections

class Token:
    def __init__(self, word, tag):
        self.word = word
        self.tag = tag

    def __str__(self):
        return f"{self.word}/{self.tag}"


def read_tokens(file, max_sents=-1, test=False):
    f = open(file)
    sentences = []
    for sent_i, l in enumerate(f.readlines()):
        if sent_i >= max_sents > 0:
            break
        l = l.rstrip()
        tokens = l.split()
        sentence = [Token('<s>', '<s>')]
        for token in tokens:
            # if no tag available or we are testing, use UNK tag
            try:
                word, tag = token.rsplit('/', 1)
            except ValueError:
                word = token
                tag = 'UNK'
            if test:
                tag = 'UNK'
            sentence.append(Token(word, tag))
        sentence.append(Token('</s>', '</s>'))
        sentences.append(sentence)
    return sentences

# =============================================================================
# Modified the code to print the accuracy vs tags
# =============================================================================
def calc_accuracy(gold, system):
    assert len(gold) == len(system), "Gold and system don't have the same number of sentence"
    tags_correct = 0
    num_tags = 0
    total=collections.defaultdict(int)
    dictt=collections.defaultdict(int)
    for sent_i in range(len(gold)):
        assert len(gold[sent_i]) == len(system[sent_i]), "Different number of token in sentence:\n%s" % gold[sent_i]
        for gold_tok, system_tok in zip(gold[sent_i], system[sent_i]):
            if gold_tok.tag == system_tok.tag:
                dictt[gold_tok.tag ]+=1
                tags_correct += 1
            total[gold_tok.tag]+=1
            num_tags += 1
    for x in dictt:
        dictt[x]=dictt[x]/total[x]
# =============================================================================
#     print(dictt) #to print the accuracy per tag
# =============================================================================
    return (tags_correct / float(num_tags)) * 100


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("PATH",
                        help="Path to file with POS annotations")
    args = parser.parse_args()

    print("First ten lines: \n")
    for i, sentence in enumerate(read_tokens(args.PATH)):
        # if i >= 10:
        #     break
        print(f"{i}: {' '.join(map(str, sentence))}")