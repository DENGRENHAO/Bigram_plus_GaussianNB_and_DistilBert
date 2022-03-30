import math
from collections import Counter, defaultdict
from typing import List

import nltk
import numpy as np
from nltk.tokenize.toktok import ToktokTokenizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.naive_bayes import GaussianNB
from tqdm import tqdm


class Ngram:
    def __init__(self, config, n=2):
        self.tokenizer = ToktokTokenizer()
        self.n = n
        self.model = None
        self.config = config

    def tokenize(self, sentence):
        '''
        E.g.,
            sentence: 'Here dog.'
            tokenized sentence: ['Here', 'dog', '.']
        '''
        return self.tokenizer.tokenize(sentence)

    def get_ngram(self, corpus_tokenize: List[List[str]]):
        '''
        Compute the co-occurrence of each pair.
        '''
        # begin your code (Part 1)

        ##################

        # new_corpus_tokenize = []
        # word_cnt = {}
        # for token in corpus_tokenize:
        #     for word in token:
        #         if word in word_cnt:
        #             word_cnt[word] += 1
        #         else:
        #             word_cnt[word] = 1
        # for i, token in enumerate(corpus_tokenize):
        #     corpus = []
        #     for word in token:
        #         if word_cnt[word] < 10:
        #             corpus.append('[UNK]')
        #         else:
        #             corpus.append(word)
        #     new_corpus_tokenize.append(corpus)
        ##################

        print("get_ngram")
        token_dict = {}
        features = {}
        for token in corpus_tokenize:
            cur_pair = []
            for word in token:
                if word in token_dict:
                    token_dict[word] += 1
                else:
                    token_dict[word] = 1
                cur_pair.append(word)
                pair = tuple(cur_pair)
                if len(pair)==self.n:
                    if pair in features:
                        features[pair] += 1
                    else:
                        features[pair] = 1
                    cur_pair.pop(0)
        # for pair in features:
        #     features[pair] = (features[pair]+1) / (token_dict[pair[0]] + len(token_dict))
        # features = dict(sorted(features.items(), key=lambda item: item[1], reverse=True))
        # for pair, num in token_dict.items():
        #     print(str(pair)+":"+str(num))
        # for pair, num in features.items():
        #     print(str(pair) + ":" + str(num))
        print("end from get_ngram")
        return token_dict, features
        # end your code
    
    def train(self, df):
        '''
        Train n-gram model.
        '''
        corpus = [['[CLS]'] + self.tokenize(document) for document in df['review']]     # [CLS] represents start of sequence

        # You may need to change the outputs, but you need to keep self.model at least.
        self.model, self.features = self.get_ngram(corpus)

    def compute_perplexity(self, df_test) -> float:
        '''
        Compute the perplexity of n-gram model.
        Perplexity = 2^(-entropy)
        '''
        if self.model is None:
            raise NotImplementedError("Train your model first")

        corpus = [['[CLS]'] + self.tokenize(document) for document in df_test['review']]
        
        # begin your code (Part 2)
        print("compute_perplexity")
        entropys = []
        for token in corpus:
            entropy = 0
            cur_pair = []
            for word in token:
                cur_pair.append(word)
                pair = tuple(cur_pair)
                if len(pair)==self.n:
                    numerator = 1
                    denominator = len(self.model)
                    if pair in self.features.keys():
                        numerator += self.features[pair]
                    if pair[0] in self.model:
                        denominator += self.model[pair[0]]
                    entropy += math.log2(numerator/denominator)
                    cur_pair.pop(0)
            entropy /= (len(token)-self.n+1)
            entropys.append(entropy)
        perplexity = [math.pow(2, -e) for e in entropys]
        print("end from compute_perplexity")
        # end your code

        return perplexity

    def train_sentiment(self, df_train, df_test):
        '''
        Use the most n patterns as features for training Naive Bayes.
        It is optional to follow the hint we provided, but need to name as the same.

        Parameters:
            train_corpus_embedding: array-like of shape (n_samples_train, n_features)
            test_corpus_embedding: array-like of shape (n_samples_train, n_features)
        
        E.g.,
            Assume the features are [(I saw), (saw a), (an apple)],
            the embedding of the tokenized sentence ['[CLS]', 'I', 'saw', 'a', 'saw', 'saw', 'a', 'saw', '.'] will be
            [1, 2, 0]
            since the bi-gram of the sentence contains
            [([CLS] I), (I saw), (saw a), (a saw), (saw saw), (saw a), (a saw), (saw .)]
            The number of (I saw) is 1, the number of (saw a) is 2, and the number of (an apple) is 0.
        '''
        # begin your code (Part 3)
        print("train_sentiment")
        # step 1. select the most feature_num patterns as features, you can adjust feature_num for better score!
        feature_num = 500
        features = dict(sorted(self.features.items(), key=lambda item: item[1], reverse=True))
        select_features = {k: features[k] for k in list(features.keys())[:feature_num]}
        for pair, num in select_features.items():
            print(str(pair) + ":" + str(num))
        # step 2. convert each sentence in both training data and testing data to embedding.
        # Note that you should name "train_corpus_embedding" and "test_corpus_embedding" for feeding the model.
        train_corpus = [['[CLS]'] + self.tokenize(document) for document in df_train['review']]
        train_corpus_embedding = [[0] * feature_num for i in range(df_train['review'].shape[0])]
        total = 0
        for index, token in enumerate(train_corpus):
            cur_pair = []
            for word in token:
                cur_pair.append(word)
                pair = tuple(cur_pair)
                if len(pair) == self.n:
                    if pair in select_features.keys():
                        train_corpus_embedding[index][list(select_features.keys()).index(pair)] += 1
                    cur_pair.pop(0)
            # print(train_corpus_embedding[index])
            if sum(train_corpus_embedding[index])>0:
                total += 1
        print(f"total={total}")

        test_corpus = [['[CLS]'] + self.tokenize(document) for document in df_test['review']]
        test_corpus_embedding = [[0] * feature_num for i in range(df_test['review'].shape[0])]
        for index, token in enumerate(test_corpus):
            cur_pair = []
            for word in token:
                cur_pair.append(word)
                pair = tuple(cur_pair)
                if len(pair) == self.n:
                    if pair in select_features.keys():
                        test_corpus_embedding[index][list(select_features.keys()).index(pair)] += 1
                    cur_pair.pop(0)
        print("end from train_sentiment")
        # end your code

        # feed converted embeddings to Naive Bayes
        nb_model = GaussianNB()
        nb_model.fit(train_corpus_embedding, df_train['sentiment'])
        y_predicted = nb_model.predict(test_corpus_embedding)
        precision, recall, f1, support = precision_recall_fscore_support(df_test['sentiment'], y_predicted, average='macro', zero_division=1)
        precision = round(precision, 4)
        recall = round(recall, 4)
        f1 = round(f1, 4)
        print(f"F1 score: {f1}, Precision: {precision}, Recall: {recall}")


if __name__ == '__main__':
    '''
    Here is TA's answer of part 1 for reference only.
    {'a': 0.5, 'saw': 0.25, '.': 0.25}

    Explanation:
    (saw -> a): 2
    (saw -> saw): 1
    (saw -> .): 1
    So the probability of the following word of 'saw' should be 1 normalized by 2+1+1.

    P(I | [CLS]) = 1
    P(saw | I) = 1; count(saw | I) / count(I)
    P(a | saw) = 0.5
    P(saw | a) = 1.0
    P(saw | saw) = 0.25
    P(. | saw) = 0.25
    '''

    # unit test
    test_sentence = {'review': ['I saw a saw saw a saw. He likes to eat apple']}
    model = Ngram(2)
    model.train(test_sentence)
    print(model.model['saw'])
    print("Perplexity: {}".format(model.compute_perplexity({'review': ['I like apple']})))
