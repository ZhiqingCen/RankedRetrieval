import sys
import os
import re
import pickle
from math import log10
from functools import reduce
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords, wordnet
from nltk.tag import pos_tag
import nltk

# stop_words = stopwords.words('english')
# tokenizer = RegexpTokenizer('')
# ps = PorterStemmer()

'''
my_index = {
    token: {
        # tf = the number of times that term t occurs in document d
        # df = number of documents that contains term t
        # df: 0, # df not needed
        # doc_id: {'tf': 0, line: [pos]}, # tf not needed
        doc_id: {line: [pos]}
    }
}
'''

class MyIndex:
    def __init__(self, input_path):
        self.my_index = {}
        self.input_path = input_path
        self.stop_words = stopwords.words('english')
        # self.ps = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        # self.tags = ['ADJ', 'ADV', 'NOUN', 'VERB']
        self.token_num = 0
        self.tags = {
            'ADJ': wordnet.ADJ,
            'ADV': wordnet.ADV,
            'ADP': wordnet.ADV,
            'NOUN': wordnet.NOUN,
            'VERB': wordnet.VERB,
        }
    
    def add_document(self, tokens, doc_id):
        # add tokens to index for one document
        for line_num , line in tokens:
            for position, token in line:
                # if token == '' or token in self.stop_words:
                if token == '':
                    # ignore empty token and stop words
                    continue
                if token in self.my_index:
                    # token exist in index, add to it
                    if doc_id in self.my_index[token]:
                        # doc_id exist in token
                        # self.my_index[token][doc_id]['tf'] += 1
                        if line_num in self.my_index[token][doc_id]:
                            # line_num exist in token
                            self.my_index[token][doc_id][line_num].append(position)
                        else:
                            self.my_index[token][doc_id][line_num] = [position]
                    else:
                        # doc_id not exist in token, update df
                        # self.my_index[token]['df'] += 1
                        # self.my_index[token][doc_id] = {'tf': 1, line_num: [position]}
                        self.my_index[token][doc_id] = {line_num: [position]}
                else:
                    # token not in index, create a new token
                    # self.my_index[token] = {'df': 1, doc_id: {'tf': 1, line_num: [position]}}
                    # self.my_index[token] = {doc_id: {'tf': 1, line_num: [position]}}
                    self.my_index[token] = {doc_id: {line_num: [position]}}
                self.token_num += 1
    
    def get_input_path(self):
        return self.input_path
    
    def get_term_num(self):
        # total number of terms
        return len(self.my_index)
    
    def get_token_num(self):
        # total number of tokens
        # token_num = 0
        # for token in self.my_index.values():
        #     for doc_id in token.keys():
        #         # if doc_id == 'df':
        #         #     continue
        #         for _ in token[doc_id].values():
        #             token_num += token[doc_id]['tf']
        # return token_num
        return self.token_num
            
    def get_whole_index(self):
        # get the whole inverted index
        return self.my_index
    
    def get_token_postings_lists(self, token):
        # get all postings lists for given token
        if token in self.my_index:
            return self.my_index[token]
        else:
            return []
        
    # def preprocessing(self, words):
    def preprocessing(self, line):
        words = word_tokenize(line)
        # token preprocessing before storing in inverted index
        tokens = []
        for word in words:
            if any(i.isdigit() for i in word):
                # numbers
                if '.' in word:
                    # ignore numbers with decimals
                    tokens.append('')
                elif ',' in word:
                    # ignore commas in numeric tokens
                    tokens.append(word.replace(',', ''))
                elif '/' in word or '-' in word:
                    # split dates into day, month and year
                    nums = re.split('/|-|_', word)
                    for i in nums:
                        tokens.append(i)
                else:
                    tokens.append(word)
            elif any(i.isalpha() for i in word) and len(word) > 1:
                # convert to lower case, ignore dot and ' in words
                word = word.lower().replace('.', '').replace("'", "")
                word_list = re.split('/|-|_', word)
                word_list = pos_tag([i for i in word_list if i], tagset='universal', lang='eng')
                for w, tag in word_list:
                    if tag in self.tags:
                        w = self.lemmatizer.lemmatize(w, self.tags[tag])
                    if w.endswith('ing'):
                        w = self.lemmatizer.lemmatize(w, wordnet.VERB)
                    tokens.append(w)
            # elif len(word) == 1 and (word == '.' or word == '?' or word == '!' or word.isalpha()):
            #     # store end of sentences
            #     tokens.append(word)
        return tokens
        
    def debug(self):
        for token in self.my_index:
            print(f'{token}: {self.my_index[token]}')
    
    def calculate_tfidf(tf, df, n):
        return (1 + log10(tf)) * log10(n/df)

def read_documents(input_path, index):
    # read each document and generate inverted index
    doc_num = 0
    for filename in os.listdir(input_path):
        # open as read-only
        with open(os.path.join(input_path, filename), 'r') as f:
            tokens = []
            line_num = 0
            pos_num = 0
            for line in f:
                # read file line by line
                # words = word_tokenize(line)
                # line = enumerate(index.preprocessing(words))
                # line = enumerate(index.preprocessing(line))
                line = [(pos+pos_num, token) for pos, token in enumerate(index.preprocessing(line))]
                tokens.append((line_num, line))
                pos_num += len(line)
                # print(line, pos_num)
                line_num += 1
            index.add_document(tokens, filename)
            doc_num += 1
    return doc_num

def print_output(doc_num, token_num, term_num):
    print(f'Total number of documents: {doc_num}')
    print(f'Total number of tokens: {token_num}')
    print(f'Total number of terms: {term_num}')
    
if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit('Error: python3 index.py [folder-of-documents] [folder-of-indexes]')
    
    input_path = sys.argv[1]
    index_path = sys.argv[2]
    if not os.path.isdir(input_path):
        sys.exit(f'Error: {input_path} is not a directory')
    if not os.path.isdir(index_path):
        # create new directory if folder-of-indexs not exists
        os.mkdir(index_path)
    
    nltk.download('stopwords', quiet=True) # TODO: remove this?
    nltk.download('punkt', quiet=True) # TODO: remove this?
    nltk.download('wordnet', quiet=True) # TODO: remove this?
    nltk.download('omw-1.4', quiet=True) # TODO: remove this?
    nltk.download('averaged_perceptron_tagger', quiet=True) # TODO: remove this?
    nltk.download('universal_tagset', quiet=True) # TODO: remove this?
    
    inverted_index = MyIndex(input_path)
    doc_num = read_documents(input_path, inverted_index)

    # inverted_index.debug() # TODO: remove

    print_output(doc_num, inverted_index.get_token_num(), inverted_index.get_term_num())
    with open(f'{index_path}/index.pkl', 'wb') as f:
        pickle.dump(inverted_index, f)
