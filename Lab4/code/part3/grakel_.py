import numpy as np
import re
from nltk.stem.porter import PorterStemmer

from grakel.utils import graph_from_networkx
from grakel.kernels import (
    WeisfeilerLehman,
    VertexHistogram,
    # ShortestPath,
    PyramidMatch,
    # RandomWalk,
    GraphletSampling
)
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import networkx as nx
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

def load_file(filename):
    labels = []
    docs = []

    with open(filename, encoding='utf8', errors='ignore') as f:
        for line in f:
            content = line.split(':')
            labels.append(content[0])
            docs.append(content[1][:-1])

    return docs, labels


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower().split()


def preprocessing(docs):
    preprocessed_docs = []
    # n_sentences = 0
    stemmer = PorterStemmer()

    for doc in docs:
        clean_doc = clean_str(doc)
        preprocessed_docs.append([stemmer.stem(w) for w in clean_doc])

    return preprocessed_docs


def get_vocab(train_docs, test_docs):
    vocab = dict()

    for doc in train_docs:
        for word in doc:
            if word not in vocab:
                vocab[word] = len(vocab)

    for doc in test_docs:
        for word in doc:
            if word not in vocab:
                vocab[word] = len(vocab)

    return vocab


path_to_train_set = '../datasets/train_5500_coarse.label'
path_to_test_set = '../datasets/TREC_10_coarse.label'

# Read and pre-process train data
train_data, y_train = load_file(path_to_train_set)
train_data = preprocessing(train_data)
print("Number of training documents: ", len(train_data))

# Read and pre-process test data
test_data, y_test = load_file(path_to_test_set)
test_data = preprocessing(test_data)

# Extract vocabulary
vocab = get_vocab(train_data, test_data)
print("Vocabulary size: ", len(vocab))


########## Task 11
def create_graphs_of_words(docs, vocab, window_size):
    graphs = list()
    for idx, doc in enumerate(docs):
        G = nx.Graph()

        for i, word in enumerate(doc):
            if not G.has_node(word):
                G.add_node(word, label=word)
            for j in range(i + 1, min(i + window_size, len(doc))):
                next_word = doc[j]
                if not G.has_node(next_word):
                    G.add_node(next_word, label=next_word)
                if G.has_edge(word, next_word):
                    G[word][next_word]['weight'] += 1
                else:
                    G.add_edge(word, next_word, weight=1)

        graphs.append(G)

    return graphs


# Create graph-of-words representations
G_train_nx = create_graphs_of_words(train_data, vocab, 3)
G_test_nx = create_graphs_of_words(test_data, vocab, 3)

######### Task 12

# Transform networkx graphs to grakel representations
G_train = graph_from_networkx(G_train_nx.copy(), node_labels_tag='label')
G_test = graph_from_networkx(G_test_nx.copy(), node_labels_tag='label')

# Weisfeiler-Lehman subtree kernel
gk = WeisfeilerLehman()

# Construct kernel matrices
K_train = gk.fit_transform(G_train)
K_test = gk.transform(G_test)

############ Task 13

# Train SVM and make predictions

cls = SVC(kernel='precomputed')
cls.fit(K_train, y_train)
y_pred = cls.predict(K_test)

# Evaluate the predictions
print("Accuracy:", accuracy_score(y_pred, y_test))
print("\n")
print("############# Task 14 #############")

######## Task 14

kernels = [
        WeisfeilerLehman(),
        VertexHistogram(),
        GraphletSampling(),
        PyramidMatch(),
]

accs = list()
print("the computation may take a while for GraphletSampling and PyramidMatch kernels")
for kernel in kernels:
    print(f"Kernel used: {str(kernel)}")
    G_train = graph_from_networkx(G_train_nx.copy(), node_labels_tag='label')
    G_test = graph_from_networkx(G_test_nx.copy(), node_labels_tag='label')

    K_train = kernel.fit_transform(G_train)
    K_test = kernel.transform(G_test)

    cls = SVC(kernel='precomputed')
    cls.fit(K_train, y_train)

    y_pred = cls.predict(K_test)
    accuracy = accuracy_score(y_pred, y_test)
    print("Accuracy: ", accuracy)
    print("\n")
    accs.append(accuracy)

print(f"Best kernel: {kernels[np.argmax(accs)]}. The accuracy is {np.max(accs)}.")
