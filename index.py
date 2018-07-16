from __future__ import absolute_import, division, print_function
import logging
import re
import nltk
import multiprocessing
import gensim.models.word2vec as w2v
import sklearn.manifold
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from nltk.corpus import brown
from nltk.corpus import reuters
import codecs
from nltk.corpus import wordnet
from scipy.stats import pearsonr
from numpy.linalg import norm
from nltk.wsd import lesk
from collections import Counter

# Proses pembersihan corpus

def sentenceToWordlist(raw):
    clean = re.sub("[^a-zA-Z]"," ", raw)
    words = clean.split()
    return words

# Method untuk membangun model latihan w2v, saat ini corpus yang dipakai adalah brown, dengan ukuran
# vocab 21 ribu kata, proses ini cukup memakan waktu bisa mencapai setengah jam

def build_model():
	# raw_corpus = u""
	# with codecs.open("harryPotterCorpus.txt", "r", "utf-8") as book_file:
	# 	raw_corpus = book_file.read()
    raw_corpus = u' '.join(brown.words()) #menggabung kumpulan artikel brown.words()
        
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(raw_corpus.casefold())

    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(sentenceToWordlist(raw_sentence))

    num_features = 300 #dimensi matrix per kata
    min_word_count = 3 #minimun kemunculan kata
    num_workers = multiprocessing.cpu_count()
    context_size = 9 #menentukan pasangan kata
    downsampling = 1e-3 #menentukan banyak jumlah subsampling, jika nilai kata kurang dari subsampling maka kata tsb diabaikan
    seed = 1

    brown2vec = w2v.Word2Vec(
        sg=True, #jika true skip gram, jika false cbow
        seed=seed,
        workers=num_workers,
        size=num_features,
        min_count=min_word_count,
        window=context_size,
        sample=downsampling
    )

    brown2vec.build_vocab(sentences)

    print("Word2Vec vocabulary length:", len(brown2vec.wv.vocab))

    brown2vec.train(sentences,total_examples=brown2vec.corpus_count, epochs=brown2vec.epochs)
    brown2vec.save("potter2vec(9win).w2v")

def compareWords(modelName,saveAs):
    simlex = pd.read_csv('simlex.csv')
    similarity = np.zeros(shape=(len(simlex)))
    simlex_norm = (simlex['val'] - min(simlex['val']))/(max(simlex['val'])-min(simlex['val']))

    for i, value in simlex.iterrows() :
        try:
            similarity[i] = getModel(modelName).similarity(value.x, value.y)
        except KeyError:
            pass

    data = pd.DataFrame(
        {
            "word1" : np.array(simlex['x']),
            "word2" : np.array(simlex['y']),
            "simlex" : np.array(simlex['val']),
            "simlex_norm" : np.array(simlex_norm),
            "w2v" : similarity
        }
    )

    data.to_csv(saveAs,sep=";")

# 2 method get dibawah adalah method yang sangat memudahkan jika ingin melakukan analisa terhadap
# data yang telah dilatih karena tidak perlu melatih lagi data yang cukup memakan waktu tapi cukup 
# memanggil hasilnya saja 

def getMatrix(matrixName):
    return pd.read_csv(matrixName)

def getModel(modelName):
    return w2v.Word2Vec.load(modelName)

def getVector(matrixName,word):
    matrix = getMatrix(matrixName)
    for _,val in matrix.iterrows() :
        if(val.word == word):
            return np.array([val.x,val.y])
    return np.zeros(shape=2)

def calculatePearson(x,y):
    return pearsonr(x,y)
    
def getRelation(matrixName):
    x  = pd.read_csv(matrixName, sep=';', lineterminator='\r')['simlex']
    w2v = pd.read_csv(matrixName, sep=';', lineterminator='\r')['w2v']
    simlex = (x - min(x))/(max(x)-min(x))
    # print(simlex)
    print(calculatePearson(simlex,w2v))

# Proses yang paling banyak memakan waktu
# Proses ini digunakan untuk memperkecil dimensi ruang vektor yang berdimensi banyak menjadi 2
# dimensi agar bisa lebih mudah direpresentasikan dan diukur
def reduceDimensionality(modelName,saveAs):  
    brown2vec = getModel(modelName)
    tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)
    all_word_vectors_matrix = brown2vec.wv.syn0
    all_word_vectors_matrix_2d = tsne.fit_transform(all_word_vectors_matrix)

    points = pd.DataFrame(
        [
            (word, coords[0], coords[1])
            for word, coords in [
                (word, all_word_vectors_matrix_2d[brown2vec.wv.vocab[word].index])
                for word in brown2vec.wv.vocab
            ]
        ],
        columns=["word", "x", "y"]
    )
    points.to_csv(saveAs)

# Dua method show dibawah digunakan untuk menampilkan pemetaan kata-kata yang telah diembed ke dalam
# vektor oleh w2v

def showBigPicture(matrixName):
    points = getMatrix(matrixName)
    fig, ax = plt.subplots()
    ax.scatter(points['x'], points['y'])

    for i, txt in enumerate(points['word']):
        ax.annotate(txt, (points['x'][i], points['y'][i]))
    
    points.plot.scatter("x", "y", s=10, figsize=(20, 12))
    plt.show()

def plot_region(x_bounds, y_bounds, matrixName):
    points = getMatrix(matrixName)

    slice = points[
        (x_bounds[0] <= points.x) &
        (points.x <= x_bounds[1]) & 
        (y_bounds[0] <= points.y) &
        (points.y <= y_bounds[1])
    ]
    
    print(slice)

    ax = slice.plot.scatter("x", "y", s=35, figsize=(10, 8))
    for i, point in slice.iterrows():
        ax.text(point.x + 0.005, point.y + 0.005, point.word, fontsize=11)
    plt.show()

def mostSimWord(word,modelName):
    words = getModel(modelName).most_similar(word)
    return words

def simWord(wordA,wordB,modelName):
    rank = getModel(modelName).similarity(wordA,wordB)
    return rank


# plot_region(x_bounds=(-40.0, -38), y_bounds=(0, 3)) ## -> Untuk menampilkan pemetaan kata pada range tertentu
# showBigPicture('thronesModel.csv') ## -> Untuk menampilkan pemetaan kata
# build_model() ## -> Untuk membuat Model Baru
# compareWords('potter2vec(9win).w2v','hasilPotter(9win).csv') ## -> Untuk menampilkan perbandingan 2 set kata dengan gold standard dan nilai-nilai yang lain
# reduceDimensionality('brown2vec.w2v','model.csv') ## -> Mereduksi dimensi data jadi matriks berdimensi n*2
# print(mostSimWord('know','brown2vec.w2v')) ## -> Untuk menampilkan 10 kata paling mirim dengan kata yang dicari
# print(simWord('know',,'brown2vec.w2v')) ## -> Untuk menampilkan tingkat kemiripan kata satu dengan lainnya

# getRelation('hasil.csv')
# getRelation('hasilPotter.csv')
# getRelation('hasilThrone.csv')
# getRelation('hasilReuters.csv')
# getRelation('hasil(5win).csv')

# reduceDimensionality('brown2vec(5win).w2v','model(5win).csv')

# model = getModel('potter2vec.w2v')

