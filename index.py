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
    min_word_count = 3 #kata yg dihitung berdasarkan windows size
    num_workers = multiprocessing.cpu_count()
    context_size = 9
    downsampling = 1e-3
    seed = 1

    brown2vec = w2v.Word2Vec(
        sg=True,
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


# 2 method get dibawah adalah method yang sangat memudahkan jika ingin melakukan analisa terhadap
# data yang telah dilatih karena tidak perlu melatih lagi data yang cukup memakan waktu tapi cukup 
# memanggil hasilnya saja 

def getMatrix(matrixName):
    return pd.read_csv(matrixName)

def getModel(modelName):
    return w2v.Word2Vec.load(modelName)

# Proses yang paling banyak memakan waktu kurang lebih 3 jam, hindari memanggil proses ini.
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


def getVector(matrixName,word):
    matrix = getMatrix(matrixName)
    for _,val in matrix.iterrows() :
        if(val.word == word):
            return np.array([val.x,val.y])
    return np.zeros(shape=2)

def calculatePearson(x,y):
    return pearsonr(x,y)
    
def calculateEuclidDist(x,y):
    return norm(x-y)


def getRelation(matrixName):
    x  = pd.read_csv(matrixName, sep=';', lineterminator='\r')['simlex']
    w2v = pd.read_csv(matrixName, sep=';', lineterminator='\r')['w2v']
    simlex = (x - min(x))/(max(x)-min(x))
    # print(simlex)
    print(calculatePearson(simlex,w2v))

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

def mostSimWord(word,modelName):
    words = getModel(modelName).most_similar(word)
    return words

def simWord(wordA,wordB,modelName):
    rank = getModel(modelName).similarity(wordA,wordB)
    return rank

def word_sense(sentence,targetWord):
    sentence = sentence.split()
    return lesk(sentence, targetWord), lesk(sentence, targetWord).definition()

def calculateWordSense(targetWord):
    raw_corpus = u' '.join(brown.words())
        
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(raw_corpus.casefold())

    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(sentenceToWordlist(raw_sentence))

    targetSentence = []
    targetWordKind = []
    targetWordDefs = [] 

    for sentence in sentences :
        if targetWord in sentence:
            targetSentence.append(' '.join(sentence))
            targetWordKind.append(lesk(sentence, targetWord))
            targetWordDefs.append(lesk(sentence, targetWord).definition())
    
    data = pd.DataFrame(
        {
            "Kalimat" : targetSentence,
            "Kata Target" : targetWordKind,
            "Definisi Kata" : targetWordDefs
        }
    )
    
    targetWord += " Word Sense Disambugation.csv" 

    data.to_csv(targetWord, sep=";")

def calculateAllWordSense():
    simlex = pd.read_csv('simlex.csv')
    raw_corpus = u' '.join(brown.words())
        
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(raw_corpus.casefold())

    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(sentenceToWordlist(raw_sentence))

    targetSentence = []
    targetWordKind = []
    targetWordDefs = [] 

    for i, value in simlex.iterrows() :
        for sentence in sentences :
            if value.x in sentence:
                targetSentence.append(' '.join(sentence))
                targetWordKind.append(lesk(sentence, value.x))
                targetWordDefs.append(lesk(sentence, value.x).definition())
                
        for sentence in sentences :
            if value.y in sentence:
                targetSentence.append(' '.join(sentence))
                targetWordKind.append(lesk(sentence, value.y))
                targetWordDefs.append(lesk(sentence, value.y).definition())
        
        print("Finish : "+value.x+" and "+value.y)
    
    data = pd.DataFrame(
        {
            "Kalimat" : targetSentence,
            "Kata Target" : targetWordKind,
            "Definisi Kata" : targetWordDefs
        }
    )

    data.to_csv(" Total.csv", sep=";")

def countAllWordSense():
    simlex = pd.read_csv('simlex.csv')
    raw_corpus = u' '.join(brown.words())
        
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(raw_corpus.casefold())

    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(sentenceToWordlist(raw_sentence))

    targetWordKind = [] 
    wordX = []
    wordXsense = []
    wordY = []
    wordYsense = []

    for i, value in simlex.iterrows() :
        targetWordKind = []
        for sentence in sentences :
            if value.x in sentence:
                targetWordKind.append(lesk(sentence, value.x))
        
        wordX.append(value.x)
        wordXsense.append(len(Counter(targetWordKind)))
                
        targetWordKind = []
        for sentence in sentences :
            if value.y in sentence:
                targetWordKind.append(lesk(sentence, value.y))
        
        wordY.append(value.y)
        wordYsense.append(len(Counter(targetWordKind)))

        print("Finish : "+value.x+" and "+value.y)
    
    data = pd.DataFrame(
        {
            "Word X" : wordX,
            "Word X Sense" : wordXsense,
            "Word Y" : wordY,
            "Word Y Sense" : wordYsense
        }
    )

    data.to_csv(" Total.csv", sep=";", index = False)

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
# print(model.wv.evaluate_word_pairs('simlex.tsv'))
# print(model.wv.evaluate_word_pairs('353.tsv'))

## untuk meemanggil metode hapus tanda pagar di awal

# print(word_sense('I want to play','play'))

# calculateWordSense("car")
countAllWordSense()
