from __future__ import print_function

import os
import sys
import re
import csv
import codecs
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation

from gensim.models import KeyedVectors

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.image import Iterator
from keras.layers import Dense, Input, Flatten, LSTM, Embedding, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from keras import optimizers

import sys
reload(sys)
sys.setdefaultencoding('utf-8')


import random
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
import pickle

import scipy.sparse as ssp

from sklearn import metrics, model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from subprocess import check_output

import gc

fidx = '1f'
# test = True
test = False

# loop over midx below
midx = '1'
# midx = '2'


# modified from NumpyArrayIterator to create a generator for sequences
# that are typically integer encoding of words in text documents
# this routine can handle large individual sequences by creating random substrings of a fixed length
class NumpySequenceIterator(Iterator):
    """Iterator yielding data from a sequence of numpy arrays

    # Arguments
        x: list of sequence input data
        y: list of labels, same length as x
        max_len: output length of each sequence, padded if input is shorter, randomly
           chosen contiguous subset if input is longer
        batch_size: integer, size of a batch
        y_to_categorical: convert integer y target to one-hot indicator matrix
        num_classes:  number of classes for categorical y
        shuffle: boolean, whether to shuffle the data between epochs
        seed: random seed for data shuffling
    """

    def __init__(self, x, y, max_len=32,
                 batch_size=32, y_to_categorical=True, num_classes=2,
                 shuffle=False, seed=None):
        if y is not None and len(x) != len(y):
            raise ValueError('X (sequences) and y (labels) '
                             'should have the same length. '
                             'Found: len(x) = %s, len(y) = %s' %
                             (len(x), len(y)))
        self.x = x
        if y is not None:
            self.y = np.asarray(y)
        else:
            self.y = None
        self.max_len = max_len
        self.batch_size = batch_size
        self.y_to_categorical = y_to_categorical
        self.num_classes = num_classes
        super(NumpySequenceIterator, self).__init__(len(x), batch_size, shuffle, seed)

    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)

        batch_x = np.zeros((current_batch_size,self.max_len), dtype=np.int)

        for i, j in enumerate(index_array):
            xi = self.x[j]
            x_len = len(xi)
            if x_len <= self.max_len:
               xi = pad_sequences([xi], maxlen=self.max_len)
            else:
               rs = np.random.randint(0,x_len-self.max_len)
               xi = xi[rs:(rs+self.max_len)]
            batch_x[i] = xi
        if self.y is None:
            return batch_x

        batch_y = self.y[index_array]
        if self.y_to_categorical:
            batch_y = to_categorical(batch_y,num_classes=self.num_classes)

        return batch_x, batch_y


# code from lystdo.py
########################################
## set directories and parameters
########################################
BASE_DIR = ''
EMBEDDING_FILE = BASE_DIR + '../bio.nlplab.org/wikipedia-pubmed-and-PMC-w2v.bin'
# TRAIN_DATA_FILE = BASE_DIR + 'train.csv'
# TEST_DATA_FILE = BASE_DIR + 'test.csv'
MAX_SEQUENCE_LENGTH = 5000
MAX_NB_WORDS = 20000
# MAX_NB_WORDS = 200000
EMBEDDING_DIM = 200
# EMBEDDING_DIM = 300
# VALIDATION_SPLIT = 0.1

np.random.seed(2017)
num_lstm = np.random.randint(175, 275)
num_dense = np.random.randint(100, 150)
rate_drop_lstm = 0.15 + np.random.rand() * 0.25
rate_drop_dense = 0.15 + np.random.rand() * 0.25
print(num_lstm,num_dense,rate_drop_lstm,rate_drop_dense)

act = 'relu'
re_weight = False # whether to re-weight classes to fit the 17.5% share in test set

# STAMP = 'lstm_%d_%d_%.2f_%.2f'%(num_lstm, num_dense, rate_drop_lstm, \
#         rate_drop_dense)
STAMP = fidx + '_' + midx

########################################
## index word vectors
########################################
print('Indexing word vectors')

word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, \
        binary=True)
print('Found %s word vectors of word2vec' % len(word2vec.vocab))

########################################
## process texts in datasets
########################################
print('Processing text dataset')

# The function "text_to_wordlist" is from
# https://www.kaggle.com/currie32/quora-question-pairs/the-importance-of-cleaning-text
def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.
    
    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
    
    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    
    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
    # Return a list of words
    return(text)

print('Loading data')

train_variants_df = pd.read_csv("training_variants.txt")
test_variants_df = pd.read_csv("test_variants.txt")
train_text_df = pd.read_csv("training_text.txt", sep="\|\|", engine="python", 
    skiprows=1, names=["ID", "Text"])
test_text_df = pd.read_csv("test_text.txt", sep="\|\|", engine="python", 
    skiprows=1, names=["ID", "Text"])
print(train_variants_df.shape,test_variants_df.shape,train_text_df.shape,
    test_text_df.shape)


# concat train and test
# all_data = train_text_df.append(test_text_df)

# build lists for keras preprocessing
labels_index = {'class1':0,'class2':1,'class3':2,'class4':3,'class5':4,'class6':5,
    'class7':6,'class8':7,'class9':8}  # dictionary mapping label name to numeric id

texts = train_text_df['Text'].tolist()
labels = (train_variants_df['Class'].values - 1).tolist()

for i,t in enumerate(texts):
    texts[i] = text_to_wordlist(t)

print('Found %s texts for training' % len(texts))

if test:
    test_texts = test_text_df['Text'].tolist()
    for i,t in enumerate(test_texts):
        test_texts[i] = text_to_wordlist(t)

    print('Found %s texts for testing' % len(test_texts_1))


# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))




# data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

# labels = to_categorical(np.asarray(labels))


# print('Shape of data tensor:', data.shape)
# print('Shape of label tensor:', labels.shape)


if test:
    test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    # test_ids = np.array(test_ids)


# code from keras/examples/pretrained_word_embeddings.py
# BASE_DIR = ''
# GLOVE_DIR = BASE_DIR + '../glove.6B/'
# TEXT_DATA_DIR = BASE_DIR + '/20_newsgroup/'
# MAX_SEQUENCE_LENGTH = 1000
# MAX_SEQUENCE_LENGTH = 2000
# MAX_NB_WORDS = 10000
# MAX_NB_WORDS = 20000
# EMBEDDING_DIM = 100
# EMBEDDING_DIM = 200
# EMBEDDING_DIM = 300
# VALIDATION_SPLIT = 0.2

# np.random.seed(123)


# first, build index mapping words in the embeddings set
# to their embedding vector

# print('Indexing word vectors')
# 
# embeddings_index = {}
# f = open(os.path.join(GLOVE_DIR, 'glove.6B.'+str(EMBEDDING_DIM)+'d.txt'))
# for line in f:
#     values = line.split()
#     word = values[0]
#     coefs = np.asarray(values[1:], dtype='float32')
#     embeddings_index[word] = coefs
# f.close()
# 
# print('Found %s word vectors.' % len(embeddings_index))



########################################
## prepare embeddings
########################################
print('Preparing embedding matrix')

nb_words = min(MAX_NB_WORDS, len(word_index))+1

embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    if word in word2vec.vocab:
        embedding_matrix[i] = word2vec.word_vec(word)

print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))


print('Training model')

# x_train0 = data
# y_train0 = labels

# nd = x_train0.shape[0]
# nc = y_train0.shape[1]
nd = len(sequences)
nc = max(np.array(labels)) + 1
print(nd,nc)

ID_train = train_variants_df.ID
ID_test = test_variants_df.ID


# folds for local cross validation
folds = pd.read_csv('folds.csv')
# folds = folds.drop(['Gene','Variation','Class'], axis=1)

# fivefold1 hold out entire genes in each fold and oof mlogloss is much worse
# fold = folds['fivefold1'].values
# fivefold2 is stratified across classes
fold = folds['fivefold2'].values
nfold = 5

oa = np.hstack((ID_train.values.reshape((nd,1)),np.zeros((nd,nc)))) 
oof = pd.DataFrame(oa)
# oof = train_variants_df[['ID']]

rez = {}
nround = 0
nfit = 0

# double loop over num_leaves and min_data
for i in range(1):

    # xgb_params['subsample'] = 0.2*(i+1)
    # params['num_leaves'] = 2**(i+5)
    # params['num_leaves'] = 75 + i*25
    # params['num_leaves'] = 100 + i*25

    for j in range(1):
    
        # xgb_params['colsample_bytree'] = 0.2*(j+1)
        # params['min_data'] = 2**(j+8)
        # params['min_data_in_leaf'] = 150 + j*50
        # params['min_data_in_leaf'] = 200 + j*50
    
        # midx = str(params['num_leaves']) + '_' + str(params['min_data_in_leaf'])
        midx = '_' + str(i+1)+ '_' + str(j+1)
        print('\n\nmodel' + midx)
    
        oofv = 'xgb' + fidx + midx
        # oof.insert(2,oofv,0)
    
        # train_columns = train_columns0.remove(tc)
        # train_columns = np.delete(train_columns0,i)
        # train_columns = train_columns0
        # x_train0 = np.array(x_train00[train_columns])
    
        # set object index values
        # for c in x_train0.dtypes[x_train0.dtypes == object].index.values:
        #   x_train0[c] = (x_train0[c] == True)

        # x_train0 = x_train0.values.astype(np.float32, copy=False)
    
        for f in range(nfold):
        
            print('fold ' + str(f+1) + '/' + str(nfold))
        
            # use fold based on date and logerror
            vft = np.where(fold!=(f+1))[0]
            vfv = np.where(fold==(f+1))[0]
            # x_valid = x_train0[vf]
            # y_valid = y_train0[vf]
            
            # x_train = x_train0[~vf]
            # y_train = y_train0[~vf]

            bs = 32
            ns = len(vft)/bs
            vs = len(vfv)
            print(bs,ns,vs)

            x_train = [sequences[i] for i in vft]
            y_train = [labels[i] for i in vft]

            x_valid = [sequences[i] for i in vfv]
            y_valid = [labels[i] for i in vfv]

            nsi_train = NumpySequenceIterator(x=x_train,y=y_train,
                batch_size=bs,max_len=MAX_SEQUENCE_LENGTH,num_classes=nc,
                shuffle=True,seed=1234)

            nsi_valid = NumpySequenceIterator(x=x_valid,y=y_valid,
                batch_size=1,max_len=MAX_SEQUENCE_LENGTH,num_classes=nc,
                shuffle=False)


            # drop out outliers
            # doo = (y_train > -0.4) & (y_train < 0.4) 
            # x_train = x_train[doo]
            # y_train = y_train[doo]

            # print('After removing outliers:')
            # print('Shape train: {}:'.format(train.shape))
            # print('Shape train: {}\nShape test: {}'.format(x_train.shape, x_test.shape))

            # x_train = train_df.drop(['parcelid', 'logerror', 'transactiondate'], axis=1)
            # y_train = train_df["logerror"].values.astype(np.float32)
            # y_mean = np.mean(y_train)
            # xgb_params['base_score'] = y_mean
            
            # use function of month as weight 
            # w_train = np.sqrt(month[~vf])
            # w_train = np.log(1.0 + np.minimum(fold[~vf],10))
            # print(w_train.min(), w_train.max())
            # print(x_train.shape, x_valid.shape, y_train.shape, y_valid.shape)
            
            # x_train = x_train.values.astype(np.float32, copy=False)
            # x_valid = x_valid.values.astype(np.float32, copy=False)
            
            # d_train = xgb.DMatrix(x_train, label=y_train)
            # d_valid = xgb.DMatrix(x_valid, label=y_valid)

            # dtest = xgb.DMatrix(x_test)

            # d_train = lgb.Dataset(x_train, label=y_train)
            # d_train = lgb.Dataset(x_train, label=y_train, weight=w_train)
            # d_valid = lgb.Dataset(x_valid, label=y_valid)
            
            # watchlist = [(d_train,'train'),(d_valid,'valid')]
   
            ########################################
            ## define the model structure
            ########################################
            embedding_layer = Embedding(nb_words,
                    EMBEDDING_DIM,
                    weights=[embedding_matrix],
                    input_length=MAX_SEQUENCE_LENGTH,
                    trainable=True)
            lstm_layer = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)
            
            sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
            embedded_sequences = embedding_layer(sequence_input)
            x = lstm_layer(embedded_sequences)
            
            x = Dropout(rate_drop_dense)(x)
            x = BatchNormalization()(x)
            
            x = Dense(num_dense, activation=act)(x)
            x = Dropout(rate_drop_dense)(x)
            x = BatchNormalization()(x)
            
            preds = Dense(nc, activation='softmax')(x)
            
            # # train a 1D convnet with global maxpooling
            # sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
            # # load pre-trained word embeddings into an Embedding layer
            # embedding_layer = Embedding(num_words,
            #                 EMBEDDING_DIM, dropout=0.2,
            #                 weights=[embedding_matrix],
            #                 input_length=MAX_SEQUENCE_LENGTH,
            #                 trainable=False)

            # embedded_sequences = embedding_layer(sequence_input)
            # x = Conv1D(128, 5, activation='relu')(embedded_sequences)
            # x = Dropout(0.25)(x)
            # x = MaxPooling1D(5)(x)
            # # x = LSTM(256,dropout=0.2,recurrent_dropout=0.2)(x)
            # x = Conv1D(128, 5, activation='relu')(x)
            # x = Dropout(0.25)(x)
            # x = MaxPooling1D(5)(x)
            # x = Conv1D(128, 5, activation='relu')(x)
            # x = Dropout(0.25)(x)
            # x = MaxPooling1D(35)(x)
            # x = Flatten()(x)
            # x = Dense(128, activation='relu')(x)
            # preds = Dense(len(labels_index), activation='softmax')(x)
            # 
            
            ########################################
            ## train the model
            ########################################
            model = Model(sequence_input, preds)
            # opt = optimizers.SGD(lr=0.1,decay=1e-6,momentum=0.9,nesterov=True)
            # opt = optimizers.RMSprop(lr=0.001)
            opt = 'nadam'
            model.compile(loss='categorical_crossentropy',
                          optimizer=opt,metrics=['acc'])
            if f==0: print(model.summary())
            print(STAMP)
            
            early_stopping = EarlyStopping(monitor='val_loss', patience=10)
            bst_model_path = STAMP + '.h5'
            model_checkpoint = ModelCheckpoint(bst_model_path, verbose=1, save_best_only=True, save_weights_only=True)
            
            # hist = model.fit(x_train, y_train, \
            #         validation_data=(x_valid, y_valid), \
            #         epochs=100, batch_size=3, shuffle=True, \
            #        callbacks=[early_stopping, model_checkpoint])
            
            hist = model.fit_generator(nsi_train, ns, \
                    validation_data=nsi_valid, validation_steps=vs, \
                    epochs=100, \
                    callbacks=[early_stopping, model_checkpoint])

            model.load_weights(bst_model_path)
            bst_val_score = min(hist.history['val_loss'])
            print(bst_val_score)


            # booster = xgb.train(xgb_params, d_train, num_boost_round=num_boost_round, 
            #    evals=watchlist,verbose_eval=verbose_eval, early_stopping_rounds=early_stopping_rounds)
         
            # nround += booster.best_iteration
            # nround += booster.best_ntree_limit
            nfit += 1

            # feature importance
            # gainf = booster.feature_importance(importance_type='gain')
            # splitf = booster.feature_importance(importance_type='split')   
        
            # print("Predicting...")
        
            # pred = model.predict(x_valid)
            pred = model.predict_generator(nsi_valid,vs)
            print(pred.shape)
            # oof.loc[vf,1:(nc+1)] = pred.reshape((x_valid.shape[0],nc))
            # oof.loc[vf,1:(nc+1)] = pred.reshape((vs,nc))
            oof.loc[vfv,1:(nc+1)] = pred

            # num_threads > 1 predicts very slowly
            # clf.reset_parameter({"num_threads":1})
            # p_testf = booster.predict(x_test)
            # clf.reset_parameter({"num_threads":-1})
            # if f == 0:
            #    gain = gainf.copy()
            #    split = splitf.copy()
                # p_test = p_testf.copy()
            # else:
            #     gain += gainf
            #    split += splitf
                # p_test += p_testf
            
            # del x_test; gc.collect()
            # del d_train, d_valid; gc.collect()
            del x_train, x_valid; gc.collect()
            
        
        # print("Saving results...")
        
        # bagged prediction
        # p_test /= nfold
            
        # feature importances
        # split = split.astype(float)
        # gain /= nfold
        # split /= nfold
        # gain /= sum(gain)
        # split /= sum(split)
        # imp = pd.DataFrame({'feature':train_columns,'gain':gain,'split':split})
        # imp.sort_values(['gain'],ascending=False,inplace=True)
        # imp.reset_index(inplace=True,drop=True)
        # print(imp.head(n=20))

        # out of fold logloss


        oof_logloss = log_loss(y_train0, np.array(oof)[:,1:(nc+1)])
        print(oofv + " oof logloss = " + str(oof_logloss))
        rez[oofv+'_oof_logloss'] = oof_logloss
        
        # imp.to_csv('imp/imp_xgb' + fidx + midx + '.csv', index=False, float_format='%.6f')
    

oof.to_csv('oof/oof_xgb' + fidx + '.csv', index=False, float_format='%.6f')
rezd = pd.DataFrame({'key':rez.keys(),'value':rez.values})
rezd.to_csv('rez/rez_xgb' + fidx + '.csv', index=False, float_format='%.6f')

if test:
    print('Training on full data...')
    d_train = lgb.Dataset(x_train0, label=y_train0)
    nround /= nfit
    print(nround)
    watchlist = [d_train]
    wnames = ['train']
    nrep = 5

    # month-specific predictions
    sub = pd.read_csv('sample_submission.csv')
    wb = pd.read_csv('wblup/wblup_month.csv',low_memory=False)

    for m in range(10,13):

        df_test.loc[:,'month'] = m
        df_test.loc[:,'wblup_month'] = wb['wblup_month'].values[m-1]

        x_test = df_test[train_columns0]
        print(m,x_test.shape)
    
        # del df_test; gc.collect()
    
        for c in x_test.dtypes[x_test.dtypes == object].index.values:
            x_test.loc[:,c] = (x_test[c] == True)
        x_test = x_test.values.astype(np.float32, copy=False)

            
        for i in range(nrep):
        
            booster.reset_parameter({'feature_fraction_seed':123+i, 'bagging_seed':234+i})
            nroundr = nround - 50 + random.randint(0,100)
        
            print(str(i+1) + '/' + str(nrep) + ' ' + str(nroundr))
        
            booster = lgb.train(params, d_train, nroundr, watchlist, valid_names=wnames,
                verbose_eval=verbose_eval)
            
            print("Predicting test set...")
            # num_threads > 1 predicts very slowly
            # booster.reset_parameter({"num_threads":1})
            p_testi = booster.predict(x_test)
            # booster.reset_parameter({"num_threads":-1})
            if (i==0): p_test = p_testi.copy()
            else: p_test += p_testi
        
        p_test /= nrep

        sub['2016'+str(m)] = p_test
        sub['2017'+str(m)] = p_test
    
    sub.to_csv('sub/sub_xgb' + fidx + midx + '.csv', index=False, float_format='%.6f')

