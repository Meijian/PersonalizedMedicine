import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
import pickle

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
# print(check_output(["ls", "."]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import xgboost as xgb
import gc

fidx = '1f'
test = True
# test = False

# midx is specified below before the TfidfVectorizor call

# xgboost params
xgb_params = {
    'eta': 0.01,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.5,
    'colsample_bylevel': 1.0,
    'min_child_weight': 1,
    'alpha': 0.0,
    'lambda': 1.0,
    'gamma': 0.0,
    'num_parallel_tree': 1,
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss',
    'silent': 1
}

# lightGBM params
# params = {}
# params['max_bin'] = 10
# params['learning_rate'] = 0.01 # shrinkage_rate
# params['boosting_type'] = 'gbdt'
# params['objective'] = 'regression_l1'
# params['metric'] = 'mae'          # or 'mae'
# params['sub_feature'] = 0.5      # feature_fraction 
# params['bagging_fraction'] = 0.85 # sub_row
# params['bagging_freq'] = 40
# params['num_leaves'] = 512        # num_leaf
# params['min_data_in_leaf'] = 500         # min_data_in_leaf
# # params['min_hessian'] = 0.05     # min_sum_hessian_in_leaf
# params['min_hessian'] = 0.001     # min_sum_hessian_in_leaf
# params['verbose'] = 0
# params['feature_fraction_seed'] = 123
# params['bagging_seed'] = 123
# params['lambda_l1'] = 1
# params['lambda_l2'] = 0

np.random.seed(123)
num_boost_round = 50000
early_stopping_rounds = 30
verbose_eval = 1000

print('Loading data...')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy.sparse as ssp

from sklearn import metrics, model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
train_variants_df = pd.read_csv("training_variants.txt")
test_variants_df = pd.read_csv("test_variants.txt")
train_text_df = pd.read_csv("training_text.txt", sep="\|\|", engine="python", skiprows=1, names=["ID", "Text"])
test_text_df = pd.read_csv("test_text.txt", sep="\|\|", engine="python", skiprows=1, names=["ID", "Text"])
# assuming sorted by ID
train_anno1 = pd.read_csv("train_variant_annotation_0711_tfidf.csv")
if test: test_anno1 = pd.read_csv("test_variant_annotation_0711_tfidf2.csv")
droplist = ['class']
for c,dtype in zip(train_anno1.columns,train_anno1.dtypes):
    # print(c,dtype)
    if dtype == 'object':
        droplist.append(c)
        # lbl = LabelEncoder()
        # lbl.fit(list(train_anno1[c].values))
        # train_anno1[c] = lbl.transform(list(train_anno1[c].values))
    else:
        train_anno1[c] = train_anno1[c].astype(np.float32)
        if test: test_anno1[c] = test_anno1[c].astype(np.float32)
        
# merge blup features
# vw = ['Chr','Ref','Alt','RefAlt','Func_refgene','ExonicFunc_refgene','ClinVar_SIG','ClinVar_STATUS','GWAS_DIS',
# 'SIFT_pred','Polyphen2_HDIV_pred','Polyphen2_HVAR_pred','LRT_pred','MutationTaster_pred','MutationAssessor_pred',
# 'FATHMM_pred','RadialSVM_pred','LR_pred','SF','proteinClass1','proteinClass2','Somatic','Germline','Tissue_Type',
# 'Molecular_Genetics','Role_in_Cancer','Translocation_Partner']

# midx = '1'
# vw = ['Chr','Ref','Alt','Func_refgene','ExonicFunc_refgene','GWAS_DIS',
# 'SIFT_pred','Polyphen2_HDIV_pred','Polyphen2_HVAR_pred','LRT_pred','MutationTaster_pred','MutationAssessor_pred',
# 'FATHMM_pred','RadialSVM_pred','LR_pred','proteinClass1','proteinClass2','Somatic','Germline','Tissue_Type',
# 'Molecular_Genetics','Role_in_Cancer','Translocation_Partner']

midx = '2'
vw = ['Chr','Func_refgene','ExonicFunc_refgene','GWAS_DIS',
'SIFT_pred','Polyphen2_HDIV_pred','Polyphen2_HVAR_pred','LRT_pred','MutationTaster_pred','MutationAssessor_pred',
'FATHMM_pred','RadialSVM_pred','LR_pred','proteinClass1','proteinClass2','Somatic','Germline','Tissue_Type',
'Molecular_Genetics','Role_in_Cancer','Translocation_Partner']

for i,v in enumerate(vw):
    wb = pd.read_csv('blup1/blup1_'+v+'.csv',low_memory=False)
    train_anno1 = train_anno1.merge(wb, how='left', on=[v])
    if test: test_anno1 = test_anno1.merge(wb, how='left', on=[v])

train_anno1.drop(droplist,inplace=True,axis=1)
if test: test_anno1.drop(droplist,inplace=True,axis=1)
        
print(train_variants_df.shape,train_text_df.shape,train_anno1.shape)
if test: print(test_variants_df.shape,test_text_df.shape,test_anno1.shape)

print('TfidfVectorizer')
# midx = '1'
# tfidf = TfidfVectorizer(
# 	min_df=5, max_features=2000, strip_accents='unicode', lowercase=True,
# 	analyzer='word', token_pattern=r'\w+', use_idf=True, ngram_range=(1,5), 
# 	smooth_idf=True, sublinear_tf=True, stop_words = 'english').fit(train_text_df["Text"])

# midx = '2'
tfidf = TfidfVectorizer(
	min_df=1, max_features=4000, strip_accents='unicode', lowercase=True,
	analyzer='word', token_pattern=r'\w+', use_idf=True, ngram_range=(1,5), 
	smooth_idf=True, sublinear_tf=True, stop_words = 'english').fit(train_text_df["Text"])

# concat train and test
test_data = train_text_df.append(test_text_df)
X_tfidf_text = tfidf.transform(test_data["Text"])
print(X_tfidf_text.shape)

# svd feature reduction 
from sklearn.decomposition import TruncatedSVD
print('TruncatedSVD')
ns = 200
svd = TruncatedSVD(ns)
SVD_data = svd.fit_transform(X_tfidf_text)
svd_fnames = np.array(['svd' + str(i+1) for i in range(ns)])

# separate train and test
X_train_text = SVD_data[:train_text_df.shape[0]]
X_test_text = SVD_data[train_text_df.shape[0]:]
print(X_train_text.shape,X_test_text.shape)

features = tfidf.get_feature_names()
# print(features)

ID_train = train_variants_df.ID
ID_test = test_variants_df.ID

y = train_variants_df.Class.values-1

train_variants_df = train_variants_df.drop(['ID','Class'], axis=1)
test_variants_df = test_variants_df.drop(['ID'], axis=1)

data = train_variants_df.append(test_variants_df)

X_data = pd.get_dummies(data)
X_data_fnames = X_data.columns.values

X = X_data[:train_variants_df.shape[0]].values
X_test = X_data[train_variants_df.shape[0]:].values

# reverse order of list from original to avoid dropping final index in test set
X = ssp.hstack([X, pd.DataFrame(X_train_text), train_anno1], format='csr')
if test: X_test = ssp.hstack([X_test, pd.DataFrame(X_test_text), test_anno1], format='csr')

print(X.shape,y.shape)
if test: print(X_test.shape)

# df_train = pd.read_csv('train_2016_v2p.csv')
fnames = np.hstack([X_data_fnames,svd_fnames,train_anno1.columns.values])

# folds for local cross validation
folds = pd.read_csv('folds.csv')
# folds = folds.drop(['Gene','Variation','Class'], axis=1)

# fivefold1 hold out entire genes in each fold and oof mlogloss is much worse
# fold = folds['fivefold2'].values
# fivefold2 is stratified across classes
fold = folds['fivefold2'].values
nfold = 5

nd = X.shape[0]
nf = X.shape[1]
nc = max(y) + 1
print(nd,nf,nc)
xgb_params['num_class'] = nc

x_train0 = X
y_train0 = y

oa = np.hstack((ID_train.values.reshape((nd,1)),np.zeros((nd,nc)))) 
oof = pd.DataFrame(oa)
# oof = train_variants_df[['ID']]

rez = {}
nround = 0
nfit = 0

# double loop over num_leaves and min_data
for i in range(1):

    # xgb_params['subsample'] = 0.2*(i+1)
    xgb_params['subsample'] = 0.6*(i+1)
    # params['num_leaves'] = 2**(i+5)
    # params['num_leaves'] = 75 + i*25
    # params['num_leaves'] = 100 + i*25

    for j in range(1):
    
        # xgb_params['colsample_bytree'] = 0.2*(j+1)
        xgb_params['colsample_bytree'] = 0.8*(j+1)
        # params['min_data'] = 2**(j+8)
        # params['min_data_in_leaf'] = 150 + j*50
        # params['min_data_in_leaf'] = 200 + j*50
    
        # midx = str(params['num_leaves']) + '_' + str(params['min_data_in_leaf'])
        # midx = '_' + str(i+1)+ '_' + str(j+1)
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
            vf = (fold==(f+1))
            x_valid = x_train0[vf]
            y_valid = y_train0[vf]
            
            x_train = x_train0[~vf]
            y_train = y_train0[~vf]

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
            print(x_train.shape, x_valid.shape, y_train.shape, y_valid.shape)
            
            # x_train = x_train.values.astype(np.float32, copy=False)
            # x_valid = x_valid.values.astype(np.float32, copy=False)
            
            d_train = xgb.DMatrix(x_train, label=y_train, feature_names=fnames)
            d_valid = xgb.DMatrix(x_valid, label=y_valid, feature_names=fnames)

            # dtest = xgb.DMatrix(x_test)

            # d_train = lgb.Dataset(x_train, label=y_train)
            # d_train = lgb.Dataset(x_train, label=y_train, weight=w_train)
            # d_valid = lgb.Dataset(x_valid, label=y_valid)
            
            watchlist = [(d_train,'train'),(d_valid,'valid')]
   
            # print('Training...')
            booster = xgb.train(xgb_params, d_train, num_boost_round=num_boost_round, 
                evals=watchlist,verbose_eval=verbose_eval, early_stopping_rounds=early_stopping_rounds)
         
            # nround += booster.best_iteration
            nround += booster.best_ntree_limit
            nfit += 1

            # feature importance
            imp = pd.DataFrame(booster.get_fscore().items(), columns=['feature','importance']).sort_values('importance', ascending=False)
            print(imp.head(n=20))

            # gainf = booster.get_score(importance_type='gain')
            # splitf = booster.get_score(importance_type='weight')   
        
            # print("Predicting...")
        
            pred = booster.predict(d_valid, ntree_limit=booster.best_ntree_limit)
            oof.loc[vf,1:(nc+1)] = pred.reshape((x_valid.shape[0],nc))
        
            # num_threads > 1 predicts very slowly
            # clf.reset_parameter({"num_threads":1})
            # p_testf = booster.predict(x_test)
            # clf.reset_parameter({"num_threads":-1})
            # if f == 0:
                # gain = gainf.copy()
                # split = splitf.copy()
                # p_test = p_testf.copy()
            # else:
                # gain += gainf
                # split += splitf
                # p_test += p_testf
            
            # del x_test; gc.collect()
            del d_train, d_valid; gc.collect()
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
    
oof.to_csv('oof/oof_xgb' + fidx + midx + '.csv', index=False, float_format='%.6f')
rezd = pd.DataFrame({'key':rez.keys(),'value':rez.values})
rezd.to_csv('rez/rez_xgb' + fidx + midx + '.csv', index=False, float_format='%.6f')

if test:
    print('Training on full data...')

    # drop out outliers
    # doo = (abs(y_train0) < outlier_cutoff) 
    # x_train = x_train0[doo]
    # y_train = y_train0[doo]

    # x_train = x_train0
    # y_train = y_train0
    # y_mean = np.average(y_train)

    d_train = xgb.DMatrix(X, label=y, feature_names=fnames)
    d_test = xgb.DMatrix(X_test, feature_names=fnames)

    nround /= nfit
    print(nround)
    # nround1 /= nfit
    # nround2 /= nfit
    # print(nround1,nround2)

    watchlist = [(d_train,'train')]
    nrep = 5

    for i in range(nrep):
    
        # xgb_params['base_score'] = y_mean
        xgb_params['random_state'] = 124+i
 
        nroundr = nround - 50 + random.randint(0,100)
        # nround1r = nround1 - 50 + random.randint(0,100)
        print(str(i+1) + '/' + str(nrep) + ' ' + str(nroundr)) 
 
        booster = xgb.train(xgb_params, d_train, num_boost_round=nroundr,
            evals=watchlist,verbose_eval=verbose_eval)
 
        # booster = xgb.train(xgb_params, d_train, num_boost_round=nroundr, 
        #    evals=watchlist,verbose_eval=verbose_eval)
        
        # # to move from l2 towards l1, compute exponential weights and refit
        # pred = booster.predict(d_train)
        # resid = y_train - pred
        # medr = np.median(resid)
        # wgt = np.exp(-abs(resid-medr)/weight_scale)
        # d_train.set_weight(wgt)
        # y_wmean = np.average(y_train, weights=wgt)
        # xgb_params['base_score'] = y_wmean
 
        # nround2r = nround2 - 50 + random.randint(0,100)
        # print(str(i+1) + '/' + str(nrep) + ' ' + str(nround2r))
        # booster = xgb.train(xgb_params, d_train, num_boost_round=nround2r, 
        #     evals=watchlist,verbose_eval=verbose_eval)
     
        print("Predicting test set...")
        # num_threads > 1 predicts very slowly
        p_testi = booster.predict(d_test)

        # may want to use geometric or logistic mean instead
        if (i==0): p_test = p_testi.copy()
        else: p_test += p_testi
         
        # feature importance
        imp = pd.DataFrame(booster.get_fscore().items(), columns=['feature','importance']).sort_values('importance', ascending=False)
        print(imp.head(n=20))

    p_test /= nrep
    sub = pd.read_csv('submissionFile.csv')
    sub[sub.columns[1:(nc+1)]] = p_test
    fname = 'sub/sub_xgb' + fidx + midx + '.csv'
    sub.to_csv(fname, index=False, float_format='%.6f')
    print(fname)
