#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF, PCA

import sys; sys.path.append("../../local_lib/")
from hac2 import HAC

from glob import glob

### PARAMETERS

config = dict(
    n_chunks = 100,
    color_thresh = None,
    n_comps = 5
)

### ONE OFF ##########################################################
### External

# Get data
def get_data(glob_pat="./quran-verse-by-verse-text/*.txt"):
    global data
    src_file_names = glob(glob_pat)
    data = []
    for file_name in sorted(src_file_names):
        chap_num, verse_num = file_name.strip()\
            .split("/")[2].split('.')[0].split("-")
        line_str = open(file_name, "r").read()
        data.append((int(chap_num), int(verse_num), line_str))

# Create DOC
def create_doc_df():
    global DOC
    DOC = pd.DataFrame(
        data, 
        columns=['chap_num','verse_num','line_str']
    ).set_index(['chap_num', 'verse_num'])
    
### ONE OFF ##########################################################


#### DEPENDS ON DOC ##################################################

# Extract CHAP
def create_chap_df():
    global CHAP
    CHAP = DOC.groupby(['chap_num']).line_str\
        .apply(lambda x: ' '.join(map(str, x))).to_frame('book_str')
    CHAP['n_tokens'] = CHAP.book_str.str.split().str.len()

# Create TOKEN
def create_token_df():
    global TOKEN, DOC
    TOKEN = DOC.line_str.str.split(expand=True).stack().to_frame('token_str')
    TOKEN.index.names = list(DOC.index.names) + ['token_num']
    TOKEN['term_str'] = TOKEN.token_str.str.lower().str.replace(r"\W", "", regex=True)
    
#### DEPENDS ON DOC ##################################################



#### ONLY DEPENDS ON TOKEN ###########################################

# Create CHUNK
def create_chunk_df():
    global CHUNK, TOKEN
    TOKEN['chunk_id'] = (
        pd.DataFrame(np.array_split(TOKEN.index, config['n_chunks']))
        .stack()
        .reset_index()
        .level_0
        .values + 1
    )
    CHUNK = TOKEN.value_counts('chunk_id').sort_index().to_frame('n_tokens')
    
#### ONLY DEPENDS ON TOKEN ###########################################



#### ONLY DEPENDS ON TOKEN and CTM ###################################

# Create CTM
def create_ctm_df(group_cols=['chunk_id']):
    global CTM, TOKEN
    CTM = TOKEN.value_counts(group_cols + ['term_str']).unstack(fill_value=0)

# Extract VOCAB
def extract_vocab_df():
    global VOCAB, CTM
    VOCAB = CTM.sum().to_frame('n')
    VOCAB.index.name = 'term_str'
    VOCAB['grams'] = VOCAB.apply(lambda x: len(x.name.split()), axis=1)
    
def add_stopwords():
    global VOCAB
    from stopwords import ENGLISH_STOP_WORDS as swlist
    SW = pd.Series(1, index=swlist)
    SW.index.name = 'term_str'
    SW.name = 'sw'
    VOCAB = VOCAB.join(SW)

def compute_term_significance(agg_func='mean'):
    global VOCAB, CTM, max_dh, SIGS
    VOCAB.sw = VOCAB.sw.fillna(0).astype(bool)
    
    # VOCAB['df'] = CTM.astype(bool).sum()
    # VOCAB['dp'] = VOCAB.df / config['n_chunks']
    # VOCAB['df'] = CTM / CTM.sum()
    # VOCAB['dh'] = VOCAB.dp * np.log2(1/VOCAB.dp)
    # max_dh = VOCAB.dh.agg(agg_func)
    # VOCAB['sig'] = (VOCAB.dh >= max_dh) & (VOCAB.sw == False)
    # VOCAB[VOCAB.sig == True].sample(10, weights='dh') # What is this?
    
    DP = CTM / CTM.sum()
    DI = np.log2(1/DP)
    VOCAB['dh'] = (DP * DI).sum()
    SIGS = VOCAB[VOCAB.sw == False].sort_values('dh', ascending=False).head(1000).index

    VOCAB['gloss'] = None

#### ONLY DEPENDS ON TOKEN and CTM ###################################


#### DEPENDS ON VOCAB, CTM, DOC ######################################

def define_doc_df(src_df, copy=False):
    global DOC
    if copy == True:
        DOC = src_df.copy()
    else:
        DOC = src_df

def l2_norm(X):
    return np.sqrt((X ** 2).sum(1))

# TFIDF
def create_tfidf_df():
    global TFIDF, CTM
    # n_chunks = len(CTM)
    # X = CTM[VOCAB[VOCAB.sig].index]
    X = CTM[SIGS]
    TF = X
    CF = TF.sum()
    IDF = np.log2((n_chunks + 1) / (CF + 1) + 1)
    TFIDF = TF * IDF
    TFIDF = TFIDF.div(l2_norm(TFIDF), axis=0)
    # tfidf_engine = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True)
    # TFIDF = pd.DataFrame(tfidf_engine.fit_transform(CTMX).toarray(), columns=CTMX.columns, index=CTMX.index)    

# HAC
def cluster_by_docsim():
    global TFIDF_SIM, tfidf_hac, TOKEN, DOC #, TREE
    TFIDF_SIM = pd.DataFrame(cosine_similarity(TFIDF), index=TFIDF.index, columns=TFIDF.index)
    tfidf_hac = HAC(TFIDF_SIM)
    tfidf_hac.get_sims()
    tfidf_hac.color_thresh = config['color_thresh']
    tfidf_hac.get_tree()
    tfidf_hac.get_cluster_labels()
    
    DOC['cluster_label'] = tfidf_hac.CLUSTER_LABELS    
    TOKEN['cluster_label'] = TOKEN.chunk_id.map(DOC.cluster_label)

# Clustering
def create_cluster_df():
    global CLUSTER, DOC
    CLUSTER = DOC.cluster_label.value_counts().to_frame('n')

    # Order clusters
    labels = {}
    ord = 0
    for lbl in DOC.sort_index().cluster_label.values:
        if lbl not in labels:
            ord += 1
            labels[lbl] = ord
    CLUSTER['ord'] = pd.Series(labels)
    CLUSTER.sort_values('ord')

def create_cluster_chunk_df(group_cols=['chunk_id']):
    global CLUSTER_DOC, TOKEN
    CLUSTER_DOC = (
        TOKEN
        .groupby(['cluster_label'] + group_cols)
        .chunk_id
        .count()
        .unstack(fill_value=0)
        .astype(bool)
        .astype(int)
    )

def create_cluster_tfidf():
    global CLUSTER_TFIDF, TFIDF, DOC
    CLUSTER_TFIDF = (
        TFIDF
        .join(DOC["cluster_label"])
        .groupby("cluster_label")
        .mean()
    )

def assign_cluster_tfidf_values():
    global VOCAB, CLUSTER_TFIDF, CLUSTER, TOKEN, DOC
    VOCAB['max_cluster'] = CLUSTER_TFIDF.idxmax()
    CLUSTER['gloss'] = CLUSTER_TFIDF.idxmax(1)
    CLUSTER['top_terms'] = CLUSTER_TFIDF.apply(lambda x: ', '.join(x.sort_values(ascending=False).head(7).index), axis=1)
    DOC['max_cluster'] = (
        TOKEN
        .join(CLUSTER_TFIDF.T, on='term_str')
        .dropna()
        .groupby('chunk_id')[CLUSTER.index]
        .mean()
        .idxmax(1)
    )
    DOC['max_cluster_gloss'] = DOC.max_cluster.map(CLUSTER.gloss)

# PCA
def compute_pca():
    global PCAX, LOADINGS, TFIDF
    X = TFIDF
    pca_engine = PCA(n_components=config['n_comps'])
    PCAX = pd.DataFrame(pca_engine.fit_transform(X), index=X.index)
    PCAX.index.name = 'chunk_id'
    LOADINGS = pd.DataFrame(pca_engine.components_.T * np.sqrt(pca_engine.explained_variance_), index = X.columns)
    LOADINGS.index.name = 'term_str'

# NMF
def compute_nmf_topics(n_topic_terms = 10, nmf_init='nndsvdar'):
    global k, k_cols, THETA, PHI, TOPIC, tfidf_hac
    k = len(set(tfidf_hac.CLUSTER_LABELS))
    k_cols = [i for i in range(k)]
    nmf_engine = NMF(n_components=k, max_iter=10000, 
        init='nndsvdar', # None, 'random', 'nndsvd', 'nndsvda', 'nndsvdar'
        # alpha_W=.01,
        # alpha_H=10,
        solver='mu', 
        beta_loss='kullback-leibler'
    )
    THETA = pd.DataFrame(nmf_engine.fit_transform(TFIDF), index=TFIDF.index)
    PHI = pd.DataFrame(nmf_engine.components_, columns=TFIDF.columns)
    PHIX = PHI * np.log2(k/PHI.astype(bool).astype(int).sum())
    TOPIC = PHIX.T.apply(lambda x: ', '.join(x.sort_values(ascending=False).head(n_topic_terms).index))\
        .T.to_frame('top_terms')
    TOPIC.index.name = 'topic_id'
    TOPIC['gloss'] = PHIX.idxmax(1)

def assign_max_topics():
    global DOC, THETA, k
    DOC[f'top_topic_{k}'] = THETA.idxmax(1).values
    # DOC['max_topic'] = THETA.idxmax(1)
    # DOC[['max_cluster_gloss', 'max_topic']].value_counts().unstack(fill_value=0).style.background_gradient()

def compute_theta_seq():
    global THETA_SIM, THETA_SEQ, THETA
    THETA_SIM = pd.DataFrame(cosine_similarity(THETA), index=THETA.index, columns=THETA.index)
    THETA_SEQ = THETA_SIM.unstack().to_frame('w')
    THETA_SEQ.index.names = ['tmp', 'chunk_id']
    THETA_SEQ = THETA_SEQ.query("chunk_id == tmp + 1")
    THETA_SEQ = THETA_SEQ.reset_index().drop(columns=['tmp']).set_index('chunk_id')
    THETA_SEQ['d'] = 1 - THETA_SEQ.w

# Classify

#### APPLIES ONE DOC MODEL TO ANOTHER DOC CORPUS

def define_doc2_df(doc2_df, copy=False):
    global DOC2
    if copy == True:
        DOC2 = doc2_df.copy()
    else:
        DOC2 = doc2_df
    
def apply_topic_model(group_cols=['chunk_id'], doc2_col='chap_num'):
    global DOC2_TOPIC_THETA, DOC2_TOPIC, TOKEN, PHI, THETA, k_cols, TOPIC
    
    DOC2_TOPIC_THETA = (
        TOKEN
        .join(THETA, on=group_cols)[THETA.columns]
        .groupby(doc2_col)
        .mean()
    )
    
    DOC2_TOPIC = (
        TOKEN
        .join(PHI.T, on='term_str')
        .dropna()
        .fillna(0)
        .groupby(doc2_col)[k_cols]
        .mean()
    )
    
    DOC2['max_topic'] = DOC2_TOPIC[k_cols].idxmax(1)
    DOC2['max_topic_gloss'] = DOC2.max_topic.map(TOPIC.gloss)
    DOC2['max_topic_theta'] = DOC2_TOPIC_THETA.idxmax(1)
    DOC2['max_topic_thea_gloss'] = DOC2.max_topic_theta.map(TOPIC.gloss)

def create_doc2_topic_theta_sim(doc2_col='chap_num'):
    global DOC2_TOPIC_THETA_SIM, DOC2_TOPIC_THETA
    DOC2_TOPIC_THETA_SIM = pd.DataFrame(
            cosine_similarity(DOC2_TOPIC_THETA), 
            index = DOC2_TOPIC_THETA.index, 
            columns = DOC2_TOPIC_THETA.index
        ).stack().to_frame('w')
    DOC2_TOPIC_THETA_SIM.index.names = ['tmp', doc2_col]
    DOC2_TOPIC_THETA_SIM = DOC2_TOPIC_THETA_SIM.query(f"{doc2_col} == tmp + 1")\
        .reset_index().set_index(doc2_col).drop(columns=['tmp'])
    DOC2_TOPIC_THETA_SIM['d'] = 1 - DOC2_TOPIC_THETA_SIM.w
    DOC2['wdiff'] = DOC2_TOPIC_THETA_SIM.d
    DOC2['new'] = (DOC2_TOPIC_THETA_SIM.d > .25).astype(bool)
    DOC2.loc[1, 'new'] = True

def create_part_df():
    global PART, DOC2
    PART = DOC2.loc[DOC2.new == True].reset_index()
    PART.index.name = 'part_num'
    PART = PART.reset_index()
    PART.part_num = PART.part_num + 1
    PART = PART.set_index('part_num')
    PART = PART.sort_index()

def save_all():
    pass
    

if __name__ == "__main__":

    config['n_chunks'] = 100
    config['color_thresh'] = 1

    # Prepare corpus
    # This creates a CHAP and a CHUNK doc
    get_data()
    create_doc_df()
    create_chap_df()
    create_token_df()
    create_chunk_df()

    # Cluster one doc table, i.e. CHUNK in this case
    # We could also apply CHAP here and forego the comparison below
    define_doc_df(CHUNK)
    create_ctm_df()
    extract_vocab_df()
    add_stopwords()
    compute_term_significance()
    create_tfidf_df()
    cluster_by_docsim()
    create_cluster_df()
    create_cluster_chunk_df()
    create_cluster_tfidf()
    assign_cluster_tfidf_values()
    compute_pca()
    compute_nmf_topics()
    assign_max_topics()
    compute_theta_seq()

    # Apply model to other doc tables, i.e. CHAP
    define_doc2_df(CHAP)
    apply_topic_model()
    create_doc2_topic_theta_sim()
    create_part_df()
    