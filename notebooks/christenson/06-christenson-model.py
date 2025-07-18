#!/usr/bin/env python
# coding: utf-8

# # The Narrative Structure of the _Popol Wuj_:<br/>A Statistical Approach
# 
# Rafael C. Alvarado | Spring 2025

# # Overview
# 
# 

# # Import libraries

# In[91]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import plotly_express as px
import seaborn as sns; sns.set_style("whitegrid")

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF, PCA
from sklearn.cluster import KMeans


# In[92]:


import sys; sys.path.append("../../local_lib/")

from hac2 import HAC
from heatmap import plot_grid, plot_map, CorrelationHeatMap as CHM


# # Set hyperparameters

# In[93]:


# Number of equal size chunks to divide the TOKEN table
n_chunks = 50

# Use if parsing strings of Mayan
# token_pattern = r"(?u)\b\w[\w']*\b"

# These are used with CounterVectorizer
min_ngram = 1
max_ngram = 1

# The limit below which words will be excluded from models
dh_thresh_agg = 'mean' # Use 'median' for much less severe cut


# # Read in CHAP and TOKEN

# In[94]:


CHAP = pd.read_csv("christenson-CHAP-with-text.csv").set_index("chap_num")
TOKEN = pd.read_csv("christenson-TOKEN_QUC.csv").set_index(['chap_num','line_num','token_num'])


# # Create CHUNK

# In[95]:


TOKEN['chunk_id'] = (
    pd.DataFrame(np.array_split(TOKEN.index, n_chunks))
    .stack()
    .reset_index()
    .level_0
    .values + 1
)


# In[96]:


TOKEN


# In[97]:


CHUNK = (
    TOKEN
    .groupby('chunk_id')
    .term_str
    .count()
    .to_frame('n_tokens')
)


# In[98]:


CHUNK.value_counts('n_tokens')


# # Create CHUNK_TO_CHAP

# Chunk to Chapters Index

# In[99]:


CHUNK_TO_CHAP = (
    TOKEN
    .reset_index()
    .groupby(['chunk_id'])
    .chap_num
    .apply(lambda x: sorted(list(set(x))))
    .apply(pd.Series).stack()
    .to_frame('chap_num')
    .sort_index()
).join(CHAP.chap_title, on='chap_num')
CHUNK_TO_CHAP.index.names = ['chunk_id', 'chap_ord']
CHUNK_TO_CHAP['chap_label'] = CHUNK_TO_CHAP.chap_num.astype(int).astype(str) + ': ' + CHUNK_TO_CHAP.chap_title
CHUNK_TO_CHAP.chap_num = CHUNK_TO_CHAP.chap_num.astype(int)


# In[100]:


CHUNK_TO_CHAP


# Chapter Labels for Chunks

# In[101]:


CHUNK['short_label'] = CHUNK_TO_CHAP.groupby('chunk_id').chap_num.apply(lambda x: '[' + str(x.name) + '] ' + ' '.join(map(str,x)))
CHUNK['long_label'] = CHUNK_TO_CHAP.groupby('chunk_id').chap_label.apply(lambda x: '[' + str(x.name) + '] ' + ' '.join(map(str, x)))


# In[102]:


CHUNK.head()


# # Create BOW for CHAP and CHUNK

# In[103]:


BOW_CHAP = TOKEN.groupby(['chap_num', 'term_str']).term_str.count()
BOW_CHUNK = TOKEN.groupby(['chunk_id', 'term_str']).term_str.count()


# # Create CTM

# In[104]:


CTM = BOW_CHUNK.unstack(fill_value=0)


# In[105]:


CTM.sample(10).T.sample(10).T.style.background_gradient(axis=None, cmap='Blues')


# # Extract VOCAB

# In[106]:


VOCAB = CTM.sum().to_frame('n')
VOCAB.index.name = 'term_str'
VOCAB['grams'] = VOCAB.apply(lambda x: len(x.name.split()), axis=1)


# In[107]:


VOCAB


# # Compute DFIDF

# In[108]:


VOCAB['df'] = CTM.astype(bool).sum()
VOCAB['dp'] = VOCAB.df / n_chunks
VOCAB['dh'] = VOCAB.dp * np.log2(1/VOCAB.dp)


# In[109]:


px.scatter(VOCAB.reset_index(), 'n', 'dh', 
           log_x=True, 
           width=750, height=500,
           hover_name='term_str',
           marginal_y='histogram',
          title="Term Significance (<i>dh</i>) v Frequency (<i>n</i>)")


# In[110]:


VOCAB.dh.describe()


# # Define stopwords

# In[111]:


max_entropy = VOCAB.dh.agg(dh_thresh_agg)


# In[112]:


dh_thresh_agg, max_entropy


# In[113]:


VOCAB['stop'] = VOCAB.dh < max_entropy


# In[114]:


VOCAB[VOCAB.stop == False]


# In[115]:


'Number of significant terms', VOCAB[~VOCAB.stop].shape[0]


# In[116]:


'Number of rejected words', len(VOCAB[VOCAB.stop]) 


# In[117]:


'Percent VOCAB used', round((len(VOCAB[~VOCAB.stop]) / len(VOCAB)), 2) * 100


# # Compute TFIDF over non-stopwords
# 
# We weight the significance of words in each chunk using TFIDF.
# 
# TFIDF is an established method for estimating the significance of a word in a context by weighing its local frequency in a context against its global frequency in the corpus.
# 
# - To the degree that a word is frequent in a specific context and the general corpus, it is not considered siginficant.
# - To the degree that a word is frequent in a specific context and infrequenct in the general corpus, it is considered significant.

# In[118]:


CTMX = CTM[VOCAB[~VOCAB.stop].index]


# In[119]:


tfidf_engine = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True)
TFIDF = pd.DataFrame(tfidf_engine.fit_transform(CTMX).toarray(), columns=CTMX.columns, index=CTMX.index)    


# In[120]:


TFIDF.sample(10).sort_index().T.sample(5).sort_index().T.style.background_gradient(axis=None, cmap='YlGnBu')


# # Cluster CHUNK docs by TFIDF
# We generate a square matrix of documents by documents showing pairwise similarities.

# In[121]:


TFIDF_SIM = pd.DataFrame(cosine_similarity(TFIDF), index=TFIDF.index)


# In[122]:


tfidf_hac = HAC(TFIDF_SIM, labels=CHUNK['short_label'].to_list())


# In[123]:


tfidf_hac.get_sims()
tfidf_hac.get_tree()


# In[124]:


TREE = pd.DataFrame(tfidf_hac.TREE, columns=['i','j','d','n'])


# # Choose cluster cut-off
# 
# Play with threshold.\
# Each cut-off highlights different parts.

# In[125]:


tfidf_hac.color_thresh = 1.2
tfidf_hac.plot_tree()


# # View heatmap of document clusters
# 
# A heatmap of the same space provides some insight into the nature of the clusters.

# In[126]:


fig = sns.clustermap(TFIDF_SIM, 
               method='ward', metric='euclidean',
               cmap='Spectral', center=0, cbar_pos=None, 
               col_cluster=False, robust=True, 
               xticklabels=True, yticklabels=True)
plt.setp(fig.ax_heatmap.get_xticklabels(), rotation=0, ha="right")
fig.ax_col_dendrogram.set_visible(False)


# **Interpretation**
# 
# - Two main groups.
# - Visual inspection shows this corresponds to the first and second halves of the book consistent with how the text is divided by all editions.
# - The exception is the group containing chs 1-8 and 39-46 -- this group reflects the theme of creation, which begins both the first and second halves.
# - This "echo" marks are thematic touch point.

# # Apply cluseter labels to chunks

# In[127]:


tfidf_hac.get_cluster_labels()


# In[128]:


CHUNK['cluster_label'] = tfidf_hac.CLUSTER_LABELS


# In[129]:


TOKEN = TOKEN.join(CHUNK.cluster_label, on="chunk_id")


# # Label chapters with clusters

# In[130]:


CHAP['cluster_label'] = (
    TOKEN
    .reset_index()
    .value_counts(['chap_num','cluster_label'])
    .unstack(fill_value=0)
    .idxmax(1)
)


# In[131]:


# CHAP['cluster_label'] = (
#     TOKEN
#     .groupby(['chap_num','cluster_label'])
#     .cluster_label
#     .count()
#     .unstack(fill_value=0)
#     .idxmax(1)
# )


# # Create CLUSTER

# In[132]:


CLUSTER = CHUNK.cluster_label.value_counts().to_frame('n_chunks')


# In[133]:


CLUSTER


# # Add order to CLUSTER

# In[134]:


labels = {}
ord = 0
for lbl in CHUNK.sort_index().cluster_label.values:
    if lbl not in labels:
        ord += 1
        labels[lbl] = ord


# In[135]:


CLUSTER['ord'] = pd.Series(labels)


# In[136]:


CLUSTER.sort_values('ord')


# # Create CLUSTER_TFIDF model

# **Cluster label glosses**
# 
# We find significant words for each cluster based on mean TFIDF grouping by cluster.
# 
# This also creates a model from the labels; that is, each cluster label is associated with a distribution over words.

# In[137]:


label_col = "cluster_label"
CLUSTER_TFIDF = (
    TFIDF
    .join(CHUNK[label_col])
    .groupby(label_col)
    .mean()
)


# In[138]:


CLUSTER_TFIDF.T.sample(5).sort_index().T.style.background_gradient(axis=None)


# In[139]:


VOCAB['max_cluster'] = CLUSTER_TFIDF.idxmax()


# In[140]:


CLUSTER['gloss'] = CLUSTER_TFIDF.idxmax(1)
CLUSTER['top_terms'] = CLUSTER_TFIDF.apply(lambda x: ', '.join(x.sort_values(ascending=False).head(7).index), axis=1)


# In[141]:


CLUSTER


# # Apply CLUSTER_TFIDF model to TOKEN

# **Group by CHUNK**

# In[142]:


CHUNK['max_cluster'] = (
    TOKEN
    .join(CLUSTER_TFIDF.T, on='term_str')
    .dropna()
    .groupby('chunk_id')[CLUSTER.index]
    .mean()
    .idxmax(1)
)
CHUNK['max_cluster_gloss'] = CHUNK.max_cluster.map(CLUSTER.gloss)


# In[183]:


CHUNK[['long_label', 'cluster_label', 'max_cluster','max_cluster_gloss']].style.background_gradient(cmap="YlGnBu")


# No difference. 

# **Group by CHAP**

# In[144]:


CHAP['max_cluster'] = (
    TOKEN
    .join(CLUSTER_TFIDF.T, on='term_str')
    .dropna()
    .groupby('chap_num')[CLUSTER.index]
    .mean()
    .idxmax(1)
)
CHAP['max_cluster_gloss'] = CHAP.max_cluster.map(CLUSTER.gloss)


# In[145]:


CHAP[['chap_title', 'cluster_label', 'max_cluster', 'max_cluster_gloss']].style.background_gradient(cmap="YlGnBu")


# In[146]:


# CHAP_THETA = CHAP.groupby(['chap_num', 'max_cluster']).max_cluster.count().unstack(fill_value=0)


# In[147]:


# fig = sns.clustermap(CHAP_THETA.T, 
#                      cmap='Spectral', method='ward',
#                      cbar_pos=None, center=0, 
#                      row_cluster=True, 
#                      col_cluster=False,
#                     figsize=(12,3))


# # Apply PCA to TFIDF
# 
# PCA sheds light on the relationship between the clusters.
# 
# We add $1$ to $k$ if odd to ensure we have pairs to display.

# In[148]:


X = TFIDF


# In[149]:


pca_engine = PCA(n_components=5)
PCAX = pd.DataFrame(pca_engine.fit_transform(X), index=X.index)
PCAX.index.name = 'chunk_id'
LOADINGS = pd.DataFrame(pca_engine.components_.T * np.sqrt(pca_engine.explained_variance_), index = X.columns)
LOADINGS.index.name = 'term_str'


# # Visualize compontents and clusters

# In[150]:


X0 = CHUNK.join(PCAX)

def plot_pca(x, y):

    px.scatter(X0, x, y, 
        text=X0.index, 
        height=850, width=950, 
        color=X0.max_cluster_gloss,
        size = [1 for i in range(len(X0))],
        marginal_x='box', 
        marginal_y='box').show()

    quantile = .99
    A = LOADINGS.loc[np.abs(LOADINGS[x]) >= np.abs(LOADINGS[x]).quantile(quantile), x]
    B = LOADINGS.loc[np.abs(LOADINGS[y]) >= np.abs(LOADINGS[y]).quantile(quantile), y]    
    C = pd.concat([A,B], axis=1).index
    
    px.scatter(LOADINGS.loc[C].join(VOCAB), x, y, 
        title=f"Quantile {quantile} Loadings for {x} and {y}",
        opacity=.5,
        text=C, 
        size='n',
        height=850, width=950).show()
    


# In[151]:


plot_pca(0,1)


# In[152]:


plot_pca(2,3)


# In[153]:


def comp_box(comp_id):
    px.box(X0[comp_id], color=X0.max_cluster_gloss, height=500, width=600, title=f'PC{i}').show()


# In[154]:


for i in range(5):
    comp_box(i)


# # Generate NMF model
# 
# Topic modeling with NMF gives further insight into the content of the clusters and how they combine in documents, i.e. units of narrative.
# 
# As a form of soft clustering, it lends insight into how each text segment relates to each cluster.
# 
# It also corroborates the clustering.

# **Choose $k$ based on cluster threshold**
# 
# We choose $k$ based on the chosen cut-off threshold for the clustering above.

# In[168]:


k = len(set(tfidf_hac.CLUSTER_LABELS))
k


# Create a list of columns for selection operations.

# In[169]:


k_cols = [i for i in range(k)]


# In[170]:


n_topic_terms = 10
nmf_engine = NMF(n_components=k, max_iter=5000, init='nndsvda', solver='mu', beta_loss='kullback-leibler')
THETA = pd.DataFrame(nmf_engine.fit_transform(TFIDF), index=TFIDF.index)
THETA_SIM = pd.DataFrame(cosine_similarity(THETA), index=THETA.index, columns=THETA.index)
PHI = pd.DataFrame(nmf_engine.components_, columns=TFIDF.columns)
TOPIC = PHI.T.apply(lambda x: ', '.join(x.sort_values(ascending=False).head(n_topic_terms).index)).T.to_frame('top_terms')
TOPIC.index.name = 'topic_id'
CHUNK[f'top_topic_{k}'] = THETA.idxmax(1).values


# In[171]:


with open("topics.md", "w") as outfile:
    TOPIC.to_markdown(outfile)


# In[172]:


TOPIC['gloss'] = PHI.idxmax(1)


# In[173]:


TOPIC


# # Compare to clusters

# In[174]:


CLUSTER.sort_values('gloss')


# # View topics over syuzhet

# In[78]:


THETA.columns = TOPIC.gloss
fig = sns.clustermap(THETA.T, 
                     cmap='Spectral', method='ward',
                     cbar_pos=None, center=0, 
                     row_cluster=True, 
                     col_cluster=False,
                    figsize=(12,3))


# # Classify chapters with NMF model

# In[79]:


# This is slow ...
# CHAP['max_topic_test'] = (
#     BOW_CHAP.to_frame('tf')
#     .join(PHI.T, on='term_str')
#     .apply(lambda x: x.tf * x[k_cols], axis=1)
#     .dropna()
#     .groupby('chap_num')
#     .mean()
#     .idxmax(1)
# )


# In[80]:


CHAP_TOPIC = (
    TOKEN
    .join(PHI.T, on='term_str')
    .dropna()
    .groupby(['chap_num'])[k_cols]
    .mean()
)


# In[81]:


CHAP_TOPIC


# In[82]:


sns.clustermap(CHAP_TOPIC.T, 
                     cmap='Spectral', method='ward',
                     cbar_pos=None, center=0, 
                     row_cluster=False, 
                     col_cluster=False,
                    figsize=(10,3));


# In[83]:


CHAP['max_topic'] = CHAP_TOPIC[k_cols].idxmax(1)
CHAP['max_topic_gloss'] = CHAP.max_topic.map(TOPIC.gloss)


# In[84]:


CHAP[['chap_title', 'max_topic_gloss','max_topic', 'cluster_label', 'max_cluster']].style.background_gradient(cmap='YlGnBu')


# # Conclusion
# **UPDATE TO INCLUDE CLASSIFICATION**
# 
# - Six chapters
# - The text is divided into two major parts, corresponding to before and after the first dawn.
# - Each major part has three parts: Creation, Part A, and Part B.
# - The goings on in Xibalba have the moist variability.
# - Proposed structure: 6 parts with a preamble and a coda.
# - Each part is distinguished temporally and ontologically.
# - The sequence matters: We go from the creation of the world to the creation of society.
# - The sequence charts a lineage from the K'iche' kings to the creator gods. 

# # Save

# In[85]:


savers = [
    (TFIDF, 'TFIDF'),
    (TFIDF_SIM, 'TFIDF_SIM')
]


# In[86]:


for saver in savers:
    obj = saver[0]
    slug = saver[1]
    obj.to_csv(f"christenson-{slug}.csv", index=True)

