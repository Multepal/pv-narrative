#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 16:43:29 2025

@author: rca2t
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import plotly_express as px
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF, PCA, LatentDirichletAllocation as LDA
from sklearn.cluster import KMeans

import sys; sys.path.append("../../local_lib/")

from hac2 import HAC
from heatmap import plot_grid, plot_map, CorrelationHeatMap as CHM

from glob import glob



class DocTablle(self):
    
    def __init__(self, doc_df, doc_str_col='doc_str'):
        """

        Parameters
        ----------
        doc_df: pd.DataFrame
            TOKEN dataframe with and OHCO multiIndex

        doc_str_col: str
            DESCRIPTION. The default is 'doc_str'.

        Returns
        -------
        None.

        """
        self.DOC = doc_df




DOC = pd.DataFrame(data, columns=['chap_num','verse_num','line_str']).set_index(['chap_num', 'verse_num'])


# In[329]:


DOC


# # CHAP

# In[330]:


CHAP = DOC.groupby(['chap_num']).line_str.apply(lambda x: ' '.join(map(str, x))).to_frame('book_str')


# In[331]:


CHAP['n_tokens'] = CHAP.book_str.str.split().str.len()


# In[332]:


CHAP


# # TOKEN

# In[333]:


TOKEN = DOC.line_str.str.split(expand=True).stack().to_frame('token_str')


# In[334]:


TOKEN.index.names = list(DOC.index.names[:2]) + ['token_num']


# In[335]:


TOKEN['term_str'] = TOKEN.token_str.str.lower().str.replace(r"\W", "", regex=True)


# In[336]:


TOKEN


# # CHUNK

# In[337]:


n_chunks = 100
TOKEN['chunk_id'] = (
    pd.DataFrame(np.array_split(TOKEN.index, n_chunks))
    .stack()
    .reset_index()
    .level_0
    .values + 1
)


# In[338]:


TOKEN


# In[339]:


CHUNK = TOKEN.value_counts('chunk_id').sort_index().to_frame('n_tokens')


# In[340]:


CHUNK


# # BOW / CTM

# In[341]:


CTM = TOKEN.value_counts(['chunk_id', 'term_str']).unstack(fill_value=0)


# In[342]:


CTM


# # VOCAB

# In[343]:


VOCAB = CTM.sum().to_frame('n')
VOCAB.index.name = 'term_str'
VOCAB['grams'] = VOCAB.apply(lambda x: len(x.name.split()), axis=1)

from stopwords import ENGLISH_STOP_WORDS as swlist

SW = pd.Series(1, index=swlist)
SW.index.name = 'term_str'
SW.name = 'sw'

VOCAB = VOCAB.join(SW)
VOCAB.sw = VOCAB.sw.fillna(0).astype(bool)


# In[344]:


VOCAB['df'] = CTM.astype(bool).sum()
VOCAB['dp'] = VOCAB.df / n_chunks
VOCAB['dh'] = VOCAB.dp * np.log2(1/VOCAB.dp)


# In[345]:


VOCAB


# In[346]:


px.scatter(VOCAB.reset_index(), 'n', 'dh', 
           log_x=True, 
           width=750, height=500,
           hover_name='term_str',
           marginal_y='histogram',
           color='sw',
          title="Term Significance (<i>dh</i>) v Frequency (<i>n</i>)")


# In[347]:


px.scatter(VOCAB.reset_index(), 'df', 'dh', 
           log_x=False, 
           width=750, height=500,
           hover_name='term_str',
           marginal_y='histogram',
          title="Term Significance (<i>dh</i>) v Doc Frequency (<i>df</i>)")


# In[348]:


max_entropy = VOCAB.dh.agg('median')


# In[349]:


VOCAB['sig'] = (VOCAB.dh >= max_entropy) & (VOCAB.sw == False)


# In[350]:


VOCAB[VOCAB.sig == True].sample(10, weights='dh')


# In[351]:


'Number of significant terms', VOCAB[VOCAB.sig].shape[0]


# In[352]:


'Number of rejected words', len(VOCAB[~VOCAB.sig]) 


# In[353]:


'Percent VOCAB used', round((len(VOCAB[VOCAB.sig]) / len(VOCAB)), 2) * 100


# In[354]:


VOCAB['gloss'] = None


# # TFIDF

# In[355]:


CTMX = CTM[VOCAB[VOCAB.sig].index]


# In[356]:


tfidf_engine = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True)
TFIDF = pd.DataFrame(tfidf_engine.fit_transform(CTMX).toarray(), columns=CTMX.columns, index=CTMX.index)    


# # HAC

# In[357]:


TFIDF_SIM = pd.DataFrame(cosine_similarity(TFIDF), index=TFIDF.index, columns=TFIDF.index)


# In[358]:


tfidf_hac = HAC(TFIDF_SIM)


# In[359]:


tfidf_hac.get_sims()
tfidf_hac.get_tree()


# In[360]:


TREE = pd.DataFrame(tfidf_hac.TREE, columns=['i','j','d','n'])


# In[361]:


tfidf_hac.color_thresh = 1 #.75
tfidf_hac.plot_tree()


# In[362]:


fig = sns.clustermap(TFIDF_SIM,
               method='ward', metric='euclidean',
               cmap='Spectral', center=0, cbar_pos=None, 
               col_cluster=False, robust=True, 
               xticklabels=True, yticklabels=True)
plt.setp(fig.ax_heatmap.get_xticklabels(), rotation=0, ha="right")
fig.ax_col_dendrogram.set_visible(False)


# In[363]:


tfidf_hac.get_cluster_labels()


# In[364]:


CHUNK['cluster_label'] = tfidf_hac.CLUSTER_LABELS


# In[365]:


try:
    TOKEN = TOKEN.join(CHUNK.cluster_label, on="chunk_id")
except:
    pass


# In[366]:


TOKEN


# In[367]:


# CHUNK.style.background_gradient()


# In[368]:


CLUSTER_CHUNK = TOKEN.groupby(['cluster_label','chunk_id']).chunk_id.count().unstack(fill_value=0).astype(bool).astype(int)


# In[369]:


sns.clustermap(CLUSTER_CHUNK, 
                    cmap='Spectral', method='ward',
                     cbar_pos=None, center=0, 
                     row_cluster=True, 
                     col_cluster=False,
                    figsize=(12,3))


# In[370]:


CLUSTER = CHUNK.cluster_label.value_counts().to_frame('n_chunks')


# In[371]:


CLUSTER


# In[372]:


labels = {}
ord = 0
for lbl in CHUNK.sort_index().cluster_label.values:
    if lbl not in labels:
        ord += 1
        labels[lbl] = ord


# In[373]:


CLUSTER['ord'] = pd.Series(labels)


# In[374]:


CLUSTER.sort_values('ord')


# In[375]:


label_col = "cluster_label"
CLUSTER_TFIDF = (
    TFIDF
    .join(CHUNK[label_col])
    .groupby(label_col)
    .mean()
)


# In[376]:


CLUSTER_TFIDF.T.sample(5).sort_index().T.style.background_gradient(axis=None)


# In[377]:


VOCAB['max_cluster'] = CLUSTER_TFIDF.idxmax()


# In[378]:


CLUSTER['gloss'] = CLUSTER_TFIDF.idxmax(1)
CLUSTER['top_terms'] = CLUSTER_TFIDF.apply(lambda x: ', '.join(x.sort_values(ascending=False).head(7).index), axis=1)


# In[379]:


CLUSTER


# In[380]:


CHUNK['max_cluster'] = (
    TOKEN
    .join(CLUSTER_TFIDF.T, on='term_str')
    .dropna()
    .groupby('chunk_id')[CLUSTER.index]
    .mean()
    .idxmax(1)
)
CHUNK['max_cluster_gloss'] = CHUNK.max_cluster.map(CLUSTER.gloss)


# In[381]:


CHUNK


# # PCA

# In[382]:


n_comps = 5
pca_engine = PCA(n_components=5)
PCAX = pd.DataFrame(pca_engine.fit_transform(TFIDF), index=TFIDF.index)
PCAX.index.name = 'chunk_id'
LOADINGS = pd.DataFrame(pca_engine.components_.T * np.sqrt(pca_engine.explained_variance_), index = TFIDF.columns)
LOADINGS.index.name = 'term_str'


# In[383]:


X0 = CHUNK.join(PCAX)

def plot_pca(x, y):

    px.scatter(X0, x, y, 
        text=X0.index, 
        height=850, width=950, 
        color=X0.max_cluster_gloss,
        # symbol=X0.max_topic,
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
    


# In[384]:


plot_pca(0,1)


# In[385]:


plot_pca(2,3)


# **Component Histograms**

# In[386]:


def comp_box(comp_id):
    px.box(X0,
           x=comp_id,
           y='max_cluster_gloss',
           color='max_cluster_gloss', 
           # height=400, width=600, 
           title=f'PC {i}').show()


# In[387]:


for i in range(n_comps):
    comp_box(i)


# # NMF

# In[388]:


k = len(set(tfidf_hac.CLUSTER_LABELS))
k


# In[389]:


k_cols = [i for i in range(k)]
k_cols


# In[390]:


n_topic_terms = 10
nmf_engine = NMF(n_components=k * 2, max_iter=10000, 
                 init='nndsvdar', # None, 'random', 'nndsvd', 'nndsvda', 'nndsvdar'
                 # alpha_W=.01,
                 # alpha_H=10,
                 solver='mu', 
                 beta_loss='kullback-leibler'
                )

THETA = pd.DataFrame(nmf_engine.fit_transform(TFIDF), index=TFIDF.index)
THETA_SIM = pd.DataFrame(cosine_similarity(THETA), index=THETA.index, columns=THETA.index)

PHI = pd.DataFrame(nmf_engine.components_, columns=TFIDF.columns)
PHIX = PHI * np.log2(k/PHI.astype(bool).astype(int).sum())
# PHIX = PHI

TOPIC = PHIX.T.apply(lambda x: ', '.join(x.sort_values(ascending=False).head(n_topic_terms).index)).T.to_frame('top_terms')
TOPIC.index.name = 'topic_id'
TOPIC['gloss'] = PHIX.idxmax(1)
# TOPIC['gloss2'] = PHIX.idxmax(1)
# TOPIC['english'] = TOPIC.gloss.map(VOCAB.gloss)
# TOPIC['english2'] = TOPIC.gloss2.map(VOCAB.gloss)


# In[391]:


THETA_SEQ = THETA_SIM.unstack().to_frame('w')
THETA_SEQ.index.names = ['tmp', 'chunk_id']
THETA_SEQ = THETA_SEQ.query("chunk_id == tmp + 1")
THETA_SEQ = THETA_SEQ.reset_index().drop(columns=['tmp']).set_index('chunk_id')
THETA_SEQ['d'] = 1 - THETA_SEQ.w


# In[392]:


TOPIC


# In[393]:


CHUNK[f'top_topic_{k}'] = THETA.idxmax(1).values


# In[394]:


CHUNK


# In[395]:


CHUNK['max_topic'] = THETA.idxmax(1)


# In[396]:


CHUNK[['max_cluster_gloss', 'max_topic']].value_counts().unstack(fill_value=0).style.background_gradient()


# In[397]:


# THETA.join(CHUNK[['max_topic']]).style.background_gradient(cmap="YlGnBu", axis=0)


# In[398]:


THETA.columns = TOPIC.gloss
fig = sns.clustermap(THETA.T, 
                     cmap='Spectral', method='ward',
                     cbar_pos=None, center=0, 
                     row_cluster=True, 
                     col_cluster=False,
                    figsize=(12,3))


# # Classify

# **Model by THETA**

# In[399]:


CHAP_TOPIC_THETA = (
    TOKEN
    .join(THETA, on='chunk_id')[THETA.columns]
    .groupby(['chap_num'])
    .mean()
)


# In[400]:


sns.clustermap(CHAP_TOPIC_THETA.T,
                     cmap='Spectral', method='ward',
                     cbar_pos=None, center=0, 
                     row_cluster=True, 
                     col_cluster=False,
                    figsize=(10,3));


# **Model by PHI**

# In[401]:


CHAP_TOPIC = (
    TOKEN
    .join(PHI.T, on='term_str')
    .dropna()
    .fillna(0)
    .groupby(['chap_num'])[k_cols]
    .mean()
)

# Normalize
# CHAP_TOPIC = (CHAP_TOPIC.T/CHAP_TOPIC.T.sum()).T


# In[402]:


# sns.clustermap(CHAP_TOPIC.T.corr(), cmap='YlGnBu', col_cluster=False);


# In[403]:


sns.clustermap(CHAP_TOPIC.T,
                     cmap='Spectral', method='ward',
                     cbar_pos=None, center=0, 
                     row_cluster=True, 
                     col_cluster=False,
                    figsize=(10,3));


# In[404]:


CHAP['max_topic'] = CHAP_TOPIC[k_cols].idxmax(1)
CHAP['max_topic_theta'] = CHAP_TOPIC_THETA.idxmax(1)
CHAP['max_topic_gloss'] = CHAP.max_topic.map(TOPIC.gloss)


# In[405]:


CHAP


# In[406]:


import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Compute color range for centered colormap
z = THETA.T.values
zmin, zmax = z.min(), z.max()
midpoint = 0
vmax = max(abs(zmin - midpoint), abs(zmax - midpoint))
zmin, zmax = midpoint - vmax, midpoint + vmax

# Create subplot layout
fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    row_heights=[0.6, 0.4],
    vertical_spacing=0.1,
    subplot_titles=("Gloss × Chunk Heatmap", "Chunk Distance from Previous")
)

# Heatmap
heatmap = go.Heatmap(
    z=z,
    x=THETA.T.columns,
    y=THETA.T.index,
    colorscale='Spectral',
    zmin=zmin,
    zmax=zmax,
    showscale=False
)
fig.add_trace(heatmap, row=1, col=1)

# Barplot of distances (by similarity)
bar = go.Bar(
    x=THETA_SEQ.index,  # assuming chunk_id is the index
    y=THETA_SEQ['d'],
    marker_color='royalblue'
)
fig.add_trace(bar, row=2, col=1)

# Layout adjustments
fig.update_layout(
    height=600,
    width=1000,
    margin=dict(t=60),
    showlegend=False
)

fig.update_annotations(font_size=14)
fig.update_yaxes(title_text="gloss", row=1, col=1)
fig.update_yaxes(title_text="d", row=2, col=1)
fig.update_xaxes(title_text="chunk_id", row=2, col=1)

fig.show()


# In[407]:


# CHAP[['max_cluster', 'max_topic_theta']].value_counts().unstack(fill_value=0).style.background_gradient()


# In[408]:


# CHAP_TOPIC_THETA


# In[409]:


CHAP_TOPIC_THETA_SIM = pd.DataFrame(cosine_similarity(CHAP_TOPIC_THETA), index=CHAP_TOPIC_THETA.index, columns=CHAP_TOPIC_THETA.index).stack().to_frame('w')
CHAP_TOPIC_THETA_SIM.index.names = ['tmp', 'chap_num']
CHAP_TOPIC_THETA_SIM = CHAP_TOPIC_THETA_SIM.query("chap_num == tmp + 1").reset_index().set_index('chap_num').drop(columns=['tmp'])
CHAP_TOPIC_THETA_SIM['d'] = 1 - CHAP_TOPIC_THETA_SIM.w
# CHAP_TOPIC_THETA_SIM


# In[410]:


CHAP['wdiff'] = CHAP_TOPIC_THETA_SIM.d
CHAP['new'] = (CHAP_TOPIC_THETA_SIM.d > .25).astype(bool)
CHAP.loc[1, 'new'] = True


# In[411]:


# Compute color range for centered colormap

# Create subplot layout
fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    row_heights=[0.6, 0.4],
    vertical_spacing=0.1,
    subplot_titles=("Glosses over Chpaters", "Chapter Distance from Previous")
)

# Heatmap
z = CHAP_TOPIC_THETA.T.values
zmin, zmax = z.min(), z.max()
midpoint = 0
vmax = max(abs(zmin - midpoint), abs(zmax - midpoint))
zmin, zmax = midpoint - vmax, midpoint + vmax
heatmap = go.Heatmap(
    z=z,
    x=CHAP_TOPIC.T.columns,
    y=TOPIC.gloss, #CHAP_TOPIC.T.index,
    colorscale='Spectral',
    zmin=zmin,
    zmax=zmax,
    showscale=False
)
fig.add_trace(heatmap, row=1, col=1)

# Barplot
X = CHAP_TOPIC_THETA_SIM.join(CHAP).reset_index()
bar = go.Bar(
    x = X.chap_num, 
    y = X.d, #.rolling(2).mean(), 
    marker_color='royalblue'
)
fig.add_trace(bar, row=2, col=1)

# Layout adjustments
fig.update_layout(
    height=600,
    width=1000,
    margin=dict(t=60),
    showlegend=False
)

fig.update_annotations(font_size=16)

# fig.update_yaxes(title_text="chapter", row=1, col=1)
fig.update_yaxes(title_text="topic", row=1, col=1)
fig.update_yaxes(title_text="d", row=2, col=1)

fig.update_xaxes(title_text="chapter number", row=2, col=1)

fig.show()


# In[412]:


CHAP


# # PART

# In[413]:


PART = CHAP.loc[CHAP.new == True].reset_index()
PART.index.name = 'part_num'
PART = PART.reset_index()
PART.part_num = PART.part_num + 1
PART = PART.set_index('part_num')
PART = PART.sort_index()


# In[414]:


PART


# In[ ]:




