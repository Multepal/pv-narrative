import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.decomposition import NMF, PCA

import sys; sys.path.append("../../local_lib/")
from hac2 import HAC

from string import ascii_uppercase

# Configs
export_dir = '.'


class Corpus:
    """
    Expects a valid prepared TOKEN data frame, i.e. with OHCO index and the columns ['token_str','term_str'].
    """

    def __init__(self, token_df:pd.DataFrame, ohco=[], slug='my_corpus'):
        self.TOKEN = token_df
        self.ohco = ohco
        self.TOKEN = self.TOKEN.reset_index().set_index(self.ohco)
        self.slug = slug
    
    def extract_vocab(self):
        self.VOCAB = self.TOKEN.term_str.value_counts().to_frame('n').sort_index()
        self.VOCAB['gloss'] = None

    def add_chunk_col(self, n_chunks=50):
        col_name = f"chunk_{n_chunks}_id"
        self.TOKEN[col_name] = pd.DataFrame(
            np.array_split(self.TOKEN.index, n_chunks)
        ).stack().reset_index().level_0.values + 1

    def export_data(self):
        tables = "TOKEN VOCAB".split()
        for table in tables:
            try:
                o = getattr(self, table)
                o.to_csv(f"{export_dir}/{self.slug}-corpus-{table}.csv", index=True)
            except AttributeError as e:
                print(f"No table {table}")
                pass        


class Doc:

    def __init__(self, corpus:Corpus, group_cols=[], doc_name='my_doc'):
        self.corpus = corpus
        self.group_cols = group_cols
        self.doc_name = doc_name

    def make_bow_df(self):
        self.BOW = self.corpus.TOKEN\
            .value_counts(self.group_cols + ['term_str'])\
            .to_frame('n').sort_index()

    def make_count_matrix(self):
        self.CTM = self.BOW['n'].unstack(fill_value=0)
        self.DOC_IDX = self.CTM.index
        self.CTM = self.CTM.reset_index(drop=True)
        self.CTM.index.name = 'doc_id'
        self.DOC = pd.DataFrame(index=self.CTM.index)
        self.DOC['n_tokens'] = self.CTM.sum(1)

    def compute_term_significance(self):
        self.VOCAB = pd.DataFrame(index=self.corpus.VOCAB.index)
        self.VOCAB['df'] = self.CTM.astype(bool).sum()
        self.VOCAB['dp'] = (self.VOCAB.df / len(self.CTM)) + .001
        self.VOCAB['dh'] = self.VOCAB.dp * np.log2(1/(self.VOCAB.dp))

    def define_sig_terms(self, max_dh=None, agg_func='mean'):
        if max_dh:
            self.max_dh = max_dh
        else:
            self.max_dh = self.VOCAB.dh.agg(agg_func)
        self.VOCAB['sig'] = self.VOCAB.dh >= self.max_dh

    def make_tfidf_df(self):
        CTMX = self.CTM[self.VOCAB[self.VOCAB.sig].index]
        tfidf_engine = TfidfTransformer(
            norm='l2', 
            use_idf=True, 
            smooth_idf=True)
        self.TFIDF = pd.DataFrame(
            tfidf_engine.fit_transform(CTMX).toarray(), 
            columns=CTMX.columns, index=CTMX.index)    
    
    def compute_pca(self, n_comps=5):
        pca_engine = PCA(n_components=n_comps)
        self.COMPS = pd.DataFrame(pca_engine.fit_transform(self.TFIDF), 
                            index=self.TFIDF.index)
        # self.COMPS.index.names = self.group_cols
        self.LOADINGS = pd.DataFrame(
            pca_engine.components_.T * np.sqrt(pca_engine.explained_variance_), 
                                     index = self.TFIDF.columns)
        self.LOADINGS.index.name = 'term_str'

    def export_data(self, dir="."):
        tables = "DOC BOW VOCAB TFIDF COMPS LOADINGS".split()
        for table in tables:
            try:
                o = getattr(self, table)
                o.to_csv(f"{export_dir}/{self.corpus.slug}-doc-{self.doc_name}-{table}.csv", index=True)
            except AttributeError as e:
                print(f"No table {table}")
                pass


class Cluster:
    
    def __init__(self, doc:Doc):
        self.doc = doc
        # self.TFIDF = self.doc.TFIDF
        self.metric = 'euclidean'
        self.DOC_CLUSTER = pd.DataFrame(index=self.doc.TFIDF.index)

    def make_tfidf_dist_df(self):
        self.X = pd.DataFrame(
            pairwise_distances(self.doc.TFIDF, metric=self.metric), 
            index=self.doc.TFIDF.index, 
            columns=self.doc.TFIDF.index)
        
    def cluster_tfidf(self):
        self.tfidf_hac = HAC(self.doc.TFIDF)
        self.tfidf_hac.get_sims()
        self.tfidf_hac.get_tree()

    def cluster_tfidf_dist(self):
        self.tfidf_dist_hac = HAC(self.X)
        self.tfidf_dist_hac.get_sims()
        self.tfidf_dist_hac.get_tree()
    
    def show_tree(self, color_thresh=None):
        if color_thresh:
            self.tfidf_hac.color_thresh = color_thresh
        self.tfidf_hac.get_cluster_labels()
        self.tfidf_hac.plot_tree()

    def get_clusters(self):
        "Note: This will recreated the column for each combo of metric and color_thresh"
        self.tfidf_hac.get_cluster_labels()
        self.k = len(set(self.tfidf_hac.CLUSTER_LABELS))
        self.DOC_CLUSTER['cluster_label'] = self.tfidf_hac.CLUSTER_LABELS    
        self.CLUSTER = self.DOC_CLUSTER.cluster_label.value_counts().to_frame('n')

        # Handle order
        self.CLUSTER['ord'] = self.CLUSTER.apply(lambda x: self.DOC_CLUSTER[self.DOC_CLUSTER.cluster_label == x.name].head(1).index[0], axis=1)
        self.CLUSTER = self.CLUSTER.sort_values('ord')
        self.CLUSTER['cluster_letter'] = list(ascii_uppercase[:len(self.CLUSTER)])
        self.DOC_CLUSTER['cluster_letter'] = self.DOC_CLUSTER.cluster_label.map(self.CLUSTER.cluster_letter)
        self.CLUSTER = self.CLUSTER.reset_index().set_index('cluster_letter')

    def get_tfidf_by_cluster(self):
        CTM = (
            self.DOC_CLUSTER[['cluster_letter']]
            .join(self.doc.CTM)
            .groupby('cluster_letter')
            .sum()
        )                
        DP = CTM.astype(bool).sum() / len(CTM)
        DH = DP * np.log2(1/DP)
        SIG = DH[DH > DH.mean()].index.tolist()
        CTMX = CTM[SIG]
        tfidf_engine = TfidfTransformer(
            norm='l2', use_idf=True, smooth_idf=True)
        self.TFIDF = pd.DataFrame(
            tfidf_engine.fit_transform(CTMX).toarray(), 
            columns=CTMX.columns, index=CTMX.index) 
        
        self.doc.VOCAB['max_cluster2'] = self.TFIDF.idxmax()
        self.CLUSTER['gloss2'] = self.TFIDF.idxmax(1)
        self.CLUSTER['top_terms2'] = self.TFIDF\
            .apply(lambda x: ', '.join(x.sort_values(ascending=False).head(7).index), axis=1)
        

    def assign_cluster_tfidf_values(self):
        self.CLUSTER_TFIDF = (
            self.doc.TFIDF
            .join(self.DOC_CLUSTER["cluster_letter"])
            .groupby("cluster_letter")
            .mean()
        )
        self.doc.VOCAB['max_cluster'] = self.CLUSTER_TFIDF.idxmax()
        self.CLUSTER['gloss'] = self.CLUSTER_TFIDF.idxmax(1)
        self.CLUSTER['top_terms'] = self.CLUSTER_TFIDF\
            .apply(lambda x: ', '.join(x.sort_values(ascending=False).head(7).index), axis=1)

    def create_cluster_model(self):
        self.CLUSTER_MODEL = self.X.join(self.DOC_CLUSTER.cluster_letter).groupby('cluster_letter').mean().T
        self.CLUSTER_MODEL.index.name = self.X.index.name
        
    def export_data(self, dir="."):
        tables = "DOC_CLUSTER CLUSTER CLUSTER_TFIDF CLUSTER_MODEL".split()
        for table in tables:
            try:
                o = getattr(self, table)
                o.to_csv(f"{export_dir}/{self.doc.corpus.slug}-cluster-{self.k}-{self.doc.doc_name}-{table}.csv", index=True)
            except AttributeError as e:
                print(f"No table {table}")
                pass


class Model:

    def __init__(self, doc:Doc):
        self.doc = doc
        self.n_topic_terms = 10
        self.k =  5 
        self.k_cols = [i for i in range(self.k)]
        self.nmf_init = 'nndsvda' # None, 'random', 'nndsvd', 'nndsvda', 'nndsvdar'
        self.max_iter = 10000
        # Increase alpha_W if you want documents to be more sharply focused on fewer topics.
        self.alpha_W = 0 # THETA doc/topic
        # Increase alpha_H if you want topics to be more distinct and defined by fewer terms.
        self.alpha_H = 0 # PHI topic/term
        self.l1_ratio = .5
        self.solver = 'mu'
        self.beta_loss ='kullback-leibler'

    def alter_k(self, new_k):
        self.k = new_k
        self.k_cols = [i for i in range(self.k)]

    def compute_topics(self):
        nmf_engine = NMF(n_components=self.k, max_iter=self.max_iter, 
            init=self.nmf_init, 
            alpha_W = self.alpha_W,
            alpha_H = self.alpha_H,
            solver = self.solver,
            beta_loss = self.beta_loss
        )
        self.THETA = pd.DataFrame(nmf_engine.fit_transform(self.doc.TFIDF), 
            index=self.doc.TFIDF.index)
        self.PHI = pd.DataFrame(nmf_engine.components_, columns=self.doc.TFIDF.columns)
        self.PHIX = self.PHI * np.log2(self.k/self.PHI.astype(bool).astype(int).sum())
        self.TOPIC = self.PHIX.T.apply(
            lambda x: ', '.join(x.sort_values(ascending=False).head(self.n_topic_terms).index))\
            .T.to_frame('top_terms')
        self.TOPIC.index.name = 'topic_id'
        self.TOPIC['gloss'] = self.PHIX.idxmax(1)

        # Assign max topics to docs
        self.DOC_TOPIC = pd.DataFrame(index=self.doc.DOC.index)
        self.DOC_TOPIC['top_topic'] = self.THETA.idxmax(1).values

        # Handle order
        self.TOPIC['ord'] = self.TOPIC.apply(
            lambda x: self.doc.DOC[self.DOC_TOPIC['top_topic'] == x.name].head(1).index[0], axis=1)
        self.TOPIC = self.TOPIC.sort_values('ord')
        self.TOPIC['topic_letter'] = list(ascii_uppercase[:len(self.TOPIC)])
        # self.TOPIC = self.TOPIC.reset_index().set_index('topic_letter')
        self.DOC_TOPIC['topic_letter'] = self.DOC_TOPIC.top_topic.map(self.TOPIC.topic_letter)
        self.TOPIC = self.TOPIC.reset_index().set_index('topic_letter')

        self.THETA.columns = self.TOPIC.sort_values('topic_id').index.tolist()


    def apply_model(self, doc2_group_cols):

        THETA = self.THETA.copy()
        THETA.index = self.doc.DOC_IDX
        self.DOC2_TOPIC_THETA = (
            self.doc.corpus.TOKEN
            .join(THETA, on=self.doc.group_cols)[THETA.columns]
            .groupby(doc2_group_cols)
            .mean()
        )
        self.DOC2_TOPIC_THETA.columns = self.TOPIC.sort_values('topic_id').index.tolist()
        
        self.DOC2_TOPIC_PHI = (
            self.doc.corpus.TOKEN
            .join(self.PHI.T, on='term_str')
            .dropna()
            .fillna(0)
            .groupby(doc2_group_cols)[self.k_cols]
            .mean()
        )
        self.DOC2_TOPIC_PHI.columns = self.TOPIC.sort_values('topic_id').index.tolist()

    def export_data(self):
        tables = "THETA PHI TOPIC THETA_SEQ DOC2_TOPIC_THETA DOC2_TOPIC_PHI".split()
        for table in tables:
            try:
                o = getattr(self, table)
                o.to_csv(f"{export_dir}/{self.doc.corpus.slug}-topic-{self.k}-{self.doc.doc_name}-{table}.csv", index=True)
            except AttributeError as e:
                print(f"No table {table}")
                pass        
