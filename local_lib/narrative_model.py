import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF, PCA
from hac2 import HAC
from narrative_parser import NarrativeParser


class NarrativeModel:
    """Models narrative structure from chunked text using TFIDF, HAC, PCA, and NMF."""

    # General
    use_sim = True
    linkage_method='ward'
    distance_metric='euclidean'

    # HAC settings
    hac_color_thresh = 1

    # PCA settings
    n_pca_comps = 5

    # NMF settings
    # n_topics = 6
    nmf_random_state = 42
    n_topic_terms = 5
    # alpha = 0.

    # def __init__(self, src_id: str, CHUNK: pd.DataFrame, TFIDF: pd.DataFrame):
    def __init__(self, parser:NarrativeParser, n_topics=6, alpha=0.0):
        self.src_id = parser.src_id
        self.CHUNK = parser.CHUNK.copy()
        self.TFIDF = parser.TFIDF.copy()
        self.n_topics = n_topics
        self.alpha = alpha


    def compute_tfidf_sim(self):
        self.TFIDF_SIM = pd.DataFrame(cosine_similarity(self.TFIDF), index=self.TFIDF.index)

    def plot_tfidf_clustermap(self):
        X = self.TFIDF_SIM if self.use_sim else self.TFIDF
        fig = sns.clustermap(X,
            method=self.linkage_method, 
            metric=self.distance_metric,
            cmap='Spectral', center=0, cbar_pos=None,
            col_cluster=False, robust=True,
            xticklabels=True, yticklabels=True, figsize=(20, 20))
        plt.setp(fig.ax_heatmap.get_xticklabels(), rotation=0, ha="right")
        fig.ax_col_dendrogram.set_visible(False)
        plt.title("Clustermap of Document Similarity", fontsize=24, y=1.01)
        plt.savefig(f"{self.src_id}-tfidf-sims-clustermap.svg", bbox_inches='tight')
        plt.savefig(f"{self.src_id}-tfidf-sims-clustermap.png", bbox_inches='tight')

    def cluster(self, plot=True):
        X = self.TFIDF_SIM if self.use_sim else self.TFIDF
        tfidf_hac = HAC(X)
        tfidf_hac.dist_metric = self.distance_metric
        tfidf_hac.linkage_method = self.linkage_method
        tfidf_hac.color_thresh = self.hac_color_thresh
        
        if plot:
            tfidf_hac.plot()
            sns.despine(left=True, bottom=True)
            plt.title("Dendrogram of Documents in TFIDF Space", fontsize=14, y=1.01)
            plt.savefig(f"{self.src_id}-tfidf-sims-hac.svg", bbox_inches='tight')
            plt.savefig(f"{self.src_id}-tfidf-sims-hac.png", bbox_inches='tight')
            plt.show()

        tfidf_hac.get_cluster_labels()
        label_col = 'hac_label'
        self.CHUNK[label_col] = tfidf_hac.CLUSTER_LABELS

        self.CLUSTER = self.CHUNK[label_col].value_counts().to_frame('n_chunks')
        grouped = self.TFIDF.join(self.CHUNK[label_col]).groupby(label_col).mean()
        self.CLUSTER['gloss'] = grouped.idxmax(1)
        self.CLUSTER['top_terms'] = grouped.apply(
            lambda x: ', '.join(x.sort_values(ascending=False).head(7).index), axis=1
        )

    def compute_pca(self):
        pca_engine = PCA(n_components=self.n_pca_comps)
        self.PCAX = pd.DataFrame(pca_engine.fit_transform(self.TFIDF), index=self.TFIDF.index)
        self.PCAX.index.name = 'doc_id'
        self.LOADINGS = pd.DataFrame(
            pca_engine.components_.T * np.sqrt(pca_engine.explained_variance_),
            index=self.TFIDF.columns
        )
        self.LOADINGS.index.name = 'term_str'
        self.pca_engine = pca_engine

    def compute_nmf(self):
        nmf_engine = NMF(
            n_components=self.n_topics, 
            max_iter=5000,
            init='nndsvda', # Default when n_components < n_features
            solver='mu', # Requred to handle beta_loss = 'kullback-leibler'
            beta_loss='kullback-leibler', # Best for count-based data or discrete distributions
            random_state=self.nmf_random_state,
            alpha_W = self.alpha, # Default is 0 ... may want to tweak
             alpha_H='same'
        )
        self.THETA = pd.DataFrame(nmf_engine.fit_transform(self.TFIDF), index=self.TFIDF.index)
        self.PHI = pd.DataFrame(nmf_engine.components_, columns=self.TFIDF.columns)

        self.TOPIC = self.PHI.T.apply(
            lambda x: ', '.join(x.sort_values(ascending=False).head(self.n_topic_terms).index)
        ).T.to_frame('top_terms')
        self.TOPIC.index.name = 'topic_id'
        self.CHUNK[f'top_topic_{self.n_topics}'] = self.THETA.idxmax(1).values
        self.TOPIC['gloss'] = self.PHI.idxmax(1)
        self.TOPIC['label'] = self.TOPIC.apply(lambda x: f"T{x.name} {x.gloss}", axis=1)

    def get_topic_sort_order(self):
        topic_seq = self.THETA.idxmax(1).tolist()
        topic_order = []
        for t in topic_seq:
            if t not in topic_order:
                topic_order.append(t)
        self.topic_sort_order = topic_order

    def plot_topics_over_time(self):
        fig, ax = plt.subplots(figsize=(10, 2))
        if not self.topic_sort_order:
            self.get_topic_sort_order()
        sns.heatmap(
            self.THETA[self.topic_sort_order].T.set_index(self.TOPIC['label']),
            cmap='Spectral', center=0, cbar=None
        )
        plt.title(f"{self.src_id.title()}: Topics over Narrative Time")
        ax.set_xlabel("Syntagm", fontsize=10)
        ax.set_ylabel("Paradigm", fontsize=10)
        plt.savefig(f"{self.src_id}-topic-over-doc.svg", bbox_inches='tight')
        plt.savefig(f"{self.src_id}-topic-over-doc.png", bbox_inches='tight')
        plt.show()

    def save(self):
        self.CHUNK.to_csv(f"{self.src_id}-CHUNK2.csv", index=True)
        self.CLUSTER.to_csv(f"{self.src_id}-CLUSTER.csv", index=True)
        self.TOPIC.to_csv(f"{self.src_id}-TOPIC.csv", index=True)
        self.THETA.to_csv(f"{self.src_id}-THETA.csv", index=True)
        self.PHI.to_csv(f"{self.src_id}-PHI.csv", index=True)
        self.LOADINGS.to_csv(f"{self.src_id}-PCA_LOADINGS.csv", index=True)
        self.PCAX.to_csv(f"{self.src_id}-PCA_DCM.csv", index=True)

    def run(self):
        self.compute_tfidf_sim()
        self.plot_tfidf_clustermap()
        self.cluster()
        self.compute_pca()
        self.compute_nmf()
        self.get_topic_sort_order()
        self.plot_topics_over_time()
        self.save()


from sklearn.base import BaseEstimator, TransformerMixin

class NarrativeModelStep(BaseEstimator, TransformerMixin):
    """
    sklearn transformer wrapping NarrativeModel.

    fit(parser)       → fits NMF; derives topic sort order; stores model.
    transform(parser) → returns THETA with columns reordered to narrative order.

    Topic column reordering (previously done by fix_cols in the notebook) is
    now encapsulated here. After fitting, topic columns always run 0..n_topics-1
    in the order the topics first appear in the narrative — so THETA tables from
    different sources are directly comparable when concatenated.

    When transform() is called on a *new* parser (not the training one), it
    projects that parser's TFIDF into the fitted topic space: C = PHI @ TFIDF.T.
    This replaces the ad-hoc TestModel class from the original notebook.
    """

    def __init__(self, n_topics=6, alpha=0.0):
        self.n_topics = n_topics
        self.alpha    = alpha

    def fit(self, parser:NarrativeParser, y=None):
        # Takes NarrativeParser object as argument
        model = NarrativeModel(parser)
        model.n_topics = self.n_topics
        model.alpha = self.alpha
        model.compute_nmf()
        model.get_topic_sort_order()
        self.model_ = model
        self._reorder = model.topic_sort_order   # fitted column order
        return self

    def _apply_order(self, df):
        """Reorder columns to narrative topic order and rename 0..n_topics-1."""
        df = df[self._reorder].copy()
        df.columns = list(range(len(self._reorder)))
        return df

    def transform(self, parser, y=None):
        if parser is self.parser_ref_:
            # Training data: return THETA already computed during fit
            return self._apply_order(self.model_.THETA)
        else:
            # New data: project parser's TFIDF into this model's topic space
            # C = PHI @ TFIDF.T  (topics × chunks)
            PHI    = self._apply_order(self.model_.PHI.T).T   # reordered
            shared = PHI.columns.intersection(parser.TFIDF.columns)
            TFIDF  = parser.TFIDF[shared]

            # Recompute L2-normalised TF-IDF for the new data over shared vocab
            TF     = TFIDF
            DF     = (TF > 0).sum()
            IDF    = np.log2((len(TF) + 1) / (DF + 1) + 1)
            TFIDF_norm = TF * IDF
            l2 = np.sqrt((TFIDF_norm ** 2).sum(1))
            TFIDF_norm = TFIDF_norm.div(l2, axis=0)

            C = PHI[shared] @ TFIDF_norm.T
            C = C / C.sum()   # normalise to proportions
            return C

    def fit_transform(self, parser, y=None):
        # Override to avoid calling transform() before parser_ref_ is set
        self.fit(parser, y)
        self.parser_ref_ = parser   # store reference for transform() guard
        return self._apply_order(self.model_.THETA)