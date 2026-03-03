import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF, PCA
from hac2 import HAC


class NarrativeModel:
    """Models narrative structure from chunked text using TFIDF, HAC, PCA, and NMF."""

    # HAC settings
    hac_color_thresh = 1

    # PCA settings
    n_pca_comps = 5

    # NMF settings
    n_topics = 6
    nmf_random_state = 42
    n_topic_terms = 5

    def __init__(self, src_id: str, CHUNK: pd.DataFrame, TFIDF: pd.DataFrame):
        self.src_id = src_id
        self.CHUNK = CHUNK.copy()
        self.TFIDF = TFIDF

    def compute_tfidf_sim(self):
        self.TFIDF_SIM = pd.DataFrame(cosine_similarity(self.TFIDF), index=self.TFIDF.index)

    def plot_tfidf_clustermap(self):
        fig = sns.clustermap(self.TFIDF_SIM,
                             method='ward', metric='euclidean',
                             cmap='Spectral', center=0, cbar_pos=None,
                             col_cluster=False, robust=True,
                             xticklabels=True, yticklabels=True, figsize=(20, 20))
        plt.setp(fig.ax_heatmap.get_xticklabels(), rotation=0, ha="right")
        fig.ax_col_dendrogram.set_visible(False)
        plt.title("Clustermap of Document Similarity", fontsize=24, y=1.01)
        plt.savefig(f"{self.src_id}-tfidf-sims-clustermap.svg", bbox_inches='tight')
        plt.savefig(f"{self.src_id}-tfidf-sims-clustermap.png", bbox_inches='tight')

    def cluster(self):
        tfidf_hac = HAC(self.TFIDF_SIM)
        tfidf_hac.color_thresh = self.hac_color_thresh
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
            n_components=self.n_topics, max_iter=5000,
            init='nndsvda', solver='mu', beta_loss='kullback-leibler',
            random_state=self.nmf_random_state
        )
        self.THETA = pd.DataFrame(nmf_engine.fit_transform(self.TFIDF), index=self.TFIDF.index)
        self.PHI = pd.DataFrame(nmf_engine.components_, columns=self.TFIDF.columns)

        self.TOPIC = self.PHI.T.apply(
            lambda x: ', '.join(x.sort_values(ascending=False).head(self.n_topic_terms).index)
        ).T.to_frame('top_terms')
        self.TOPIC.index.name = 'topic_id'
        self.CHUNK[f'top_topic_{self.n_topics}'] = self.THETA.idxmax(1).values
        self.TOPIC['gloss'] = self.PHI.idxmax(1)
        self.TOPIC['label'] = self.TOPIC.apply(lambda x: f"{x.gloss} T{x.name}", axis=1)

    def plot_topics_over_time(self):
        fig, ax = plt.subplots(figsize=(10, 2))
        sns.heatmap(
            self.THETA.T.set_index(self.TOPIC['label']),
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
        self.plot_topics_over_time()
        self.save()
