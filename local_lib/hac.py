import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from pandas import DataFrame as df

class HAC:
    """
    Takes an arbitrary vector space and represents it 
    as a hierarhical agglomerative cluster tree. 
    The number of observations should be sufficiently 
    small to allow being plotted.

    Attributes:
        w (int): The width of the figure in inches.
        label_size (int): The font size of the labels in points.
        orientation (str): The orientation of the figure; 'top', 'bottom', 'left', 'right'. Defaults to 'left'.
        dist_measure (str): The distance measure to use; braycurtis, canberra, chebyshev, cityblock, correlation, 
            cosine, dice, euclidean, hamming, jaccard, jensenshannon, kulsinski, kulczynski1, mahalanobis, matching, 
            minkowski, rogerstanimoto, russellrao, seuclidean, sokalmichener, sokalsneath, sqeuclidean, yule. 
            Defaults to euclidean.
        linkage_method (str): The linkage method to use; single, complete, average, weighted, centroid, median, ward
            Defaults to ward.
        norm_type (str): The vector normalization type; l1, l2, max. Defaults to l2.
        color_thresh (float): The threshhold at which to apply coloring in the dendropgram. Defaults to None.
    """

    w:int = 10
    label_size:int = 14
    orientation:str = 'left'
    dist_metric:str = 'euclidean'
    linkage_method:str = 'ward' 
    norm_type:str = 'l2' 
    color_thresh:float = 0
    show_thresh_line:bool = True
    
    def __init__(self, X, labels=None):
        self.X = X
        self.h = X.shape[0]
        if labels:
            self.labels = labels            
        else:
            self.labels = X.index.tolist()

    def get_sims(self):
        self.SIMS = pdist(normalize(self.X, norm=self.norm_type), metric=self.dist_metric)

    def get_tree(self):
        self.TREE = sch.linkage(self.SIMS, method=self.linkage_method)        
        
    def plot_tree(self):
        
        if not self.color_thresh:
            self.color_thresh = df(self.TREE)[2].median()

        fig, ax = plt.subplots(figsize=(self.w, self.h / 3))

        sch.dendrogram(self.TREE,
                    labels=self.labels,
                    orientation=self.orientation,
                    count_sort=False,
                    distance_sort=True,
                    above_threshold_color='.75',
                    color_threshold=self.color_thresh,
                    ax=ax)
        
        ax.tick_params(axis='both', which='major', labelsize=self.label_size)

        # Remove all spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # Add vertical grid lines
        ax.xaxis.grid(True, linestyle='-', alpha=0.9, linewidth=0.5)
        ax.set_axisbelow(True)  # Place grid lines behind the dendrogram

        # Add vertical line at color threshold
        if self.show_thresh_line:
            ax.axvline(x=self.color_thresh, color='red', linestyle='--')
        
        self.fig = fig
        self.ax = ax

    def get_cluster_labels(self):
        self.CLUSTER_LABELS = sch.fcluster(self.TREE, t=self.color_thresh, criterion='distance')
        
    def plot(self):
        self.get_sims()
        self.get_tree()
        self.plot_tree()