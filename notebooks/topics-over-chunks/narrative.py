import pandas as pd
import numpy as np
import plotly_express as px

from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.preprocessing import normalize

import sys
sys.path.append("../../local_lib/")
from hac2 import HAC
from heatmap import plot_grid, plot_map, CorrelationHeatMap as CHM

class Chunker:

    def __init__(self, docs, factor=75):
        self.docs = docs
        self.factor = factor


    def create_chunk_df(self):

        # Convert list of doc strings in list of tokens
        bigline = (' '.join(self.docs)).split()

        # Computer the number of tokens per chunk
        chunk_size = len(bigline) // self.factor

        # Create list of chunks (segments) 
        chunks = [bigline[i:i + chunk_size] for i in range(0, len(bigline), chunk_size)]

        # Convert list of chunks into data frame of 
        CHUNK = pd.DataFrame(chunks)\
            .apply(lambda x: ' '.join(map(str, x)), axis=1)\
            .to_frame('chunk_str')

        # Create two-level OHCO
        CHUNK['level_1'] = CHUNK.index // 10
        CHUNK['level_2'] = CHUNK.index % 10
        CHUNK = CHUNK.reset_index(drop=True).set_index(['level_1', 'level_2'])
        CHUNK['n_chars'] = CHUNK.chunk_str.str.len()
        self.CHUNK = CHUNK

    def viz_chunks(self):
        fig = px.bar(
            x=[f"{idx[0]}:{idx[1]}" for idx in CHUNK.index.to_list()], 
            y=CHUNK.n_chars, 
            color=CHUNK.reset_index().level_1, 
            color_continuous_scale = px.colors.qualitative.Pastel
        )
        fig.plot()


class Vectorizer:

        def __init__(self, chunk_df, stops=None):
            self.CHUNK = chunk_df
            self.stops = stops

        def create_count_matrix(self):
            count_engine = CountVectorizer(
                lowercase=True,
                analyzer='word',
                token_pattern=r"(?u)\b[a-z'][a-z']+\b",
                max_df=.9,
                min_df=5, 
                stop_words=self.stops,
                ngram_range = (1,2)
            )
            self.CTM = pd.DataFrame(count_engine.fit_transform(self.CHUNK.chunk_str).toarray(), 
                columns=count_engine.get_feature_names_out(), 
                index=self.CHUNK.index)

        def create_tfidf_matrix(self):
            tfidf_engine = TfidfTransformer(norm='l2', use_idf=True)
            self.TFIDF = pd.DataFrame(tfidf_engine.fit_transform(self.CTM).toarray(), columns=CTM.columns, index=CTM.index)


        
        
        