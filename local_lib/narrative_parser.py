import pandas as pd
import numpy as np

class NarrativeParser:
    """Parses a DOC dataframe into TOKEN, VOCAB, CHUNK, CTM, and TFIDF."""

    def __init__(self, src_id: str, DOC: pd.DataFrame, n_chunks=60, no_above=0.4):
        self.src_id = src_id
        self.DOC = DOC.copy()
        self._doc_key = DOC.index.names[0]
        self.n_chunks = 60
        # n_top_sigs = 500
        # n_sw = 20
        self.no_above = .4 # max DF threshold 

    def tokenize(self):
        """DOC → TOKEN: split each doc string into individual tokens."""
        TOKEN = self.DOC.doc_str.str.split(expand=True).stack().to_frame('token_str')
        TOKEN.index.names = self.DOC.index.names + ['token_num']
        TOKEN['term_str'] = (
            TOKEN.token_str
            .str.lower()
            .str.replace(r"[^a-z']", "", regex=True)
        )
        TOKEN = TOKEN[TOKEN.term_str != ""].copy()
        self.TOKEN = TOKEN.dropna() # Why did I have to add the .dropna()???

    def compute_vocab(self):
        """TOKEN → VOCAB: term frequencies and information content."""
        VOCAB = self.TOKEN.term_str.value_counts().to_frame('n').sort_index()
        VOCAB['p'] = VOCAB.n / VOCAB.n.sum()
        VOCAB['i'] = np.log2(1 / VOCAB.p)
        VOCAB['h'] = VOCAB.p * VOCAB.i
        self.VOCAB = VOCAB

    def chunk(self):
        """TOKEN → CHUNK: divide tokens into equal-sized narrative chunks."""
        self.TOKEN['chunk_num'] = pd.cut(
            self.TOKEN.reset_index().index,
            self.n_chunks,
            labels=list(range(self.n_chunks))
        )
        CHUNK = (
            self.TOKEN
            .groupby('chunk_num', observed=True)
            .term_str.apply(lambda x: ' '.join(x))
            .to_frame('chunk_str')
        )
        CHUNK['doc_id'] = (
            self.TOKEN
            .join(CHUNK, on='chunk_num')
            .reset_index()
            .value_counts(['chunk_num', self._doc_key])
            .sort_index()
            .reset_index()
            .groupby('chunk_num', observed=True)[self._doc_key]
            .apply(lambda x: " ".join(map(str, x)))
        )
        self.CHUNK = CHUNK

    def overlap_chunk(self, text, chunk_size=150, overlap=20):
        """Split text into overlapping word-level chunks."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if len(chunk.split()) >= 50:  # drop tiny tail chunks
                chunks.append(chunk)

    def compute_ctm(self):
        """TOKEN → CTM: chunk-term matrix (raw counts)."""
        self.CTM = (
            self.TOKEN
            .groupby(['chunk_num', 'term_str'], observed=True)
            .term_str.count()
            .unstack(fill_value=0)
        )
        self.VOCAB['df'] = (self.CTM > 0).sum()
        self.VOCAB['idf'] = np.log2(self.n_chunks/self.VOCAB.df)

    def select_sigs(self):
        """
        Choose SIGS: top terms by distributional entropy across chunks.
        """
        max_df = round(self.no_above * self.n_chunks)
        non_sw = self.VOCAB[self.VOCAB.df <= max_df].index
        self.SIGS = self.VOCAB.loc[non_sw].index #.sort_values('dh').index

    def l2_norm(self, X):
        return np.sqrt((X ** 2).sum(1))

    def compute_tfidf(self):
        """
        CTM → TFIDF: TF-IDF weighted and L2-normalized.
        """
        TF = self.CTM[self.SIGS]
        DF = (TF > 0).sum()
        IDF = np.log2((self.n_chunks + 1) / (DF + 1) + 1)
        TFIDF = TF * IDF        
        self.TFIDF = TFIDF.div(self.l2_norm(TFIDF), axis=0)

    def run(self):
        self.tokenize()
        self.compute_vocab()
        self.chunk()
        self.compute_ctm()
        self.select_sigs()
        self.compute_tfidf()
        self.save()

    def save(self):
        """Save TOKEN, VOCAB, CHUNK, and TFIDF to CSV."""
        self.TOKEN.to_csv(f"{self.src_id}-TOKEN.csv", index=True)
        self.VOCAB.to_csv(f"{self.src_id}-VOCAB.csv", index=True)
        self.CHUNK.to_csv(f"{self.src_id}-CHUNK-{self.n_chunks}.csv", index=True)
        self.TFIDF.to_csv(f"{self.src_id}-TFIDF-{self.n_chunks}.csv", index=True)


from sklearn.base  import BaseEstimator, TransformerMixin

class NarrativeParserStep(BaseEstimator, TransformerMixin):
    """
    sklearn transformer wrapping NarrativeParser.

    fit(DOC)       → runs the full parse pipeline; stores fitted parser.
    transform(DOC) → returns the fitted parser object so the next step
                     has access to TOKEN, CHUNK, TFIDF, and VOCAB.

    Passing the parser object (rather than just TFIDF) is intentional:
    NarrativeModel needs CHUNK as well as TFIDF, and TestModel needs TOKEN.
    """

    def __init__(self, src_id, n_chunks=60, no_above=0.4):
        self.src_id = src_id
        self.n_chunks = n_chunks
        self.no_above = no_above

    def fit(self, DOC:pd.DataFrame, y=None):

        # Initialize and run parser
        parser = NarrativeParser(self.src_id, DOC)
        parser.n_chunks = self.n_chunks
        parser.no_above = self.no_above
        parser.run()                     # Check if this does more than you want!
        # parser.tokenize()              # Else replace with individual method calls
        # parser.compute_vocab()
        # parser.chunk()
        # parser.compute_ctm()
        # parser.select_sigs()
        # parser.compute_tfidf()

        self.parser_ = parser
        return self

    def transform(self, DOC, y=None):
        # Return the fitted parser; NarrativeModelStep unpacks what it needs.
        return self.parser_