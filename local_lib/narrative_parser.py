import pandas as pd
import numpy as np


class NarrativeParser:
    """Parses a DOC dataframe into TOKEN, VOCAB, CHUNK, CTM, and TFIDF."""

    n_chunks = 60
    n_top_sigs = 500

    def __init__(self, src_id: str, DOC: pd.DataFrame):
        self.src_id = src_id
        self.DOC = DOC.copy()
        self._doc_key = DOC.index.names[0]

    def tokenize(self):
        """DOC → TOKEN: split each doc string into individual tokens."""
        TOKEN = self.DOC.doc_str.str.split(expand=True).stack().to_frame('token_str')
        TOKEN.index.names = self.DOC.index.names + ['token_num']
        TOKEN['term_str'] = (
            TOKEN.token_str
            .str.lower()
            .str.replace(r"[^a-z']", "", regex=True)
        )
        self.TOKEN = TOKEN

    def compute_vocab(self):
        """TOKEN → VOCAB: term frequencies and information content."""
        VOCAB = self.TOKEN.term_str.value_counts().to_frame('n').sort_index()
        VOCAB = VOCAB[VOCAB.index != ""].copy()
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

    def compute_ctm(self):
        """TOKEN → CTM: chunk-term matrix (raw counts)."""
        self.CTM = (
            self.TOKEN
            .groupby(['chunk_num', 'term_str'], observed=True)
            .term_str.count()
            .unstack(fill_value=0)
        )

    def select_sigs(self):
        """Choose SIGS: top terms by distributional entropy across chunks."""
        DP = self.CTM / self.CTM.sum()
        DI = np.log2(1 / DP).replace(np.inf, 0)
        DH = DP * DI
        self.VOCAB['dh'] = DH.sum()
        self.SIGS = self.VOCAB.sort_values('dh').tail(self.n_top_sigs).index

    def compute_tfidf(self):
        """CTM → TFIDF: TF-IDF weighted and L2-normalized."""
        TF = self.CTM[self.SIGS]
        DF = TF[TF > 0].sum()
        IDF = np.log((self.n_chunks + 1) / (DF + 1) + 1)
        TFIDF = TF * IDF
        L2_norm = np.sqrt((TFIDF ** 2).sum(1))
        self.TFIDF = TFIDF.div(L2_norm, axis=0)

    def save(self):
        """Save TOKEN, VOCAB, CHUNK, and TFIDF to CSV."""
        self.TOKEN.to_csv(f"{self.src_id}-TOKEN.csv", index=True)
        self.VOCAB.to_csv(f"{self.src_id}-VOCAB.csv", index=True)
        self.CHUNK.to_csv(f"{self.src_id}-CHUNK-{self.n_chunks}.csv", index=True)
        self.TFIDF.to_csv(f"{self.src_id}-TFIDF-{self.n_chunks}.csv", index=True)

    def run(self):
        self.tokenize()
        self.compute_vocab()
        self.chunk()
        self.compute_ctm()
        self.select_sigs()
        self.compute_tfidf()
        self.save()
