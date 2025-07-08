import pandas as pd
import numpy as np
import warnings

# Prevents error thrown by np.array_slit()
warnings.filterwarnings("ignore", category=FutureWarning, message=".*swapaxes.*")

# A future version will expect a TOKEN object 
def chunk_tokens(tokens_df, chunk_id_col='chunk_id', df_str='term_str', n_chunks=10):
    """
    Takes a standard TOKEN table and evenly divides it into n chunks.
    """
    tokens_df[chunk_id_col] = np.concatenate([
        np.full(len(chunk), i) 
        for i, chunk 
        in enumerate(np.array_split(tokens_df, n_chunks))
    ])
    chunks_df = (
        tokens_df.groupby(chunk_id_col)
            .term_str.apply(lambda x: ' '.join(map(str, x))).to_frame('chunk_str')
    )
    chunks_df['n_tokens'] = chunks_df.chunk_str.str.split().str.len()
    
    return chunks_df, tokens_df