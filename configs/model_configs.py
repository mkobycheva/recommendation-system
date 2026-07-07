"""Model hyperparameter configurations."""

SVD_CONFIG = {}
ALS_CONFIG = {}
ITEM2VEC_CONFIG = {
    "vector_size": 128,
    "window": 5,
    "sg": 1,
    "min_count": 3,
    "workers": 4,
    "seed": 42,
    "k": 10,
    "batch_size": 5000,
}
LSTM_CONFIG = {}
SASREC_CONFIG = {}
BERT4REC_CONFIG = {}
ENSEMBLE_CONFIG = {}
