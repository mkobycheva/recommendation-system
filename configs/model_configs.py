"""Model hyperparameter configurations."""

SVD_CONFIG = {
    "n_factors": 100,
    "k": 10,
    "batch_size": 512,
}
ALS_CONFIG = {
    "factors": 128,
    "regularization": 0.05,
    "iterations": 20,
    "alpha": 15,
    "k": 10,
    "random_state": 42,
}
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
LSTM_CONFIG = {
    "embedding_dim": 64,
    "hidden_dim": 128,
    "max_len": 10,
    "batch_size": 512,
    "lr": 0.001,
    "weight_decay": 1e-5,
    "dropout": 0.2,
    "epochs": 5,
    "k": 10,
}
SASREC_CONFIG = {}
BERT4REC_CONFIG = {}
ENSEMBLE_CONFIG = {}
