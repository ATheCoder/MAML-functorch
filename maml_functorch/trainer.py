from dataclasses import dataclass

@dataclass(frozen=True)
class HyperParameters:
    epochs: int
    alpha: float
    beta: float
    batch_size: int
    num_meta_learn_loop: int
    second_order: bool
    shots: int
    ways: int
    embedding_feats: int
    query_size: int