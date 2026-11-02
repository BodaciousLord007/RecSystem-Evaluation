from dataclasses import dataclass


@dataclass
class Config:
    seed: int = 42
    max_users: int = 500
    max_items: int = 2000
    eval_users: int = 100
    min_ratings: int = 10
    k: int = 10
    neighbors: int = 30
    like_threshold: float = 4.0
    factors: int = 24
    epochs: int = 20

