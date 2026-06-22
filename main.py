"""
Train all three models. Sample sizes below are what I used on a MacBook —
bump max_train_samples if you have a GPU and patience.
"""
from src.pipeline import run

if __name__ == "__main__":
    run(max_train_samples=5000, max_test_samples=1200, top_n_pathologies=15, epochs=2)