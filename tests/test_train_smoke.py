from fraudradar.config import TrainConfig
from fraudradar.data import synthetic
from fraudradar.data.loader import load_raw
from fraudradar.models.train import train


def test_end_to_end_on_synthetic(tmp_path):
    synthetic.write(tmp_path / "raw", n_accounts=800, n_days=90, seed=3)
    df = load_raw(tmp_path / "raw")
    tcfg = TrainConfig(num_boost_round=50, early_stopping_rounds=10)

    raw = train(df, "raw", tcfg=tcfg)
    pit = train(df, "pit", tcfg=tcfg, out_dir=tmp_path / "out")

    assert (tmp_path / "out" / "model.txt").exists()
    assert pit["test"]["roc_auc"] > 0.7
    # Entity history should help on data where fraud is account-level.
    assert pit["test"]["roc_auc"] > raw["test"]["roc_auc"]
