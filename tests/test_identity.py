import pandas as pd

from fraudradar.data.identity import add_entity_keys, add_graph_features


def _txn(ts, card1, addr1, D1, device=None):
    return {"ts": ts, "day": ts // 86_400, "card1": card1, "card2": 111.0, "card3": 150.0,
            "card4": "visa", "card5": 226.0, "card6": "debit", "addr1": addr1, "D1": D1,
            "DeviceInfo": device, "id_30": None, "id_31": None, "id_33": None}


def test_account_uid_is_stable_across_days_and_separates_cardholders():
    df = pd.DataFrame([
        _txn(0, 1000, 300.0, 10.0),          # card started day -10
        _txn(5 * 86_400, 1000, 300.0, 15.0),  # same card, 5 days later -> same start day
        _txn(5 * 86_400, 1000, 300.0, 3.0),   # same card attributes, different cardholder
        _txn(6 * 86_400, 1000, 300.0, None),  # D1 missing -> unknown account
    ])
    out = add_entity_keys(df)
    assert out.loc[0, "account_uid"] == out.loc[1, "account_uid"]
    assert out.loc[0, "account_uid"] != out.loc[2, "account_uid"]
    assert pd.isna(out.loc[3, "account_uid"])
    assert out["card_uid"].nunique() == 1


def test_graph_links_shared_devices_and_caps_hubs():
    rows = [_txn(i, 1000 + i, 300.0, 0.0, device="ring-dev") for i in range(3)]
    rows += [_txn(10 + i, 2000 + i, 300.0, 0.0, device="Windows") for i in range(5)]
    df = add_graph_features(add_entity_keys(pd.DataFrame(rows)), max_hub_degree=3)

    assert df["cluster_n_accounts"].tolist()[:3] == [1, 2, 3]  # grows as the ring is revealed
    # Hub: only the first 3 accounts are linked; later ones stay singletons.
    assert df["cluster_n_accounts"].tolist()[3:] == [1, 2, 3, 1, 1]
    assert df["device_n_accounts"].tolist()[3:] == [1, 2, 3, 4, 5]
