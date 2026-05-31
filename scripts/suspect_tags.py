"""What do the head's worst misses have in common? (tag enrichment of systematic errors)

Finds TRAIN images you rated >=4 but the head scores <1.5 (the systematic 'you-high/model-low'
errors), then compares their tag distribution against ALL your >=4 train images to surface
ENRICHED tags — the kinds of work the frozen SigLIP embedding systematically fails to read as
good. High enrichment + visually-good images = a SigLIP blind spot for that style/subject.

    uv run python scripts/suspect_tags.py
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import torch

from silva.models.aesthetic import EmbeddingAestheticModel
from silva.scoring import ordinal_score_from_logits

DB = r"E:/pictoria/server/illustration/images/.pictoria/pictoria.sqlite"
CKPT = "outputs/v1_stage1_head"  # dir -> reads best.safetensors via load_checkpoint


def main() -> None:
    import sqlite3

    from silva_train.checkpoint import load_checkpoint

    state, config, _ = load_checkpoint(CKPT)
    mc = config["model"]
    model = EmbeddingAestheticModel(mc["embedding_dim"], mc.get("dropout", 0.1), mc.get("hidden_dims", []))
    model.load_state_dict(state)
    model.eval()

    df = pd.read_parquet("data/manifest.parquet", columns=["post_id", "personal_score", "split", "embedding"])
    x = torch.tensor(np.stack(df["embedding"].to_numpy()), dtype=torch.float32)
    with torch.no_grad():
        df["pred"] = ordinal_score_from_logits(model(x)["logits"]).float().numpy()

    sus = df[(df.split == "train") & (df.personal_score >= 4) & (df.pred < 1.5)]
    base = df[(df.split == "train") & (df.personal_score >= 4)]
    sus_ids = [int(i) for i in sus.post_id]
    base_ids = [int(i) for i in base.post_id]

    con = sqlite3.connect(DB)

    def tag_counts(ids: list[int]) -> dict[str, int]:
        con.execute("DROP TABLE IF EXISTS _t")
        con.execute("CREATE TEMP TABLE _t(id INTEGER PRIMARY KEY)")
        con.executemany("INSERT INTO _t VALUES(?)", [(i,) for i in ids])
        q = "SELECT pt.tag_name, COUNT(*) FROM post_has_tag pt JOIN _t ON _t.id=pt.post_id GROUP BY pt.tag_name"
        return {n: c for n, c in con.execute(q)}

    sc, bc = tag_counts(sus_ids), tag_counts(base_ids)
    ns, nb = len(sus_ids), len(base_ids)

    rows = []
    for tag, c in sc.items():
        if c < 5:
            continue
        sf = c / ns
        bf = bc.get(tag, 0) / nb
        rows.append((tag, c, sf, bf, sf / bf if bf > 0 else float("inf")))
    rows.sort(key=lambda r: -r[4])

    print(f"suspects (train, you>=4, model<1.5) = {ns}   base (all train you>=4) = {nb}\n")
    print(f"{'tag':<30}{'n_sus':>6}{'sus%':>7}{'base%':>7}{'enrich':>8}")
    print("-" * 60)
    for tag, c, sf, bf, enr in rows[:30]:
        print(f"{tag:<30}{c:>6}{sf * 100:>6.1f}{bf * 100:>7.1f}{enr:>8.1f}")

    src = Counter(r for (r,) in con.execute(f"SELECT source FROM posts WHERE id IN ({','.join('?' * ns)})", sus_ids))
    rat = Counter(r for (r,) in con.execute(f"SELECT rating FROM posts WHERE id IN ({','.join('?' * ns)})", sus_ids))
    print(f"\nsource: {src.most_common(8)}")
    print(f"rating: {rat.most_common()}")
    con.close()


if __name__ == "__main__":
    main()
