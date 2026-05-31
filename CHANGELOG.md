## v0.2.0

[v0.1.1...v0.2.0](https://github.com/Jannchie/silva/compare/v0.1.1...v0.2.0)

### :sparkles: Features

- **calibration**: histogram-specify library scores onto the label distribution - By [Jianqi Pan](mailto:jannchie@gmail.com) in [060f79f](https://github.com/Jannchie/silva/commit/060f79f)
- **losses**: add ordinal label smoothing (eps=0.2) - By [Jianqi Pan](mailto:jannchie@gmail.com) in [607bae4](https://github.com/Jannchie/silva/commit/607bae4)
- **losses**: add quadratic-weighted-kappa loss to suppress large-gap errors - By [Jianqi Pan](mailto:jannchie@gmail.com) in [76d1c6d](https://github.com/Jannchie/silva/commit/76d1c6d)
- **manifest**: deterministic hash splits with incremental-update diff - By [Jianqi Pan](mailto:jannchie@gmail.com) in [be658a1](https://github.com/Jannchie/silva/commit/be658a1)
- **scripts**: add pictoria DB scoring and label-review tooling - By [Jianqi Pan](mailto:jannchie@gmail.com) in [c42a46f](https://github.com/Jannchie/silva/commit/c42a46f)
- **silva**: bake calibration into the model, rename SDK facade to SilvaScorer - By [Jianqi Pan](mailto:jannchie@gmail.com) in [5cdacc1](https://github.com/Jannchie/silva/commit/5cdacc1)

### :wrench: Chores

- **scripts**: add label-audit and model diagnostics tooling - By [Jianqi Pan](mailto:jannchie@gmail.com) in [84aaa1e](https://github.com/Jannchie/silva/commit/84aaa1e)

## v0.1.1

[v0.1.0...v0.1.1](https://github.com/Jannchie/silva/compare/v0.1.0...v0.1.1)

### :construction_worker: CI

- force JS actions onto Node 24 (silence Node 20 deprecation) - By [Jianqi Pan](mailto:jannchie@gmail.com) in [caaff2c](https://github.com/Jannchie/silva/commit/caaff2c)
- trigger PyPI publish on version tags (v*) + manual dispatch - By [Jianqi Pan](mailto:jannchie@gmail.com) in [240bfab](https://github.com/Jannchie/silva/commit/240bfab)

## v0.1.0

[v0.1.0...v0.1.0](https://github.com/Jannchie/silva/compare/v0.1.0...v0.1.0)

### :construction_worker: CI

- trigger PyPI publish on version tags (v*) + manual dispatch - By [Jianqi Pan](mailto:jannchie@gmail.com) in [240bfab](https://github.com/Jannchie/silva/commit/240bfab)

## v0.1.0

[31e64909189e13beb81143027f43aea486500e84...v0.1.0](https://github.com/Jannchie/silva/compare/31e64909189e13beb81143027f43aea486500e84...v0.1.0)

### :sparkles: Features

- backbone embedder (patch14/pooler_output) + silva score CLI + [backbone] extra - By [Jianqi Pan](mailto:jannchie@gmail.com) in [ca28abf](https://github.com/Jannchie/silva/commit/ca28abf)
- publish the aesthetic head to the Hugging Face Hub - By [Jianqi Pan](mailto:jannchie@gmail.com) in [a8e8fa3](https://github.com/Jannchie/silva/commit/a8e8fa3)
- stronger regularization for the deep trunk (val 0.727 -> 0.738) - By [Jianqi Pan](mailto:jannchie@gmail.com) in [25cb4de](https://github.com/Jannchie/silva/commit/25cb4de)
- deeper head + soft-Spearman loss + retuned lr (test Spearman 0.716 -> 0.733) - By [Jianqi Pan](mailto:jannchie@gmail.com) in [702f159](https://github.com/Jannchie/silva/commit/702f159)
- MLP head + pairwise ranking loss (beats waifu), wandb tracking - By [Jianqi Pan](mailto:jannchie@gmail.com) in [3406e77](https://github.com/Jannchie/silva/commit/3406e77)
- SILVA v1-minimal — ordinal aesthetic scorer (train + eval) - By [Jianqi Pan](mailto:jannchie@gmail.com) in [26d9774](https://github.com/Jannchie/silva/commit/26d9774)

### :adhesive_bandage: Fixes

- restore ordinal_score model output (revert unsanctioned Task 2 change) - By [Jianqi Pan](mailto:jannchie@gmail.com) in [d29abba](https://github.com/Jannchie/silva/commit/d29abba)
- correct hub publish to patch14 + raw pooled feature; add embedding verifier - By [Jianqi Pan](mailto:jannchie@gmail.com) in [b831b37](https://github.com/Jannchie/silva/commit/b831b37)
- address review findings (dead config, corrupt-image recursion, nan guard, build-system) - By [Jianqi Pan](mailto:jannchie@gmail.com) in [6e6d555](https://github.com/Jannchie/silva/commit/6e6d555)
- stop .gitignore from ignoring silva/data/ source (anchor data/ to root) - By [Jianqi Pan](mailto:jannchie@gmail.com) in [700f7c6](https://github.com/Jannchie/silva/commit/700f7c6)

### :art: Refactors

- single 0-1 score API + AestheticScorer facade; safetensors checkpoints - By [Jianqi Pan](mailto:jannchie@gmail.com) in [6a04008](https://github.com/Jannchie/silva/commit/6a04008)
- drop legacy silva/ tree; export library public API - By [Jianqi Pan](mailto:jannchie@gmail.com) in [801ffa7](https://github.com/Jannchie/silva/commit/801ffa7)
- move model_card to silva_train; repoint scripts to split packages - By [Jianqi Pan](mailto:jannchie@gmail.com) in [ee839b0](https://github.com/Jannchie/silva/commit/ee839b0)
- move training pipeline into silva_train; losses reuse silva.scoring - By [Jianqi Pan](mailto:jannchie@gmail.com) in [a5b85a0](https://github.com/Jannchie/silva/commit/a5b85a0)
- move HubAestheticModel into silva library + offline round-trip test - By [Jianqi Pan](mailto:jannchie@gmail.com) in [651d75c](https://github.com/Jannchie/silva/commit/651d75c)
- move model into silva library + split scoring out of losses - By [Jianqi Pan](mailto:jannchie@gmail.com) in [4468144](https://github.com/Jannchie/silva/commit/4468144)
- embedding-based training contract + ordinal-only loss - By [Jianqi Pan](mailto:jannchie@gmail.com) in [991642a](https://github.com/Jannchie/silva/commit/991642a)
- make parquet manifest the contract; DB is just one source - By [Jianqi Pan](mailto:jannchie@gmail.com) in [962b3f4](https://github.com/Jannchie/silva/commit/962b3f4)

### :memo: Documentation

- rewrite READMEs around the two-package split - By [Jianqi Pan](mailto:jannchie@gmail.com) in [64c398b](https://github.com/Jannchie/silva/commit/64c398b)
- fix hub.py docstring — huggingface-hub is a core dep, no 'hub' extra - By [Jianqi Pan](mailto:jannchie@gmail.com) in [78dcdc1](https://github.com/Jannchie/silva/commit/78dcdc1)
- fix push_to_hub usage — no more 'hub' extra (huggingface-hub is now a core dep) - By [Jianqi Pan](mailto:jannchie@gmail.com) in [6196226](https://github.com/Jannchie/silva/commit/6196226)
- implementation plan for monorepo split + HF publishing - By [Jianqi Pan](mailto:jannchie@gmail.com) in [da68359](https://github.com/Jannchie/silva/commit/da68359)
- simplify the published model card (readme-writing guidelines) - By [Jianqi Pan](mailto:jannchie@gmail.com) in [43db0b4](https://github.com/Jannchie/silva/commit/43db0b4)
- monorepo split + HF publishing design spec - By [Jianqi Pan](mailto:jannchie@gmail.com) in [70d3d8d](https://github.com/Jannchie/silva/commit/70d3d8d)
- record fixed-res vs NaFlex decision (baseline first, A/B later) - By [Jianqi Pan](mailto:jannchie@gmail.com) in [8b14221](https://github.com/Jannchie/silva/commit/8b14221)
- canonical output [0,1] from zero + rescale helper - By [Jianqi Pan](mailto:jannchie@gmail.com) in [6aba230](https://github.com/Jannchie/silva/commit/6aba230)
