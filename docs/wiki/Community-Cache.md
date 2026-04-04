# Community Cache

The community cache lets AutoTuner users share and reuse precomputed merge results via a public Hugging Face dataset. LoRA merge analysis is hardware-agnostic — the same LoRA files always produce the same conflict metrics and optimal config regardless of GPU tier, so results computed on one machine are valid everywhere.

---

## How It Works

When `community_cache=upload_and_download` is set on the AutoTuner node, each run does two things:

**Before analysis:**
- Computes a content hash (SHA256[:16]) for each LoRA file
- Downloads any matching per-LoRA and pairwise conflict metrics from the community dataset
- If a full winning config exists for this LoRA set + architecture, replays it immediately — skipping the entire sweep

**After a full sweep (or when replaying from local memory):**
- Uploads your per-LoRA and pairwise analysis entries
- Uploads the winning config if your score beats the current community score

Results are keyed by file content, not filename — so a LoRA renamed or moved still matches.

---

## Privacy

**LoRA filenames are never uploaded.** Only content hashes are used as keys. The shared data contains:

- Per-prefix conflict stats (cosine similarity, sign conflict ratio, subspace overlap)
- Winning merge configuration (sparsification, merge mode, refinement level, etc.)
- A composite quality score

No file paths, no usernames, no model names.

---

## Setup

**One-time setup:**

```bash
pip install huggingface_hub
huggingface-cli login
```

That's it. The node reads the stored token automatically. No environment variables needed for most users.

**Alternative (headless/server):** set `HF_TOKEN` as an environment variable.

**In the node:** set `community_cache` to `upload_and_download`.

---

## What Gets Cached

| File type | Key | Contents | When uploaded |
|-----------|-----|----------|---------------|
| `lora/{hash}.lora.json` | SHA256[:16] of LoRA file | Per-prefix conflict stats for one LoRA | After full sweep |
| `pair/{ha}_{hb}.pair.json` | Sorted pair of content hashes | Pairwise conflict metrics | After full sweep |
| `config/{hashes}_{arch}.config.json` | All content hashes + arch | Best merge config + score | After full sweep or memory hit (if score beats community) |

---

## Score-Based Replacement

Configs are only uploaded if your local score beats the community score. This means the community cache naturally accumulates the best-known configs over time — users with more thorough sweeps (`top_n=10`) or better hardware contribute higher-quality results.

---

## Cache Hits

**Config hit** — full sweep skipped, winning config replayed directly (~2–5s instead of 30–120s+):
```
[AutoTuner Community] Config HIT — score=0.7842, arch=dit
[AutoTuner Community] COMMUNITY CACHE HIT — replaying config (score=0.7842), skipping full sweep
```

**Partial hit** — lora/pair entries fill in missing analysis data, speeding up Pass 1:
```
[AutoTuner Community] Analysis cache: 2/3 lora hit, 2/3 pair hit
[AutoTuner Community] No config found — will run full sweep
```

**Miss** — no community data for this LoRA set yet:
```
[AutoTuner Community] Analysis cache: 0/3 lora hit, 0/3 pair hit
[AutoTuner Community] No config found — will run full sweep
```

---

## Version Compatibility

All community entries include an `algo_version` field. Entries from a different algorithm version are ignored automatically — no stale results will be used.

---

## Troubleshooting

### Nothing uploads

- Check logs for `[AutoTuner Community]` lines — if absent, `community_cache` may be `disabled`
- Check for `No HF token found` warning → run `huggingface-cli login` or set `HF_TOKEN`
- Check for `Community config score >= local` → your result wasn't better than what's already there

### Upload fails with ImportError

`huggingface_hub` is not installed. Run:
```bash
pip install huggingface_hub
```

### Community cache is slow

Each run makes several HTTP requests (one per LoRA, one per pair, one for config). On a slow connection this adds a few seconds before analysis starts. Network errors are silently ignored and fall back to local computation.

### "Could not hash" warning

The LoRA file couldn't be found or read for content hashing. Community cache is disabled for that run. Check that the LoRA file exists and is readable.
