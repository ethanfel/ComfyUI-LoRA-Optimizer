# community_cache='upload_only' Mode Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a third `community_cache` enum value — `upload_only` — that runs the full local sweep (skips the HF download / cache-hit short-circuit) but still uploads the resulting config back to HF.

**Architecture:** The download path at `lora_optimizer.py:8875` is split into two conditional blocks. Content-hash computation runs whenever community_cache ∈ {`upload_and_download`, `upload_only`} (uploads need hashes). The actual `_community_download_caches` call and subsequent cache-HIT replay stay gated on `upload_and_download` only. Both upload guards at 9053 and 9695 accept either mode.

**Tech Stack:** Python, ComfyUI node API, unittest.

**Motivation:** During the combinator rerun to backfill `per_prefix_decisions`, `upload_and_download` short-circuits before the sweep runs (cache HIT → no enrichment). `disabled` avoids the short-circuit but also skips uploads, so the enriched config never reaches HF. `upload_only` is the mode that lets the rerun actually publish backfilled data.

---

## Task 1: Add `upload_only` enum value and route hash computation + uploads

**Files:**
- Modify: `lora_optimizer.py:7593` (`LoRAOptimizerSettings.INPUT_TYPES`)
- Modify: `lora_optimizer.py:7925` (`LoRAAutoTunerSettings.INPUT_TYPES`)
- Modify: `lora_optimizer.py:8875-8895` (download block split)
- Modify: `lora_optimizer.py:9053-9054` (memory-hit upload guard)
- Modify: `lora_optimizer.py:9695-9696` (post-sweep upload guard)
- Modify: `tests/test_lora_optimizer.py` (append tests)

**Step 1: Write the failing tests**

Append to `tests/test_lora_optimizer.py` near the existing AutoTuner settings tests (find `TestLoRAAutoTunerSettings` or similar; otherwise add a new test class at the end):

```python
class TestCommunityCacheUploadOnly(unittest.TestCase):
    """Tests for the community_cache='upload_only' mode."""

    def test_optimizer_settings_enum_includes_upload_only(self):
        inputs = lora_optimizer.LoRAOptimizerSettings.INPUT_TYPES()
        choices = inputs["required"]["community_cache"][0]
        self.assertIn("upload_only", choices)
        self.assertIn("upload_and_download", choices)
        self.assertIn("disabled", choices)

    def test_autotuner_settings_enum_includes_upload_only(self):
        inputs = lora_optimizer.LoRAAutoTunerSettings.INPUT_TYPES()
        choices = inputs["required"]["community_cache"][0]
        self.assertIn("upload_only", choices)
        self.assertIn("upload_and_download", choices)
        self.assertIn("disabled", choices)
```

**Step 2: Run the tests to verify they fail**

Run:
```
pytest tests/test_lora_optimizer.py::TestCommunityCacheUploadOnly -v
```

Expected: 2 failures — `upload_only` not in the enum.

**Step 3: Implement enum changes**

In `lora_optimizer.py:7593`:

```python
                "community_cache": (["disabled", "upload_only", "upload_and_download"], {
                    "default": "disabled",
                    "tooltip": "Community cache: share and reuse LoRA analysis results via Hugging Face.\n"
                               "download_only: anonymously download cached results before analysis — no account needed.\n"
                               "upload_only: run locally and upload results; do NOT replay HF cache hits. Useful for backfilling enriched configs.\n"
                               "upload_and_download: also upload your results after tuning. Requires HF_TOKEN environment variable."
                }),
```

In `lora_optimizer.py:7925`:

```python
                "community_cache": (["disabled", "upload_only", "upload_and_download"], {
                    "default": "disabled",
                    "tooltip": "Community-backed cache on Hugging Face. Download precomputed results and contribute yours back. Requires huggingface-cli login or HF_TOKEN env var."
                }),
```

**Step 4: Split the download block so `upload_only` computes hashes but skips the cache replay**

Replace `lora_optimizer.py:8872-8895` (from `# --- Community cache download ---` down to the `else: content_hashes = {}`) with:

```python
        # --- Community cache: hash computation + optional download ---
        _community_tuner_data = None
        content_hashes = {}
        if community_cache in ("upload_and_download", "upload_only") and not _is_sub_merge:
            logging.info(f"[AutoTuner Community] Mode: {community_cache} — computing content hashes "
                         f"for {len(active_loras)} LoRA(s)...")
            _all_hashed = True
            for _i, _lora in enumerate(active_loras):
                _ch = self._lora_content_hash(_lora)
                if _ch is not None:
                    content_hashes[_i] = _ch
                else:
                    logging.warning(
                        f"[AutoTuner Community] Could not hash '{_lora['name']}', disabling community cache")
                    _all_hashed = False
                    break
            if _all_hashed and community_cache == "upload_and_download":
                _arch_key_for_community, _ = _resolve_arch_preset(
                    architecture_preset, getattr(self, '_detected_arch', None) or 'unknown')
                _community_tuner_data = self._community_download_caches(
                    active_loras, content_hashes, lora_caches, pair_caches,
                    arch_preset=_arch_key_for_community, top_n=top_n)
            elif not _all_hashed:
                content_hashes = {}
```

**Step 5: Broaden the upload guards**

In `lora_optimizer.py:9053-9054`:

```python
                    if (community_cache in ("upload_and_download", "upload_only")
                            and len(content_hashes) == len(active_loras)):
```

In `lora_optimizer.py:9695-9696`:

```python
        if (community_cache in ("upload_and_download", "upload_only") and not _is_sub_merge
                and len(content_hashes) == len(active_loras)):
```

**Step 6: Run the tests to verify they pass**

Run:
```
pytest tests/test_lora_optimizer.py::TestCommunityCacheUploadOnly -v
pytest tests/test_lora_optimizer.py -v
```

Expected: `TestCommunityCacheUploadOnly` passes (2/2). Full file remains green (no other test exercises the download/upload branches directly — behavior is unchanged for the two existing enum values).

**Step 7: Commit**

```bash
git add lora_optimizer.py tests/test_lora_optimizer.py
git commit -m "feat(autotuner): add community_cache='upload_only' mode

Runs the full local sweep and uploads the resulting config to the HF
community cache, but never replays HF cache hits. Intended for backfill
reruns that need to re-publish enriched configs without being
short-circuited by the existing cached (unenriched) entry."
```

---

## Out of scope

- CLI/workflow migration. Default for both settings nodes stays `disabled`; existing saved workflows keep their current behavior.
- `download_only` mode. The enum docs mention it but it doesn't exist in the current enum — keeping scope focused on the backfill use case.
