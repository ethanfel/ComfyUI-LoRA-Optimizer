# Architecture-Aware Key Normalization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add architecture detection and per-architecture key normalization to `LoRAOptimizer` so LoRAs trained by different tools get their keys aligned before merging — including Z-Image QKV tensor splitting/re-fusion.

**Architecture:** A `normalize_keys` boolean input gates the feature. When enabled, `_normalize_stack()` auto-detects architecture from LoRA keys, then normalizes each LoRA's state dict to a canonical format. For Z-Image, fused QKV tensors are split into separate Q/K/V for per-component conflict analysis, then re-fused at patch application time.

**Tech Stack:** Python, PyTorch, ComfyUI node API

---

### Task 1: Add `_detect_architecture()` to `_LoRAMergeBase`

**Files:**
- Modify: `lora_optimizer.py:65-94` (after `_get_compute_device`, before `_load_lora`)

**Step 1: Add `import re` at top of file**

The file currently has no `import re`. Add it after `import time` (line 13):

```python
import re
```

Check first — if `import re` already exists, skip this step.

**Step 2: Add `_detect_architecture` static method**

Insert after `_get_compute_device()` (after line 83), before `_load_lora()`:

```python
    @staticmethod
    def _detect_architecture(lora_sd):
        """
        Detect model architecture from LoRA key patterns.
        Returns: 'zimage', 'flux', 'wan', 'sdxl', 'ltx', 'qwen_image', or 'unknown'.
        """
        keys = list(lora_sd.keys())
        keys_str = ' '.join(k.lower() for k in keys)

        # Z-Image Turbo (Lumina2): layers.N with attention patterns
        # Handles: diffusion_model.layers.N, single_transformer_blocks.N (non-FLUX),
        #          lora_unet_layers_N (Musubi Tuner)
        if any('diffusion_model.layers.' in k and ('attention' in k or 'adaln' in k.lower())
               for k in keys):
            return 'zimage'
        if any('lora_unet_layers_' in k and 'attention' in k.lower() for k in keys):
            return 'zimage'
        # single_transformer_blocks WITHOUT transformer. prefix = Z-Image
        if any('single_transformer_blocks' in k and 'transformer.single_transformer_blocks' not in k
               for k in keys):
            return 'zimage'

        # FLUX: double/single blocks in various trainer formats
        if any('transformer.single_transformer_blocks' in k or 'transformer.transformer_blocks' in k
               for k in keys):
            return 'flux'
        if any('transformer_single_transformer_blocks' in k or 'transformer_double_blocks' in k
               for k in keys):
            return 'flux'
        if any('double_blocks' in k or 'single_blocks' in k for k in keys):
            return 'flux'

        # Wan: blocks.N with self_attn/cross_attn/ffn
        if any(('blocks.' in k or 'blocks_' in k) and
               any(x in k for x in ['self_attn', 'cross_attn', 'ffn'])
               for k in keys):
            return 'wan'

        # LTX Video: transformer_blocks with attn1/attn2 and adaln_single
        if any('adaln_single' in k for k in keys):
            return 'ltx'
        if any('transformer_blocks' in k and ('attn1' in k or 'attn2' in k)
               and not any('transformer_blocks' in k2 and 'img_mlp' in k2 for k2 in keys)
               for k in keys):
            # transformer_blocks with attn1/attn2 but not Qwen-Image patterns
            return 'ltx'

        # Qwen-Image: transformer_blocks with img_mlp/txt_mlp/img_mod/txt_mod
        if any('transformer_blocks' in k and
               any(x in k for x in ['img_mlp', 'txt_mlp', 'img_mod', 'txt_mod', 'add_q_proj'])
               for k in keys):
            return 'qwen_image'

        # SDXL: text encoders or UNet block patterns
        if 'lora_te1_' in keys_str or 'lora_te2_' in keys_str:
            return 'sdxl'
        if any('input_blocks' in k or 'output_blocks' in k for k in keys):
            return 'sdxl'
        if any('down_blocks' in k or 'up_blocks' in k for k in keys):
            return 'sdxl'

        return 'unknown'
```

**Step 3: Verify syntax**

Run: `python -c "import ast; ast.parse(open('lora_optimizer.py').read()); print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add lora_optimizer.py
git commit -m "feat: add _detect_architecture() to _LoRAMergeBase"
```

---

### Task 2: Add `_normalize_keys()` for Z-Image

This is the most complex normalizer — handles QKV tensor splitting and output projection remapping.

**Files:**
- Modify: `lora_optimizer.py` (add method to `_LoRAMergeBase`, after `_detect_architecture`)

**Step 1: Add `_normalize_keys_zimage` static method**

Insert after `_detect_architecture()`:

```python
    @staticmethod
    def _normalize_keys_zimage(lora_sd):
        """
        Normalize Z-Image Turbo (Lumina2) LoRA keys to a canonical format.

        Handles:
        1. Split fused QKV (attention.qkv) into separate to_q/to_k/to_v
           for per-component conflict analysis during merge.
        2. Remap attention.out -> attention.to_out.0 (diffusers convention).
        3. Standardize Musubi Tuner format (lora_unet_layers_N_...).
        4. Ensure diffusion_model.layers.N prefix.

        Returns new dict with normalized keys. Original dict is not modified.
        """
        normalized = {}
        processed = set()

        # First pass: standardize prefixes (Musubi Tuner -> canonical)
        prefix_fixed = {}
        for k, v in lora_sd.items():
            new_k = k
            # Musubi Tuner: lora_unet_layers_N_attention_... -> diffusion_model.layers.N.attention...
            if new_k.startswith('lora_unet_'):
                new_k = new_k.replace('lora_unet_', 'diffusion_model.')
                # Convert underscore-separated to dot-separated for known patterns
                new_k = re.sub(r'layers_(\d+)_', r'layers.\1.', new_k)
                new_k = re.sub(r'attention_', 'attention.', new_k)
                new_k = re.sub(r'feed_forward_', 'feed_forward.', new_k)
            # Ensure diffusion_model. prefix
            if new_k.startswith('layers.'):
                new_k = 'diffusion_model.' + new_k
            prefix_fixed[new_k] = v

        # Second pass: split fused QKV and remap output projection
        # Find all layer indices
        layer_pattern = re.compile(r'((?:diffusion_model\.)?layers\.(\d+)\.attention)\.')
        layers_seen = set()
        for k in prefix_fixed:
            m = layer_pattern.search(k)
            if m:
                layers_seen.add((m.group(1), int(m.group(2))))

        for base, layer_idx in layers_seen:
            # --- Split fused QKV ---
            for lora_fmt in [('.lora_A.weight', '.lora_B.weight'),
                             ('.lora_up.weight', '.lora_down.weight'),
                             ('.lora_B.weight', '.lora_A.weight'),
                             ('.lora.up.weight', '.lora.down.weight')]:
                # Try each LoRA format for the fused QKV key
                down_suffix, up_suffix = lora_fmt[0], lora_fmt[1]
                qkv_down_key = f"{base}.qkv{down_suffix}"
                qkv_up_key = f"{base}.qkv{up_suffix}"

                if qkv_down_key in prefix_fixed and qkv_up_key in prefix_fixed:
                    qkv_down = prefix_fixed[qkv_down_key]  # [rank*3, dim] or [rank, dim]
                    qkv_up = prefix_fixed[qkv_up_key]      # [dim*3, rank] or [dim, rank]

                    # Determine which is A (down) and B (up) by shape
                    # lora_A/down: [rank, in_features] or [rank*3, in_features]
                    # lora_B/up: [out_features, rank] or [out_features*3, rank]
                    rank_down = qkv_down.shape[0]
                    rank_up = qkv_up.shape[1] if qkv_up.dim() == 2 else qkv_up.shape[0]

                    # Split down matrix (rank*3 -> 3 x rank, or just chunk into 3)
                    if rank_down % 3 == 0:
                        q_down, k_down, v_down = torch.chunk(qkv_down, 3, dim=0)
                    else:
                        # Can't split evenly — skip
                        break

                    # Split up matrix (out_features*3 -> 3 x out_features)
                    out_dim = qkv_up.shape[0]
                    if out_dim % 3 == 0:
                        q_up, k_up, v_up = torch.chunk(qkv_up, 3, dim=0)
                    else:
                        break

                    for comp, comp_down, comp_up in [('to_q', q_down, q_up),
                                                      ('to_k', k_down, k_up),
                                                      ('to_v', v_down, v_up)]:
                        normalized[f"{base}.{comp}{down_suffix}"] = comp_down
                        normalized[f"{base}.{comp}{up_suffix}"] = comp_up

                    # Copy alpha (same for all three components)
                    alpha_key = f"{base}.qkv.alpha"
                    if alpha_key in prefix_fixed:
                        for comp in ('to_q', 'to_k', 'to_v'):
                            normalized[f"{base}.{comp}.alpha"] = prefix_fixed[alpha_key]
                        processed.add(alpha_key)

                    processed.add(qkv_down_key)
                    processed.add(qkv_up_key)
                    break  # Found QKV format, don't try others

            # --- Remap output projection: attention.out -> attention.to_out.0 ---
            for lora_fmt in [('.lora_A.weight', '.lora_B.weight'),
                             ('.lora_up.weight', '.lora_down.weight'),
                             ('.lora_B.weight', '.lora_A.weight'),
                             ('.lora.up.weight', '.lora.down.weight')]:
                sfx_a, sfx_b = lora_fmt
                out_a = f"{base}.out{sfx_a}"
                out_b = f"{base}.out{sfx_b}"
                if out_a in prefix_fixed and out_b in prefix_fixed:
                    normalized[f"{base}.to_out.0{sfx_a}"] = prefix_fixed[out_a]
                    normalized[f"{base}.to_out.0{sfx_b}"] = prefix_fixed[out_b]
                    processed.add(out_a)
                    processed.add(out_b)

                    out_alpha = f"{base}.out.alpha"
                    if out_alpha in prefix_fixed:
                        normalized[f"{base}.to_out.0.alpha"] = prefix_fixed[out_alpha]
                        processed.add(out_alpha)
                    break

        # Pass through all unprocessed keys
        for k, v in prefix_fixed.items():
            if k not in processed:
                normalized[k] = v

        return normalized
```

**Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('lora_optimizer.py').read()); print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add lora_optimizer.py
git commit -m "feat: add _normalize_keys_zimage() for QKV splitting and key remap"
```

---

### Task 3: Add `_normalize_keys()` for FLUX, Wan, SDXL, LTX, Qwen-Image

**Files:**
- Modify: `lora_optimizer.py` (add methods to `_LoRAMergeBase`, after `_normalize_keys_zimage`)

**Step 1: Add `_normalize_keys_flux` static method**

```python
    @staticmethod
    def _normalize_keys_flux(lora_sd):
        """
        Normalize FLUX LoRA keys from various trainer formats to canonical format.

        Canonical format: diffusion_model.double_blocks.N.* / diffusion_model.single_blocks.N.*

        Handles:
        - AI-Toolkit: transformer.transformer_blocks.N -> double_blocks.N
                      transformer.single_transformer_blocks.N -> single_blocks.N
        - Kohya: lora_transformer_double_blocks_N -> double_blocks.N
                 lora_transformer_single_transformer_blocks_N -> single_blocks.N
        - Standard: double_blocks.N / single_blocks.N (ensure prefix)
        """
        normalized = {}
        for k, v in lora_sd.items():
            new_k = k

            # AI-Toolkit format
            # transformer.single_transformer_blocks.N -> diffusion_model.single_blocks.N
            new_k = re.sub(
                r'^transformer\.single_transformer_blocks\.(\d+)\.',
                r'diffusion_model.single_blocks.\1.', new_k)
            # transformer.transformer_blocks.N -> diffusion_model.double_blocks.N
            new_k = re.sub(
                r'^transformer\.transformer_blocks\.(\d+)\.',
                r'diffusion_model.double_blocks.\1.', new_k)

            # Kohya underscore format
            # lora_transformer_single_transformer_blocks_N_... -> diffusion_model.single_blocks.N....
            m = re.match(r'^lora_transformer_single_transformer_blocks_(\d+)_(.*)', new_k)
            if m:
                block_num = m.group(1)
                rest = m.group(2).replace('_', '.')
                new_k = f"diffusion_model.single_blocks.{block_num}.{rest}"
            m = re.match(r'^lora_transformer_double_blocks_(\d+)_(.*)', new_k)
            if m:
                block_num = m.group(1)
                rest = m.group(2).replace('_', '.')
                new_k = f"diffusion_model.double_blocks.{block_num}.{rest}"

            # Ensure diffusion_model. prefix for standard format
            if new_k.startswith('double_blocks.') or new_k.startswith('single_blocks.'):
                new_k = 'diffusion_model.' + new_k

            # Generic transformer. prefix -> diffusion_model.
            if new_k.startswith('transformer.'):
                new_k = new_k.replace('transformer.', 'diffusion_model.', 1)

            normalized[new_k] = v
        return normalized
```

**Step 2: Add `_normalize_keys_wan` static method**

```python
    @staticmethod
    def _normalize_keys_wan(lora_sd):
        """
        Normalize Wan LoRA keys from various trainer formats to canonical format.

        Canonical format: diffusion_model.blocks.N.{self_attn,cross_attn,ffn}.*

        Handles LyCORIS, diffusers, Fun LoRA, finetrainer formats.
        Also applies RS-LoRA alpha compensation if detected.
        """
        normalized = {}
        for k, v in lora_sd.items():
            new_k = k

            # LyCORIS/aitoolkit format
            if new_k.startswith('lycoris_blocks_'):
                new_k = new_k.replace('lycoris_blocks_', 'blocks.')
                new_k = new_k.replace('_cross_attn_', '.cross_attn.')
                new_k = new_k.replace('_self_attn_', '.self_attn.')
                new_k = new_k.replace('_ffn_net_0_proj', '.ffn.0')
                new_k = new_k.replace('_ffn_net_2', '.ffn.2')
                new_k = new_k.replace('to_out_0', 'o')

            # Diffusers format prefixes
            if new_k.startswith('transformer.'):
                new_k = new_k.replace('transformer.', 'diffusion_model.', 1)
            if new_k.startswith('pipe.dit.'):
                new_k = new_k.replace('pipe.dit.', 'diffusion_model.', 1)
            if new_k.startswith('blocks.'):
                new_k = 'diffusion_model.' + new_k
            if new_k.startswith('vace_blocks.'):
                new_k = 'diffusion_model.' + new_k

            # Common diffusers cleanup
            new_k = new_k.replace('.default.', '.')
            new_k = new_k.replace('.diff_m', '.modulation.diff')
            new_k = new_k.replace('base_model.model.', 'diffusion_model.')

            # Fun LoRA format: lora_unet__blocks_N_...
            if new_k.startswith('lora_unet__'):
                parts = new_k.split('.')
                main_part = parts[0]
                weight_type = '.'.join(parts[1:]) if len(parts) > 1 else None

                if 'blocks_' in main_part:
                    components = main_part[len('lora_unet__'):].split('_')
                    rebuilt = 'diffusion_model'

                    if components[0] == 'blocks':
                        rebuilt += f".blocks.{components[1]}"
                        idx = 2
                        if idx < len(components):
                            if (components[idx] == 'self' and idx + 1 < len(components)
                                    and components[idx + 1] == 'attn'):
                                rebuilt += '.self_attn'
                                idx += 2
                            elif (components[idx] == 'cross' and idx + 1 < len(components)
                                  and components[idx + 1] == 'attn'):
                                rebuilt += '.cross_attn'
                                idx += 2
                            elif components[idx] == 'ffn':
                                rebuilt += '.ffn'
                                idx += 1
                        if idx < len(components):
                            component = components[idx]
                            idx += 1
                            if idx < len(components) and components[idx] == 'img':
                                component += '_img'
                                idx += 1
                            rebuilt += f'.{component}'

                    if weight_type:
                        if weight_type == 'alpha':
                            rebuilt += '.alpha'
                        elif weight_type in ('lora_down.weight', 'lora_down'):
                            rebuilt += '.lora_A.weight'
                        elif weight_type in ('lora_up.weight', 'lora_up'):
                            rebuilt += '.lora_B.weight'
                        else:
                            rebuilt += f'.{weight_type}'
                            if not rebuilt.endswith('.weight'):
                                rebuilt += '.weight'
                    new_k = rebuilt
                else:
                    new_k = main_part.replace('lora_unet__', 'diffusion_model.')
                    new_k = new_k.replace('_', '.')
                    if weight_type:
                        if weight_type == 'alpha':
                            new_k += '.alpha'
                        elif weight_type in ('lora_down.weight', 'lora_down'):
                            new_k += '.lora_A.weight'
                        elif weight_type in ('lora_up.weight', 'lora_up'):
                            new_k += '.lora_B.weight'
                        else:
                            new_k += f'.{weight_type}'
                            if not new_k.endswith('.weight'):
                                new_k += '.weight'

            # Finetrainer format
            new_k = new_k.replace('.attn1.to_q.', '.self_attn.q.')
            new_k = new_k.replace('.attn1.to_k.', '.self_attn.k.')
            new_k = new_k.replace('.attn1.to_v.', '.self_attn.v.')
            new_k = new_k.replace('.attn1.to_out.0.', '.self_attn.o.')
            new_k = new_k.replace('.attn2.to_q.', '.cross_attn.q.')
            new_k = new_k.replace('.attn2.to_k.', '.cross_attn.k.')
            new_k = new_k.replace('.attn2.to_v.', '.cross_attn.v.')
            new_k = new_k.replace('.attn2.to_out.0.', '.cross_attn.o.')

            normalized[new_k] = v

        # RS-LoRA compensation: detect and fix alpha scaling
        rs_marker = 'diffusion_model.blocks.0.cross_attn.k.lora_A.weight'
        if rs_marker in normalized:
            rank = normalized[rs_marker].shape[0]
            import math
            corrected_alpha = torch.tensor(rank * (rank ** 0.5))
            for nk in list(normalized.keys()):
                if nk.endswith('.lora_A.weight'):
                    alpha_key = nk.replace('.lora_A.weight', '.alpha')
                    if alpha_key not in normalized:
                        normalized[alpha_key] = corrected_alpha

        return normalized
```

**Step 3: Add `_normalize_keys_sdxl` static method**

```python
    @staticmethod
    def _normalize_keys_sdxl(lora_sd):
        """
        Normalize SDXL LoRA keys to canonical format.

        Canonical format: lora_unet_* / lora_te1_* / lora_te2_* (Kohya convention).

        Handles diffusers-format keys (down_blocks.N, up_blocks.N, mid_block).
        """
        normalized = {}
        for k, v in lora_sd.items():
            new_k = k

            # Diffusers format: text_encoder.* -> lora_te1_*, text_encoder_2.* -> lora_te2_*
            if new_k.startswith('text_encoder_2.'):
                new_k = 'lora_te2_' + new_k[len('text_encoder_2.'):].replace('.', '_')
            elif new_k.startswith('text_encoder.'):
                new_k = 'lora_te1_' + new_k[len('text_encoder.'):].replace('.', '_')

            # Diffusers UNet: unet.* -> lora_unet_*
            if new_k.startswith('unet.'):
                new_k = 'lora_unet_' + new_k[len('unet.'):].replace('.', '_')

            # base_model.model -> strip prefix
            new_k = new_k.replace('base_model.model.', '')

            normalized[new_k] = v
        return normalized
```

**Step 4: Add `_normalize_keys_ltx` static method**

```python
    @staticmethod
    def _normalize_keys_ltx(lora_sd):
        """
        Normalize LTX Video LoRA keys to canonical format.

        Canonical format: diffusion_model.transformer_blocks.N.attn1/attn2.to_q/to_k/to_v.*

        LTX uses standard separate Q/K/V — only prefix standardization needed.
        """
        normalized = {}
        for k, v in lora_sd.items():
            new_k = k

            # Kohya format: lora_unet_transformer_blocks_N_... -> diffusion_model.transformer_blocks.N...
            if new_k.startswith('lora_unet_'):
                new_k = new_k.replace('lora_unet_', 'diffusion_model.')
                # Convert underscores back to dots for known structural segments
                new_k = re.sub(r'transformer_blocks_(\d+)_', r'transformer_blocks.\1.', new_k)
                new_k = re.sub(r'attn(\d)_', r'attn\1.', new_k)
                new_k = re.sub(r'to_(q|k|v|out)_', r'to_\1.', new_k)
                new_k = re.sub(r'ff_net_(\d+)_', r'ff.net.\1.', new_k)

            # Diffusers format: unet.* -> diffusion_model.*
            if new_k.startswith('unet.'):
                new_k = new_k.replace('unet.', 'diffusion_model.', 1)

            # Ensure diffusion_model. prefix
            if new_k.startswith('transformer_blocks.'):
                new_k = 'diffusion_model.' + new_k

            # base_model.model prefix
            new_k = new_k.replace('base_model.model.', 'diffusion_model.')

            normalized[new_k] = v
        return normalized
```

**Step 5: Add `_normalize_keys_qwen_image` static method**

```python
    @staticmethod
    def _normalize_keys_qwen_image(lora_sd):
        """
        Normalize Qwen-Image LoRA keys to canonical format.

        Canonical format: diffusion_model.transformer_blocks.N.*

        Qwen-Image uses separate Q/K/V with dual-stream (image+text) attention.
        Supports transformer.*, lycoris_*, and direct key formats.
        """
        normalized = {}
        for k, v in lora_sd.items():
            new_k = k

            # LyCORIS format: lycoris_transformer_blocks_N_... -> diffusion_model.transformer_blocks.N...
            if new_k.startswith('lycoris_'):
                new_k = new_k.replace('lycoris_', 'diffusion_model.')
                new_k = re.sub(r'transformer_blocks_(\d+)_', r'transformer_blocks.\1.', new_k)
                # Restore dots for known component names
                for comp in ['to_q', 'to_k', 'to_v', 'add_q_proj', 'add_k_proj', 'add_v_proj',
                             'img_mlp', 'txt_mlp', 'img_mod', 'txt_mod', 'img_norm1', 'img_norm2',
                             'txt_norm1', 'txt_norm2']:
                    new_k = new_k.replace(f'_{comp}_', f'.{comp}.')
                    if new_k.endswith(f'_{comp}'):
                        new_k = new_k[:-len(f'_{comp}')] + f'.{comp}'

            # transformer.* -> diffusion_model.*
            if new_k.startswith('transformer.'):
                new_k = new_k.replace('transformer.', 'diffusion_model.', 1)

            # Ensure diffusion_model. prefix
            if new_k.startswith('transformer_blocks.'):
                new_k = 'diffusion_model.' + new_k

            # base_model.model prefix
            new_k = new_k.replace('base_model.model.', 'diffusion_model.')

            normalized[new_k] = v
        return normalized
```

**Step 6: Add dispatcher `_normalize_keys` method**

```python
    @classmethod
    def _normalize_keys(cls, lora_sd, architecture):
        """
        Dispatch to architecture-specific key normalizer.
        Returns a new dict with normalized keys.
        """
        if architecture == 'zimage':
            return cls._normalize_keys_zimage(lora_sd)
        elif architecture == 'flux':
            return cls._normalize_keys_flux(lora_sd)
        elif architecture == 'wan':
            return cls._normalize_keys_wan(lora_sd)
        elif architecture == 'sdxl':
            return cls._normalize_keys_sdxl(lora_sd)
        elif architecture == 'ltx':
            return cls._normalize_keys_ltx(lora_sd)
        elif architecture == 'qwen_image':
            return cls._normalize_keys_qwen_image(lora_sd)
        return lora_sd  # unknown — pass through unchanged
```

**Step 7: Verify syntax**

Run: `python -c "import ast; ast.parse(open('lora_optimizer.py').read()); print('OK')"`
Expected: `OK`

**Step 8: Commit**

```bash
git add lora_optimizer.py
git commit -m "feat: add key normalizers for FLUX, Wan, SDXL, LTX, Qwen-Image and dispatcher"
```

---

### Task 4: Wire normalization into `_normalize_stack()` and add `normalize_keys` input

**Files:**
- Modify: `lora_optimizer.py` — `LoRAOptimizer.INPUT_TYPES` (~line 538), `_normalize_stack` (~line 463), `optimize_merge` (~line 1272), `_compute_cache_key` (~line 576), `IS_CHANGED` (~line 598)

**Step 1: Add `normalize_keys` to INPUT_TYPES optional inputs**

In `LoRAOptimizer.INPUT_TYPES`, add after the `svd_device` entry (before the closing `}` of optional):

```python
                "normalize_keys": (["disabled", "enabled"], {
                    "default": "disabled",
                    "tooltip": "Auto-detect LoRA architecture and normalize keys before merging. Enable when mixing LoRAs from different trainers (e.g., Kohya + AI-Toolkit) or for Z-Image QKV fusion."
                }),
```

**Step 2: Add `normalize_keys` to `_compute_cache_key` signature and hash**

Modify `_compute_cache_key` to include `normalize_keys` in the hash:

Change the hash update line from:
```python
        h.update(f"|os={output_strength}|csm={clip_strength_multiplier}|as={auto_strength}|om={optimization_mode}|cp={compress_patches}|sd={svd_device}".encode())
```
To:
```python
        h.update(f"|os={output_strength}|csm={clip_strength_multiplier}|as={auto_strength}|om={optimization_mode}|cp={compress_patches}|sd={svd_device}|nk={normalize_keys}".encode())
```

Add `normalize_keys="disabled"` parameter to the `_compute_cache_key` signature.

**Step 3: Add `normalize_keys` to `IS_CHANGED` signature**

Add `normalize_keys="disabled"` parameter and pass it through to `_compute_cache_key`.

**Step 4: Modify `_normalize_stack` to accept and use `normalize_keys`**

Add `normalize_keys="disabled"` parameter to `_normalize_stack`. After building the normalized list (both tuple and dict paths), add before the `return`:

```python
        # Architecture-aware key normalization
        if normalize_keys == "enabled" and len(normalized) > 0:
            first_sd = normalized[0]["lora"]
            arch = self._detect_architecture(first_sd)
            if arch != "unknown":
                logging.info(f"[LoRA Optimizer] Architecture detected: {arch}")
                logging.info(f"[LoRA Optimizer] Normalizing keys for {len(normalized)} LoRAs...")
                for item in normalized:
                    item["lora"] = self._normalize_keys(item["lora"], arch)
                self._detected_arch = arch
            else:
                logging.info("[LoRA Optimizer] Architecture: unknown (no key normalization applied)")
                self._detected_arch = None
        else:
            self._detected_arch = None
```

Note: We store the detected architecture on `self._detected_arch` for use in Z-Image re-fusion later.

**Step 5: Pass `normalize_keys` through `optimize_merge`**

Add `normalize_keys="disabled"` to `optimize_merge` signature. Pass it to:
- `self._normalize_stack(lora_stack, normalize_keys=normalize_keys)`
- `self._compute_cache_key(..., normalize_keys=normalize_keys)`

**Step 6: Verify syntax**

Run: `python -c "import ast; ast.parse(open('lora_optimizer.py').read()); print('OK')"`
Expected: `OK`

**Step 7: Commit**

```bash
git add lora_optimizer.py
git commit -m "feat: wire normalize_keys input through stack normalization and caching"
```

---

### Task 5: Add Z-Image re-fusion at patch application time

When the detected architecture is `zimage`, merged patches for `to_q`/`to_k`/`to_v` must be re-fused to `qkv` before applying to the model, because the model expects fused QKV.

**Files:**
- Modify: `lora_optimizer.py` — add `_refuse_zimage_patches` method, call it in `optimize_merge` before `model.add_patches()`

**Step 1: Add `_refuse_zimage_patches` method to `_LoRAMergeBase`**

Insert after `_normalize_keys`:

```python
    @staticmethod
    def _refuse_zimage_patches(patches):
        """
        Re-fuse split to_q/to_k/to_v patches back into fused QKV patches
        for Z-Image Turbo models. Also remaps to_out.0 -> out.

        Called after merging, before applying patches to the model.
        Returns a new dict with fused patches.
        """
        fused = {}
        qkv_groups = {}  # base -> {comp: (key, patch)}
        out_remap = {}   # base -> (old_key, patch)

        for key, patch in patches.items():
            # Handle both string keys and tuple keys
            if isinstance(key, tuple):
                key_str = key[0]
            else:
                key_str = key

            # Detect to_q/to_k/to_v patterns in the key
            m = re.search(r'(layers\.\d+\.attention)\.to_(q|k|v)(?:\.|$)', key_str)
            if m:
                base = m.group(1)
                comp = m.group(2)
                if base not in qkv_groups:
                    qkv_groups[base] = {}
                qkv_groups[base][comp] = (key, patch)
                continue

            # Detect to_out.0 -> out remap
            m_out = re.search(r'(layers\.\d+\.attention)\.to_out\.0(?:\.|$)', key_str)
            if m_out:
                base_out = m_out.group(1)
                new_key_str = key_str.replace('.to_out.0', '.out')
                if isinstance(key, tuple):
                    new_key = (new_key_str,) + key[1:]
                else:
                    new_key = new_key_str
                fused[new_key] = patch
                continue

            # Not a QKV or out key — pass through
            fused[key] = patch

        # Fuse QKV groups
        for base, comps in qkv_groups.items():
            if len(comps) == 3 and 'q' in comps and 'k' in comps and 'v' in comps:
                # All three components present — fuse
                q_key, q_patch = comps['q']
                k_key, k_patch = comps['k']
                v_key, v_patch = comps['v']

                # Build the fused key name
                if isinstance(q_key, tuple):
                    fused_key_str = re.sub(r'\.to_q(?=\.|$)', '.qkv', q_key[0])
                    fused_key = (fused_key_str,) + q_key[1:]
                else:
                    fused_key_str = re.sub(r'\.to_q(?=\.|$)', '.qkv', q_key)
                    fused_key = fused_key_str

                # Handle different patch formats
                if isinstance(q_patch, tuple) and q_patch[0] == "diff":
                    # Full-rank diff patch: ("diff", (tensor,))
                    q_diff = q_patch[1][0]
                    k_diff = k_patch[1][0]
                    v_diff = v_patch[1][0]
                    fused_diff = torch.cat([q_diff, k_diff, v_diff], dim=0)
                    fused[fused_key] = ("diff", (fused_diff,))
                elif isinstance(q_patch, LoRAAdapter):
                    # Low-rank patch — concatenate A and B matrices
                    q_data = q_patch.lora_data
                    k_data = k_patch.lora_data
                    v_data = v_patch.lora_data
                    # (mat_up, mat_down, alpha, mid, dora_scale, bias)
                    fused_up = torch.cat([q_data[0], k_data[0], v_data[0]], dim=0)
                    fused_down = torch.cat([q_data[1], k_data[1], v_data[1]], dim=0)
                    fused_alpha = q_data[2]  # alpha is the same for all components
                    fused_mid = None  # Mid not expected for attention
                    fused_patch = LoRAAdapter(set(), (fused_up, fused_down, fused_alpha, fused_mid, None, None))
                    fused[fused_key] = fused_patch
                else:
                    # Unknown patch format — pass through unfused
                    for comp_key, comp_patch in [(q_key, q_patch), (k_key, k_patch), (v_key, v_patch)]:
                        fused[comp_key] = comp_patch
            else:
                # Incomplete QKV group — pass through individual components
                for comp, (comp_key, comp_patch) in comps.items():
                    fused[comp_key] = comp_patch

        return fused
```

**Step 2: Call re-fusion in `optimize_merge` before `add_patches`**

In `optimize_merge`, after Pass 2 merge is done and before the "Apply patches" section (around line 1798), add:

```python
        # Re-fuse Z-Image QKV patches if architecture normalization was used
        if getattr(self, '_detected_arch', None) == 'zimage':
            if len(model_patches) > 0:
                model_patches = self._refuse_zimage_patches(model_patches)
                logging.info(f"[LoRA Optimizer] Re-fused Z-Image QKV patches ({len(model_patches)} model patches)")
```

**Step 3: Verify syntax**

Run: `python -c "import ast; ast.parse(open('lora_optimizer.py').read()); print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add lora_optimizer.py
git commit -m "feat: add Z-Image QKV re-fusion at patch application time"
```

---

### Task 6: Add architecture info to report and update report signature

**Files:**
- Modify: `lora_optimizer.py` — `_build_report`, `optimize_merge`

**Step 1: Add architecture info to `_build_report`**

Add `detected_arch=None, normalize_keys="disabled"` parameters to `_build_report`. After the header lines (after line 1119 `"=" * 50`), add:

```python
        # Architecture info (when normalization is enabled)
        if normalize_keys == "enabled" and detected_arch:
            arch_names = {
                'zimage': 'Z-Image Turbo (Lumina2)',
                'flux': 'FLUX',
                'wan': 'Wan 2.1/2.2',
                'sdxl': 'SDXL',
                'ltx': 'LTX Video',
                'qwen_image': 'Qwen-Image',
            }
            lines.append(f"Architecture: {arch_names.get(detected_arch, detected_arch)} (auto-detected)")
            lines.append(f"Key normalization: enabled")
            lines.append("")
```

**Step 2: Pass architecture info from `optimize_merge` to `_build_report`**

In the `_build_report` call in `optimize_merge`, add:

```python
            detected_arch=getattr(self, '_detected_arch', None),
            normalize_keys=normalize_keys,
```

**Step 3: Verify syntax**

Run: `python -c "import ast; ast.parse(open('lora_optimizer.py').read()); print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add lora_optimizer.py
git commit -m "feat: add architecture info to analysis report"
```

---

### Task 7: Final verification

**Step 1: Full syntax check**

Run: `python -c "import ast; ast.parse(open('lora_optimizer.py').read()); print('OK')"`
Expected: `OK`

**Step 2: Verify all node classes are registered**

Run: `python -c "from lora_optimizer import NODE_CLASS_MAPPINGS; print(list(NODE_CLASS_MAPPINGS.keys()))"`
Expected: `['LoRAStack', 'LoRAOptimizer']`

**Step 3: Verify the new methods exist on `_LoRAMergeBase`**

Run:
```python
python -c "
from lora_optimizer import _LoRAMergeBase
for m in ['_detect_architecture', '_normalize_keys', '_normalize_keys_zimage',
          '_normalize_keys_flux', '_normalize_keys_wan', '_normalize_keys_sdxl',
          '_normalize_keys_ltx', '_normalize_keys_qwen_image',
          '_refuse_zimage_patches']:
    assert hasattr(_LoRAMergeBase, m), f'Missing {m}'
print('All methods present')
"
```
Expected: `All methods present`

**Step 4: Verify `normalize_keys` appears in INPUT_TYPES**

Run:
```python
python -c "
from lora_optimizer import LoRAOptimizer
inputs = LoRAOptimizer.INPUT_TYPES()
assert 'normalize_keys' in inputs['optional'], 'normalize_keys not in optional inputs'
print('normalize_keys input present')
"
```
Expected: `normalize_keys input present`

**Step 5: Commit any final cleanup**

```bash
git add lora_optimizer.py
git commit -m "chore: final verification pass"
```
