"""Thin HF-transformers wrapper around Qwen2.5-VL for Turing GPUs (no bf16, no flash-attn).

dtype modes:
  fp16       everything fp16 (fast; risk of overflow)
  fp16-vit32 LLM fp16, vision tower fp32 (default; the ViT is the part that overflows)
  fp32       everything fp32 (<= 3B models only on a 24 GB card)
"""
import math

import torch
from PIL import Image


class LastTokenHead(torch.nn.Module):
    """Inference-only lm_head wrapper: transformers 4.52's Qwen2.5-VL computes logits for every
    prompt position at prefill (no logits_to_keep), i.e. batch x ~2000 x 152k fp16 = several GB.
    generate() only uses the last position, so only that one is computed (same results)."""

    def __init__(self, head):
        super().__init__()
        self.head = head
        self.weight = head.weight

    def forward(self, h):
        if h.dim() == 3 and h.shape[1] > 1:
            h = h[:, -1:, :]
        return self.head(h)


def _varlen_sdpa(q, k, v, cu_seqlens):
    """Block-diagonal attention over the segments in cu_seqlens without an N x N mask.
    q, k, v: [seq, heads, dim]. Segments of equal length are batched; SDPA without a mask can use
    the memory-efficient kernel on sm_75 (no FlashAttention-2 there)."""
    import torch.nn.functional as F
    from collections import defaultdict
    cu = cu_seqlens.tolist()
    groups = defaultdict(list)
    for a, b in zip(cu[:-1], cu[1:]):
        if b > a:
            groups[b - a].append(a)
    out = torch.empty_like(q)
    for L, starts in groups.items():
        idx = (torch.tensor(starts, device=q.device)[:, None] + torch.arange(L, device=q.device)[None]).reshape(-1)
        n = len(starts)
        qq, kk, vv = (x[idx].reshape(n, L, x.shape[1], x.shape[2]).transpose(1, 2) for x in (q, k, v))
        o = F.scaled_dot_product_attention(qq, kk, vv, dropout_p=0.0)
        out[idx] = o.transpose(1, 2).reshape(n * L, q.shape[1], q.shape[2])
    return out


def patch_vision_attention():
    """transformers' Qwen2.5-VL SDPA vision attention builds a dense [N, N] mask over ALL patches of
    ALL images in a batch (N = 7296 per 1600x900 image) -> O(N^2) memory and OOM on 24 GB cards.
    Replace it by the mathematically identical per-segment attention (_varlen_sdpa)."""
    from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as M
    if getattr(M.Qwen2_5_VLVisionSdpaAttention, '_olp_patched', False):
        return

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb=None, position_embeddings=None):
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        if position_embeddings is None:
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos, sin = emb.cos(), emb.sin()
        else:
            cos, sin = position_embeddings
        q, k = M.apply_rotary_pos_emb_vision(q, k, cos, sin)
        o = _varlen_sdpa(q, k, v, cu_seqlens)
        return self.proj(o.reshape(seq_length, -1))

    M.Qwen2_5_VLVisionSdpaAttention.forward = forward
    M.Qwen2_5_VLVisionSdpaAttention._olp_patched = True

    # Run the vision tower on at most VIT_CHUNK images at a time (images are independent in the
    # ViT, so this is exact); bounds fp32-ViT activation memory when many prompts are batched.
    # Identical images (same pixels, e.g. the same frame in the prompts of the OpenEMMA-style planner)
    # are encoded once (exact pixel equality, torch.equal, against a cache of the last
    # VIT_CACHE_SIZE images); equivalent to re-encoding up to fp rounding. VIT_CACHE=0 disables it.
    if hasattr(M, 'Qwen2_5_VLModel') and hasattr(M.Qwen2_5_VLModel, 'get_image_features'):
        import os
        cache = []  # (pixels, grid tuple, embeds), most recent last

        def get_image_features(self, pixel_values, image_grid_thw=None):
            chunk = int(os.environ.get('VIT_CHUNK', '2'))
            use_cache = os.environ.get('VIT_CACHE', '1') != '0'
            pixel_values = pixel_values.type(self.visual.dtype)
            sizes = [int(x) for x in image_grid_thw.prod(-1).tolist()]
            starts = [0]
            for n in sizes:
                starts.append(starts[-1] + n)
            grids = [tuple(int(v) for v in g) for g in image_grid_thw.tolist()]
            px = [pixel_values[starts[i]:starts[i + 1]] for i in range(len(sizes))]
            outs = [None] * len(sizes)
            unique, dup = [], {}
            for i in range(len(sizes)):
                if use_cache:
                    for cpx, cg, emb in reversed(cache):
                        if cg == grids[i] and torch.equal(cpx, px[i]):
                            outs[i] = emb
                            break
                    if outs[i] is not None:
                        continue
                    same = [j for j in unique if grids[j] == grids[i] and torch.equal(px[j], px[i])]
                    if same:
                        dup[i] = same[0]
                        continue
                unique.append(i)
            m = self.visual.spatial_merge_size ** 2
            for k in range(0, len(unique), chunk):
                idx = unique[k:k + chunk]
                emb = self.visual(torch.cat([px[i] for i in idx], 0), grid_thw=image_grid_thw[idx])
                off = 0
                for i in idx:
                    n = sizes[i] // m
                    outs[i] = emb[off:off + n]
                    off += n
                    if use_cache:
                        cache.append((px[i].clone(), grids[i], outs[i]))
            for i, j in dup.items():
                outs[i] = outs[j]
            del cache[:-int(os.environ.get('VIT_CACHE_SIZE', '4'))]
            return torch.cat(outs, 0)

        M.Qwen2_5_VLModel.get_image_features = get_image_features


def load_qwen25vl(path, dtype_mode='fp16-vit32', device='cuda:0', attn='sdpa', min_pixels=None, max_pixels=None):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    patch_vision_attention()
    dt = torch.float32 if dtype_mode == 'fp32' else torch.float16
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        path, torch_dtype=dt, attn_implementation=attn, device_map=device)
    if dtype_mode == 'fp16-vit32':
        model.visual.float()
    model.lm_head = LastTokenHead(model.lm_head)
    model.eval()
    proc = load_processor(path, min_pixels, max_pixels)
    return model, proc


def load_processor(path, min_pixels=None, max_pixels=None):
    """AutoProcessor; for checkpoints saved with transformers 4.49.dev (Impromptu-VLA), whose
    preprocessor_config.json names the since-removed `Qwen2_5_VLImageProcessor`, the processor is
    rebuilt from the same config values with Qwen2VLImageProcessor (identical preprocessing)."""
    import json
    import os
    from transformers import AutoProcessor
    kw = {}
    if min_pixels:
        kw['min_pixels'] = min_pixels
    if max_pixels:
        kw['max_pixels'] = max_pixels
    try:
        proc = AutoProcessor.from_pretrained(path, **kw)
    except ValueError as e:
        if 'image processor' not in str(e):
            raise
        from transformers import AutoTokenizer, Qwen2_5_VLProcessor, Qwen2VLImageProcessor, Qwen2VLVideoProcessor
        cfg = json.load(open(os.path.join(path, 'preprocessor_config.json')))
        for k in ('image_processor_type', 'processor_class', 'size'):
            cfg.pop(k, None)
        cfg.update(kw)
        ip = Qwen2VLImageProcessor(**cfg)
        vp = Qwen2VLVideoProcessor(**{k: v for k, v in cfg.items() if k in (
            'min_pixels', 'max_pixels', 'patch_size', 'temporal_patch_size', 'merge_size', 'image_mean', 'image_std')})
        tok = AutoTokenizer.from_pretrained(path)
        ct = None
        if os.path.exists(os.path.join(path, 'chat_template.json')):
            ct = json.load(open(os.path.join(path, 'chat_template.json')))['chat_template']
        proc = Qwen2_5_VLProcessor(image_processor=ip, tokenizer=tok, video_processor=vp, chat_template=ct)
        print('[qwen_vl] rebuilt processor (Qwen2VLImageProcessor) for', path, {k: cfg[k] for k in ('min_pixels', 'max_pixels', 'patch_size', 'merge_size')})
    proc.tokenizer.padding_side = 'left'
    return proc


def visual_dtype(model):
    return next(model.visual.parameters()).dtype


@torch.no_grad()
def generate(model, proc, texts, images, max_new_tokens=128, gen_kwargs=None, videos=None):
    """texts: list of fully formatted chat strings (with <|vision_start|><|image_pad|>... placeholders);
    images: list (per sample) of lists of PIL images (flattened in order). Returns decoded strings."""
    flat = [im for ims in images for im in ims] if images else None
    inputs = proc(text=texts, images=flat or None, videos=videos, padding=True, return_tensors='pt')
    inputs = inputs.to(model.device)
    if 'pixel_values' in inputs:
        inputs['pixel_values'] = inputs['pixel_values'].to(visual_dtype(model))
    if 'pixel_values_videos' in inputs:
        inputs['pixel_values_videos'] = inputs['pixel_values_videos'].to(visual_dtype(model))
    kw = dict(max_new_tokens=max_new_tokens)
    kw.update(gen_kwargs or {})
    out = model.generate(**inputs, **kw)
    out = out[:, inputs['input_ids'].shape[1]:]
    return proc.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=False), inputs


def chat_text(proc, messages):
    return proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def resize_max_pixels(im, max_pixels):
    """LLaMA-Factory style pre-resize (keeps aspect ratio, int truncation)."""
    if im.width * im.height > max_pixels:
        f = math.sqrt(max_pixels / float(im.width * im.height))
        im = im.resize((int(im.width * f), int(im.height * f)))
    return im


def open_rgb(path):
    return Image.open(path).convert('RGB')
