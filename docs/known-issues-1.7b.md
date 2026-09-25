# Known issue: 1.7B models do not generate usable audio

The 0.6B models work. Both 1.7B variants (`-Base` and `-CustomVoice`, in F16 and
Q8_0) either ramble to the generation cap, stop after ~1.5s, or crash. Measured on
a Tesla V100 with `GGML_VK_VISIBLE_DEVICES` pinned to it and 7GB of free VRAM, so
these are not out-of-memory symptoms.

## Reproductions

Garbage output (1.7B CustomVoice F16, preset voice `ryan`, greedy):

```bash
./build/qwen3-tts-cli -m models --tts-model qwen3-tts-1.7b-customvoice-f16.gguf \
    --speaker ryan -t "The quick brown fox jumps over the lazy dog." \
    --temperature 0 -o out.wav
# 5.4s of audio whose transcript is: "I'm the genius, I'm the genius. I have a lot
# of fun, so I'm pretty much done."  (Whisper, language probability 0.59)
```

Runaway (1.7B Base F16, cloned voice, greedy):

```bash
./build/qwen3-tts-cli -m models --tts-model qwen3-tts-1.7b-base-f16.gguf \
    -r reference.wav -t "The quick brown fox jumps over the lazy dog." \
    --temperature 0 -o out.wav
# 163.8s of audio for 44 characters — it runs to the 2048-frame cap.
```

Early stop on a higher repetition penalty; the same text and model:

| `--repetition-penalty` | 44 chars | 309 chars |
|---|---|---|
| 1.05 | 5.4s (rambling transcript) | 8.9s (truncated) |
| 1.30 | 2.1s | 1.5s |
| 1.50 | 1.5s | 1.5s |

Segment fault with a community 1.7B Base GGUF (`Serveurperso/Qwen3-TTS-GGUF`,
`qwen-talker-1.7b-base-Q8_0.gguf`) when a voice is supplied:

```
#9  qwen3_tts::Qwen3TTS::prepare_icl_prompt_from_samples(...)
#10 qwen3_tts::Qwen3TTS::synthesize_with_voice_samples_unlocked(...)
#11 qwen3_tts::Qwen3TTS::synthesize_with_voice(...)
```

### Status of that crash

Two distinct causes were found and are addressed in this fork:

1. **Metadata key names differ between converters.** The in-repo converter writes
   `qwen3-tts.speaker_encoder.embedding_length`; community files write
   `qwen3-tts.spk_enc.embedding_length`. Reading only the first silently fell back
   to the 0.6B default (`1024`) for a 2048-wide file, and the mismatch aborted
   inside `ggml_reshape_1d`. The encoder now reads both spellings.
2. **Mixed weight types inside the encoder.** That file ships
   `spk_enc.fc.weight` as **F32** while the other thirteen encoder weights are
   F16. GGML's CPU backend aborts when the F32 weight meets F16 activations. The
   loader now warns about the mix and `apply_conv1d` promotes the activations to
   F32 for that layer.

With both fixes the file loads and reaches synthesis instead of dying in the
encoder, but it still aborts, now inside `ggml_compute_forward_im2col`
(`ggml_conv_1d` → im2col → `ggml_abort`) during the encoder graph. That is a
GGML-level limitation for this particular conversion, not something this fork
works around. Re-converting the checkpoint with `scripts/convert_tts_to_gguf.py`
(which keeps the encoder uniform) avoids it entirely — the in-repo conversion of
the same checkpoint runs, it just hits the generation problem below.


## Ruled out

Checked against the HF checkpoints (`Qwen/Qwen3-TTS-12Hz-1.7B-Base` and
`-CustomVoice`) and the GGUF written by `scripts/convert_tts_to_gguf.py`:

- **Code-predictor geometry.** The 1.7B `code_predictor_config` is
  `hidden_size=1024, num_attention_heads=16, num_key_value_heads=8, head_dim=128,
  intermediate_size=3072, num_hidden_layers=5, vocab_size=2048`, and the
  checkpoint's `q_proj` is `(2048, 1024)` / `k_proj` `(1024, 1024)`. The converter
  writes `code_pred.attention.head_count=16`, `head_count_kv=8`,
  `attention.key_length=128`, `feed_forward_length=3072` — consistent.
- **The 1.7B-only MTP projection.** `small_to_mtp_projection.weight` is
  `(1024, 2048)` in the checkpoint and `(2048, 1024)` in the GGUF (ggml order,
  ne0=input), matching what `ggml_mul_mat` needs. The C++ loads it and takes the
  `needs_mtp` path.
- **Speaker-encoder width.** `spk_enc.fc.weight` is `(2048, 3072, 1)` in the 1.7B
  checkpoint, so `qwen3-tts.speaker_encoder.embedding_length=2048` (written by the
  converter) is correct — it is not a 0.6B value leaking through.
- **Tensor inventory.** The 1.7B GGUF has exactly the 0.6B reference tensor set
  plus `code_pred.mtp_proj.{weight,bias}`.
- **VRAM.** Reproduced with ~7GB free; the crashes are not allocation failures
  (`--temperature 0` runs load fine and generate, they just generate nonsense).

## Where to look

The 0.6B reference GGUF (`koboldcpp/tts`) carries **no** `qwen3-tts.code_pred.*`
metadata at all, while converted 1.7B files do. Something else in the 1.7B path
differs from 0.6B and the project's own `scripts/setup_pipeline_models.py` only
ever builds the 0.6B model, so this path looks unvalidated:

1. Dump codebook-0 token ids for a fixed text/preset from the C++ pipeline and
   from the Python reference (`transformers` + the HF checkpoint) and compare — the
   first divergence says whether the talker prefill or the code predictor is wrong.
2. Confirm whether the stop condition is driven by codebook 0 reaching
   `codec_eos_id`; the observed behaviour (rambling to the cap at one penalty,
   stopping after 1.5s at another, garbage text at a third) points at the
   codebook-0 head or the delay-pattern revert rather than at sampling.
3. Check the 1.7B prefill layout: `text_projection` is a two-layer MLP in both
   sizes, but the talker hidden differs (1024 vs 2048) while the code predictor
   hidden does not (1024 vs 1024), so any code path that assumes
   `hidden_size == code_pred_hidden_size` will only be exercised by the 1.7B.

Until this is fixed, pin the 0.6B models, which are stable.
