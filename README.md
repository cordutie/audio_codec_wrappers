# audio_codec_wrappers

One API for several neural audio codecs (EnCodec 24 kHz, EnCodec 48 kHz, SpectroStream), used to turn
audio into continuous latents `[B, d, T]` and back. The legacy EnCodec and DAC processors (`processors/`) and their
example notebooks are unchanged; see the last section.

## Install

From this folder, with the environment next to the repository (`../venv_audio_codec_wrappers`):

    bash scripts/install_env.sh                       # torch 2.10.0+cu128, pinned set, git dependencies, this package (editable)
    ../venv_audio_codec_wrappers/bin/python scripts/fetch_weights.py --codec all   # SpectroStream weights, sha256 in weights.lock.json
    ../venv_audio_codec_wrappers/bin/python -m pytest -p no:cacheprovider tests -q -rs

`requirements.lock` is the exact `pip freeze` of the environment used for the reports. Its first line installs the
working tree of this repository (`-e .`), so `pip install -r requirements.lock` after cloning installs this checkout.
It still needs the torch index (`--extra-index-url https://download.pytorch.org/whl/cu128`) for the `+cu128` builds.
EnCodec 24k and 48k download their weights through their own packages on first use.

## API

    from audio_codec_wrappers import list_codecs, codec_info, load_codec
    codec = load_codec("encodec24k", device="cuda")        # cached per (name, device)
    enc = codec.encode(audio, sample_rate)                  # Encoded(z [B, d, T], scale or None, num_samples)
    y = codec.decode(enc.z, enc.scale, seed=0)              # [B, C, T*h], sample 0 aligned with the input
    dec = codec.streaming_decoder(batch_size=1)             # causal codecs only
    out = dec.push(z_block); tail = dec.flush()

- Audio is float32 `[B, C, N]` in [-1, 1]; latents are float32 `[B, d, T]`, channel first. `encode` resamples
  (torchaudio) and maps channels (mono to stereo duplicates, stereo to mono is the mean) to the codec's own rate and layout.
- Aliases of the same bound methods: `encode_continuous` and `audio_to_latents` (= `encode`), `latents_to_audio` (= `decode`).
- `streaming_decoder()` is exact: `cat(push outputs..., flush())[..., L:L+T*h]` equals `decode(z)` for every chunking
  of `z`, where `L` is `dec.latency_samples`. Non-causal codecs raise `NotImplementedError`.
- `window_decoder(window_frames)` re-decodes the last `window_frames` latents from a fresh state. It exists only as the
  instrument of `scripts/window_report.py`; it is not a decoding mode.
- `codec_info(name)` is static (no jax, transformers or encodec import).

| name | rate | channels | hop h | latent d | frame rate | quantum q | causal | latency L | scale | stochastic decoder | backend |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `encodec24k` | 24000 | 1 | 320 | 128 | 75 Hz | 1 | yes | 0 | no | no | torch |
| `spectrostream` | 48000 | 2 | 1920 | 256 | 25 Hz | 1 | yes | 1920 samples (1 frame, 40 ms) | no | no | jax |
| `encodec48k` | 48000 | 2 | 320 | 128 | 150 Hz | 1 | no | none | yes (per-item RMS) | no | torch |
| `toy_causal`, `toy_noncausal` | 8000 | 1 | 8 | 16 | 1000 Hz | 1 | yes / no | 0 / none | no / yes | no | torch |

Tools: `scripts/roundtrip_report.py` (S1 fidelity, alignment, streaming errors, calibration of `s1_thresholds.json`) and
`scripts/window_report.py` (window versus stream decoding and the empirical decoder memory).

## Environment and numerical behaviour (recorded, not options)

- **JAX/XLA.** Importing the package sets `XLA_PYTHON_CLIENT_PREALLOCATE=false` and appends `--xla_gpu_autotune_level=0`
  to `XLA_FLAGS` (before jax is imported). Without that flag the first call per input shape took 10 to 80 s and held
  8.5 GiB of GPU memory; with it 0.6 s and 0.3 GiB (slice A2 measurement). Both are in the `env.json` of the reports.
- **JAX matmul precision.** The SpectroStream calls run under `jax.default_matmul_precision("highest")` (full fp32).
  With the default (TF32) one-shot and streaming decoding differed by about 1e-3.
- **EnCodec 24k.** The encoder runs with torch's default TF32 settings, one item at a time. TF32 rounding makes its
  output depend on the batch composition by about 1e-2 relative, which is why items are encoded one by one. The
  decoder runs with TF32 off; with it on, streaming and one-shot decoding differ by 2.5e-3 instead of 2e-6.

## Codec notes

**EnCodec 24k** (`encodec` package, 75 Hz, 128-d, quantiser bypassed, `normalize=False`).
`decode` is the zero-history causal decoder: the official decoder with its convolutions made causal and zero history at
the start, run in one shot. `streaming_decoder()` is the same function carried over blocks (convolution input tails,
transposed-convolution overlap tails, LSTM state) and equals it. `decode_official` is the unchanged official decoder,
which pads the start by reflection. The two differ mainly in the first part of the clip (build report, max abs per 0.1 s: up to 0.37 in the
first 0.1 s, about 1e-2 to 8e-2 at 0.5 to 1 s, and 1e-5 to 5e-2 at the end of the 10 s clips, depending on the clip);
the curve is in `encodec24k_decode_vs_official.png` of the report.

**SpectroStream** (Magenta RT 2, 48 kHz stereo, 25 Hz, 256-d, JAX/flax). The latent is the encoder output; the decoder
is the RVQ-bypass decoder. `latency_samples` is 1920 (one frame, 40 ms) as measured from the sequence-layers output
latency. Weights: the codec-only `encoder/decoder.safetensors` of `google/magenta-realtime-2`.

**EnCodec 48k** (HF `facebook/encodec_48khz`, no chunking, no overlap). The input is divided by its RMS and the decoder
output multiplied back; the RMS is the `scale` returned by `encode`. Equal to the legacy `EncodecProcessor(sr=48000)`.

## Reports

`scripts/roundtrip_report.py --calibrate` measured `s1_thresholds.json` on the calibration set (guitar plus four
10 s train-region microtex clips); it is frozen. `scripts/window_report.py` gives window versus stream decoding for the
causal codecs.

## Legacy: notebooks for the EnCodec and DAC processors

1. Create a virtual environment and `pip install -r requirements.txt`.
2. Open any notebook in `examples/` (`example_dac.ipynb`, `example_encodec.ipynb`) and run the cells.

The legacy methods are `audio_to_latents()`, `latents_to_codes()`, `codes_to_latents()` and `latents_to_audio()` of the
processors in `processors/`.
