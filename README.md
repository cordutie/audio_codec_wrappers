# EnCodec Processor - Complete Audio Processing Pipeline

## Overview

The `EncodecProcessor` class provides a unified interface for EnCodec audio processing with support for both 24kHz and 48kHz models. It implements the complete EnCodec pipeline:

```
audio → embeddings → codes → embeddings → audio
```

This class handles the conversion between different audio representations, supporting both:
- **Continuous path**: Direct embeddings (pre-quantization, AudioBox-style)
- **Quantized path**: Discrete codes (standard EnCodec compression)

## Key Features

- **Unified API**: Same interface for both 24kHz (torchaudio) and 48kHz (HuggingFace) models
- **Complete Pipeline**: Full support for the audio→embeddings→codes→embeddings→audio workflow
- **Automatic Handling**: Audio format conversion, device placement, and model-specific preprocessing
- **Metadata Support**: Proper handling of scales and padding for chunked audio (48kHz model)
- **Flexible Input**: Supports various audio tensor formats and automatic resampling

## Installation

```bash
pip install torch torchaudio transformers encodec librosa
```

## Quick Start

```python
from encodec_processor import EncodecProcessor
import librosa

# Initialize processor (24kHz or 48kHz)
processor = EncodecProcessor(sr=24000)  # or sr=48000

# Load audio
audio, _ = librosa.load("audio.wav", sr=24000, mono=True)

# Complete pipeline test
print("Testing complete EnCodec pipeline...")

# 1. Audio → Continuous embeddings
cont_embeddings = processor.encode_audio_emb(audio)

# 2. Audio → Codes (quantized)
codes, metadata = processor.encode_audio_codes(audio, kbps=6.0)

# 3. Codes → Embeddings (quantized)
quant_embeddings = processor.decode_codes_emb(codes)

# 4. Embeddings → Audio reconstruction
audio_from_cont = processor.decode_emb_audio(cont_embeddings)
audio_from_quant = processor.decode_emb_audio(quant_embeddings, metadata)

# Direct codes → audio (recommended for quantized)
audio_from_codes = processor.decode_codes_audio(codes, metadata)
```

## API Reference

### Core Methods

The processor provides these main methods for the complete pipeline:

#### 1. `encode_audio_emb(audio, sample_rate=None)`
Encode audio to continuous embeddings (pre-quantization).
- **Input**: Audio tensor or numpy array
- **Output**: Continuous embeddings tensor
- **Use case**: AudioBox-style processing, lossless intermediate representation

#### 2. `encode_audio_codes(audio, kbps=6.0, sample_rate=None, include_padding_mask=False)`
Encode audio to discrete quantized codes.
- **Input**: Audio tensor/array, target bandwidth in kbps
- **Output**: Tuple of (codes, metadata)
- **Supported kbps**: 1.5, 3.0, 6.0, 12.0, 24.0
- **Use case**: Standard EnCodec compression

#### 3. `decode_codes_emb(codes)`
Convert discrete codes back to quantized embeddings.
- **Input**: Discrete codes tensor
- **Output**: Quantized embeddings tensor
- **Use case**: Intermediate step in codes→embeddings→audio pipeline

#### 4. `decode_codes_audio(codes, metadata=None)`
Decode codes directly to audio (recommended for quantized path).
- **Input**: Codes tensor and metadata (required for 48kHz model)
- **Output**: Reconstructed audio tensor
- **Use case**: Direct audio reconstruction from codes

#### 5. `decode_emb_audio(embeddings, metadata=None)`
Decode embeddings back to audio.
- **Input**: Embeddings tensor and optional metadata
- **Output**: Reconstructed audio tensor
- **Use case**: Audio reconstruction from any embeddings type

#### 6. `get_model_info()`
Get detailed information about the loaded model.
- **Output**: Dictionary with model specifications

### Pipeline Options

#### Continuous Pipeline (Lossless)
```python
# audio → continuous embeddings → audio
embeddings = processor.encode_audio_emb(audio)
reconstructed = processor.decode_emb_audio(embeddings)
```

#### Quantized Pipeline (Compressed - Recommended)
```python
# audio → codes → audio (direct, preserves timing)
codes, metadata = processor.encode_audio_codes(audio, kbps=6.0)
reconstructed = processor.decode_codes_audio(codes, metadata)
```

#### Full Pipeline (Complete workflow)
```python
# audio → embeddings → codes → embeddings → audio
cont_embeddings = processor.encode_audio_emb(audio)
codes, metadata = processor.encode_audio_codes(audio, kbps=6.0) 
quant_embeddings = processor.decode_codes_emb(codes)
reconstructed = processor.decode_emb_audio(quant_embeddings, metadata)
```
## Model Support

### 24kHz Model (torchaudio/encodec)
- **Library**: `encodec` package
- **Input**: Mono or stereo audio
- **Chunking**: No chunking (processes full audio)
- **Metadata**: Not required for reconstruction

### 48kHz Model (HuggingFace transformers)
- **Library**: `transformers` package  
- **Input**: Stereo audio (automatically converts mono)
- **Chunking**: Processes audio in chunks for long sequences
- **Metadata**: Required for proper reconstruction (scales, padding)

## Test Files

The repository includes comprehensive test notebooks:

### `test_full_pipeline.ipynb`
Demonstrates the complete EnCodec pipeline with both models:
1. **24kHz Model Testing**
   - Audio → Continuous embeddings
   - Audio → Codes (quantized)
   - Codes → Embeddings
   - Embeddings → Audio reconstruction
   - Audio comparison and playback

2. **48kHz Model Testing**
   - Same pipeline as 24kHz with metadata handling
   - Demonstrates chunked processing
   - Shows proper scale and padding application

### `test_encodec_processor_24khz.ipynb` & `test_encodec_processor_48khz.ipynb`
Individual model testing with:
- Multiple bitrate testing (1.5, 3.0, 6.0, 12.0, 24.0 kbps)
- Context dependency analysis
- Quality comparisons

## Technical Details

### Audio Processing Pipeline

```
Input Audio (any format)
    ↓ _prepare_audio_tensor()
Normalized Tensor [batch, channels, time]
    ↓ encode_audio_emb() / encode_audio_codes()
Embeddings / Codes
    ↓ decode_codes_emb() / decode_emb_audio() / decode_codes_audio()
Reconstructed Audio
```

### Model Differences

| Feature | 24kHz Model | 48kHz Model |
|---------|-------------|-------------|
| Library | torchaudio/encodec | HuggingFace transformers |
| Chunking | None | Automatic for long audio |
| Metadata | Not needed | Required (scales, padding) |
| Input Channels | 1 or 2 | 2 (auto-converts mono) |
| Code Shape | `[num_quantizers, batch, time]` | `[chunks, batch, num_quantizers, time]` |

### Key Implementation Details

- **Automatic tensor reshaping** for different input formats
- **Device handling** with automatic GPU detection
- **Sample rate conversion** when input differs from model requirements
- **Metadata management** for proper chunked reconstruction (48kHz)
- **Error handling** for unsupported configurations

## Examples

### Basic Usage

```python
from encodec_processor import EncodecProcessor
import librosa
import torch

# Initialize processor
processor = EncodecProcessor(sr=24000, device='cuda')  # or sr=48000

# Load audio file
audio, _ = librosa.load("audio.wav", sr=24000, mono=True)

# Get model information
info = processor.get_model_info()
print(f"Model: {info['model_type']}")
print(f"Sample rate: {info['sample_rate']} Hz")
```

### Complete Pipeline Example

```python
# Test complete pipeline: audio → embeddings → codes → embeddings → audio
print("Testing complete EnCodec pipeline...")

# Step 1: Audio → Continuous embeddings
cont_embeddings = processor.encode_audio_emb(audio)
print(f"Continuous embeddings shape: {cont_embeddings.shape}")

# Step 2: Audio → Discrete codes
codes, metadata = processor.encode_audio_codes(audio, kbps=6.0)
print(f"Codes shape: {codes.shape}")

# Step 3: Codes → Quantized embeddings
quant_embeddings = processor.decode_codes_emb(codes)
print(f"Quantized embeddings shape: {quant_embeddings.shape}")

# Step 4: Embeddings → Audio reconstruction
audio_from_cont = processor.decode_emb_audio(cont_embeddings)
audio_from_quant = processor.decode_emb_audio(quant_embeddings, metadata)
audio_from_codes = processor.decode_codes_audio(codes, metadata)

print(f"Reconstructed audio shapes:")
print(f"  From continuous: {audio_from_cont.shape}")
print(f"  From quantized embeddings: {audio_from_quant.shape}")
print(f"  Direct from codes: {audio_from_codes.shape}")
```

### Bitrate Comparison

```python
# Test different compression rates
bitrates = [1.5, 3.0, 6.0, 12.0, 24.0]
reconstructions = {}

for kbps in bitrates:
    codes, metadata = processor.encode_audio_codes(audio, kbps=kbps)
    reconstructed = processor.decode_codes_audio(codes, metadata)
    reconstructions[kbps] = reconstructed
    print(f"{kbps} kbps: {codes.shape} codes → {reconstructed.shape} audio")
```

### Working with Both Models

```python
# Compare 24kHz vs 48kHz models
for sr in [24000, 48000]:
    print(f"\n=== Testing {sr}Hz model ===")
    processor = EncodecProcessor(sr=sr)
    
    # Load and resample audio
    audio, _ = librosa.load("audio.wav", sr=sr, mono=True)
    
    # Test pipeline
    codes, metadata = processor.encode_audio_codes(audio, kbps=6.0)
    reconstructed = processor.decode_codes_audio(codes, metadata)
    
    print(f"Input: {audio.shape}")
    print(f"Codes: {codes.shape}")
    print(f"Output: {reconstructed.shape}")
```

## Requirements

```
torch>=1.9.0
torchaudio>=0.9.0
transformers>=4.21.0
encodec>=0.1.1
librosa>=0.8.0
numpy
```

## Notes

- **Metadata importance**: Always use metadata with `decode_emb_audio()` when working with quantized embeddings from the 48kHz model
- **Performance**: The 48kHz model processes audio in chunks, which may result in different output lengths
- **Memory**: Continuous embeddings require more memory than quantized codes
- **Quality**: Direct codes→audio reconstruction (`decode_codes_audio`) is recommended for the quantized pipeline
