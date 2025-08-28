"""
EnCodec Audio Processor Class

A simple wrapper class for EnCodec models (24kHz and 48kHz) that provides easy access to:
- Continuous embeddings (pre-quantization)
- Quantized embeddings 
- Audio reconstruction from both types

This class handles the conversion between audio tensors and EnCodec representations,
supporting both the continuous path (used by AudioBox) and quantized path (standard EnCodec).
"""

import torch
import torchaudio
from encodec import EncodecModel
from transformers import EncodecModel as HFEncodecModel, AutoProcessor
import numpy as np
from typing import Optional, Tuple, Union


class EncodecProcessor:
    """
    A simple processor for EnCodec models with support for both continuous and quantized embeddings.
    
    This class provides an easy-to-use interface for:
    - Converting audio to continuous embeddings (AudioBox-style)
    - Converting audio to quantized embeddings (standard EnCodec)
    - Reconstructing audio from both embedding types
    - Support for both 24kHz and 48kHz models
    """
    
    def __init__(self, sr: int = 24000, device: Optional[str] = None):
        """
        Initialize the EnCodec processor.
        
        Args:
            sr: Sample rate (24000 for 24kHz model, 48000 for 48kHz model)
            device: Device to run the model on ('cuda', 'cpu', or None for auto-detect)
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.target_sr = sr
        
        print(f"Initializing EnCodec processor on device: {self.device}")
        
        if sr == 24000:
            # Load EnCodec 24kHz model (torchaudio version)
            self.model = EncodecModel.encodec_model_24khz()
            self.model.eval()
            self.model = self.model.to(self.device)
            self.sample_rate = self.model.sample_rate  # 24000 Hz
            self.frame_rate = self.model.frame_rate    # ~75 Hz
            self.model_type = "24kHz"
            self.is_hf_model = False
            print(f"✓ EnCodec 24kHz model loaded (torchaudio)")
            
        elif sr == 48000:
            # Load EnCodec 48kHz model (HuggingFace Transformers version)
            self.model = HFEncodecModel.from_pretrained("facebook/encodec_48khz")
            self.processor = AutoProcessor.from_pretrained("facebook/encodec_48khz")
            self.model.eval()
            self.model = self.model.to(self.device)
            self.sample_rate = 48000
            self.frame_rate = 75  # Approximate frame rate for 48kHz model
            self.model_type = "48kHz"
            self.is_hf_model = True
            print(f"✓ EnCodec 48kHz model loaded (HuggingFace)")
            
        else:
            raise ValueError(f"Unsupported sample rate: {sr}. Supported rates: 24000, 48000")
        
        print(f"✓ Sample rate: {self.sample_rate} Hz")
        print(f"✓ Frame rate: {self.frame_rate} Hz")
    
    def _prepare_audio_tensor(self, audio: Union[torch.Tensor, np.ndarray], sample_rate: Optional[int] = None) -> torch.Tensor:
        """
        Prepare audio tensor for EnCodec processing.
        
        Args:
            audio: Audio data as tensor or numpy array
            sample_rate: Sample rate of input audio (will resample if needed)
            
        Returns:
            Audio tensor of shape [batch, channels, time] ready for EnCodec
        """
        # Convert to tensor if needed
        if not isinstance(audio, torch.Tensor):
            audio = torch.tensor(audio, dtype=torch.float32)
        
        # Ensure correct shape based on model type
        if self.is_hf_model:
            # HuggingFace 48kHz model expects stereo: [channels, time] -> [2, time]
            if audio.dim() == 1:
                audio = torch.stack([audio, audio])  # [time] -> [2, time] (stereo)
            elif audio.dim() == 2:
                if audio.shape[0] == 1:
                    # [1, time] -> [2, time] (duplicate to stereo)
                    audio = torch.cat([audio, audio], dim=0)
                elif audio.shape[1] == 1:
                    # [time, 1] -> [2, time]
                    audio = audio.squeeze(1)
                    audio = torch.stack([audio, audio])
                elif audio.shape[0] > audio.shape[1]:
                    # Likely [time, channels] -> transpose to [channels, time]
                    audio = audio.T
                    if audio.shape[0] == 1:
                        audio = torch.cat([audio, audio], dim=0)
            elif audio.dim() == 3:
                # [batch, channels, time] -> [channels, time] (take first batch)
                audio = audio[0]
                if audio.shape[0] == 1:
                    audio = torch.cat([audio, audio], dim=0)
        else:
            # Torchaudio 24kHz model expects: [batch, channels, time]
            if audio.dim() == 1:
                audio = audio.unsqueeze(0).unsqueeze(0)  # [time] -> [1, 1, time]
            elif audio.dim() == 2:
                if audio.shape[0] > audio.shape[1]:
                    # Likely [time, channels] -> transpose to [channels, time]
                    audio = audio.T.unsqueeze(0)  # -> [1, channels, time]
                else:
                    # Likely [channels, time] -> add batch dimension
                    audio = audio.unsqueeze(0)  # -> [1, channels, time]
        
        # Move to device
        audio = audio.to(self.device)
        
        # Resample if needed
        if sample_rate is not None and sample_rate != self.sample_rate:
            print(f"Resampling from {sample_rate}Hz to {self.sample_rate}Hz")
            if self.is_hf_model:
                # For HF model, resample each channel separately
                resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate).to(self.device)
                audio = resampler(audio)
            else:
                resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate).to(self.device)
                audio = resampler(audio)
        
        return audio
    
    def encode_audio_emb(self, audio: Union[torch.Tensor, np.ndarray], sample_rate: Optional[int] = None) -> torch.Tensor:
        """
        Encode audio to continuous embeddings (pre-quantization).
        This is what AudioBox uses for training and generation.
        
        Args:
            audio: Audio tensor of shape [time] or [channels, time] or [batch, channels, time]
            sample_rate: Sample rate of input audio (will resample if needed)
            
        Returns:
            Continuous embeddings tensor of shape [batch, channels, time_frames]
        """
        with torch.no_grad():
            audio_tensor = self._prepare_audio_tensor(audio, sample_rate)
            
            if self.is_hf_model:
                # HuggingFace 48kHz model - needs CPU tensor for processor
                audio_cpu = audio_tensor.cpu().numpy()
                inputs = self.processor(raw_audio=audio_cpu, sampling_rate=self.sample_rate, return_tensors="pt")
                input_values = inputs["input_values"].to(self.device)
                
                # Apply normalization if the model uses it
                if self.model.config.normalize:
                    mono = torch.sum(input_values, 1, keepdim=True) / input_values.shape[1]
                    scale = mono.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-8
                    input_values = input_values / scale
                
                # Extract continuous embeddings from encoder (BEFORE quantization!)
                continuous_embeddings = self.model.encoder(input_values)
            else:
                # Torchaudio 24kHz model
                # Extract continuous embeddings from encoder (BEFORE quantization!)
                continuous_embeddings = self.model.encoder(audio_tensor)
            
            return continuous_embeddings
    
    def encode_audio_codes(self, audio: Union[torch.Tensor, np.ndarray], 
                     kbps: float = 6.0, sample_rate: Optional[int] = None) -> torch.Tensor:
        """
        Encode audio to quantized codes (standard EnCodec with discrete codes).
        
        Args:
            audio: Audio tensor of shape [time] or [channels, time] or [batch, channels, time]
            kbps: Target bandwidth in kbps. Supported: 1.5, 3.0, 6.0, 12.0, 24.0
            sample_rate: Sample rate of input audio (will resample if needed)
            
        Returns:
            Discrete codes tensor of shape [batch, num_quantizers, time_frames]
        """
        with torch.no_grad():
            audio_tensor = self._prepare_audio_tensor(audio, sample_rate)
            
            if self.is_hf_model:
                # HuggingFace 48kHz model - needs CPU tensor for processor
                audio_cpu = audio_tensor.cpu().numpy()
                inputs = self.processor(raw_audio=audio_cpu, sampling_rate=self.sample_rate, return_tensors="pt")
                input_values = inputs["input_values"].to(self.device)
                padding_mask = inputs["padding_mask"].to(self.device)
                
                # Set target bandwidth
                self.model.config.target_bandwidths = [kbps]
                
                # Standard encode method - includes quantization
                encoded = self.model.encode(input_values, padding_mask)
                codes = encoded.audio_codes

                # Return codes with metadata needed for proper reconstruction
                metadata = {
                    'audio_scales': encoded.audio_scales,
                    'padding_mask': padding_mask
                }

                return codes, metadata
            else:
                # Torchaudio 24kHz model
                # Set target bandwidth
                self.model.set_target_bandwidth(kbps)
                
                # Get continuous embeddings first
                continuous_embeddings = self.model.encoder(audio_tensor)
                
                # Quantize the embeddings
                codes = self.model.quantizer.encode(continuous_embeddings, self.frame_rate, kbps)
                return codes, None
        
    def decode_codes_emb(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Decode discrete audio codes back to quantized embeddings.
        
        Args:
            codes: Discrete codes tensor 
            
        Returns:
            Quantized embeddings tensor of shape [batch, channels, time_frames]
        """
        with torch.no_grad():
            if self.is_hf_model:
                # HuggingFace 48kHz model
                # The codes come as [chunks, batch, num_quantizers, time_frames] for chunked encoding
                # We need to process all chunks and concatenate the results
                
                if codes.dim() == 4:
                    # codes shape: [chunks, batch, num_quantizers, time_frames]
                    chunks, batch_size, num_quantizers, time_frames = codes.shape
                    
                    # Process each chunk separately
                    chunk_embeddings = []
                    for chunk_idx in range(chunks):
                        chunk_codes = codes[chunk_idx]  # [batch, num_quantizers, time_frames]
                        
                        # Transpose to get [num_quantizers, batch, time_frames] which is what quantizer expects
                        chunk_codes = chunk_codes.transpose(0, 1)  # [num_quantizers, batch, time_frames]
                        
                        # Decode this chunk
                        chunk_emb = self.model.quantizer.decode(chunk_codes)  # [batch, channels, time_frames]
                        chunk_embeddings.append(chunk_emb)
                    
                    # Concatenate all chunks along the time dimension
                    quantized_embeddings = torch.cat(chunk_embeddings, dim=2)  # [batch, channels, total_time]
                    
                # elif codes.dim() == 3:
                #     # Single chunk case: [num_quantizers, batch, time_frames]
                #     quantized_embeddings = self.model.quantizer.decode(codes)
                    
                else:
                    raise ValueError(f"Unexpected codes dimensions: {codes.shape}. Should be [chunks, batch, num_quantizers, time_frames]")

            else:
                # Torchaudio 24kHz model
                # codes are in [num_quantizers, batch, time_frames] format already
                quantized_embeddings = self.model.quantizer.decode(codes)
                
            return quantized_embeddings

    def decode_codes_audio(self, codes: torch.Tensor, metadata: Optional[dict] = None) -> torch.Tensor:
        """
        Decode discrete audio codes directly to audio using the complete quantized pipeline.
        This is the proper way to reconstruct audio from quantized codes.
        
        Args:
            codes: Discrete codes tensor 
            metadata: For HF model, dict with 'audio_scales' and 'padding_mask'
            
        Returns:
            Reconstructed audio tensor
        """
        with torch.no_grad():
            if self.is_hf_model:
                # HuggingFace 48kHz model - use complete decode pipeline
                if metadata is None:
                    raise ValueError("HuggingFace 48kHz model requires metadata (at least audio_scales) for proper reconstruction")

                audio_scales = metadata['audio_scales']
                # padding_mask = metadata['padding_mask']
                
                # Use the complete decode method which handles scales and padding properly
                # reconstructed_audio = self.model.decode(codes, audio_scales, padding_mask)[0]
                reconstructed_audio = self.model.decode(codes, audio_scales)[0]

            else:
                # Torchaudio 24kHz model - decode codes to embeddings then to audio
                embeddings = self.model.quantizer.decode(codes)
                reconstructed_audio = self.model.decoder(embeddings)
            
            return reconstructed_audio

    def decode_emb_audio(self, embeddings: torch.Tensor, metadata: Optional[dict] = None) -> torch.Tensor:
        """
        Decode embeddings back to audio.
        Works for both continuous and quantized embeddings.
        
        Args:
            embeddings: Tensor of shape [batch, channels, time_frames]
            
        Returns:
            Reconstructed audio tensor of shape [batch, channels, time]
        """
        with torch.no_grad():
            if self.is_hf_model:
                # HuggingFace 48kHz model
                if metadata is not None:
                    # Use audio_scales and padding_mask for proper reconstruction
                    audio_scales = metadata.get('audio_scales', None)
                    padding_mask = metadata.get('padding_mask', None)
                    
                    # For chunked encoding, we need to split the embeddings back to chunks
                    # and apply the corresponding scale to each chunk's audio output
                    if isinstance(audio_scales, list) and len(audio_scales) > 1:
                        # Calculate chunk size in embedding frames
                        total_frames = embeddings.shape[-1]  # 1050
                        num_chunks = len(audio_scales)  # 7 
                        chunk_size = total_frames // num_chunks  # 150
                        
                        chunk_audio = []
                        for i in range(num_chunks):
                            # Extract chunk from concatenated embeddings
                            start_idx = i * chunk_size
                            end_idx = start_idx + chunk_size if i < num_chunks - 1 else total_frames
                            chunk_emb = embeddings[..., start_idx:end_idx]
                            scale = audio_scales[i]
                            
                            # Decode chunk to audio first, then apply scale
                            audio = self.model.decoder(chunk_emb)
                            if scale is not None:
                                audio = audio * scale.view(-1, 1, 1)
                            chunk_audio.append(audio)
                        
                        # Use overlap-add to reconstruct full audio
                        stride = getattr(self.model.config, 'chunk_stride', None)
                        if stride is None:
                            # Estimate stride from audio chunks
                            stride = chunk_audio[0].shape[-1] // 2 if len(chunk_audio) > 1 else chunk_audio[0].shape[-1]
                        reconstructed_audio = self.model._linear_overlap_add(chunk_audio, stride)
                        
                        # Remove padding if mask is present
                        if padding_mask is not None and padding_mask.shape[-1] < reconstructed_audio.shape[-1]:
                            reconstructed_audio = reconstructed_audio[..., :padding_mask.shape[-1]]
                    else:
                        # Single chunk case
                        scale = audio_scales[0] if isinstance(audio_scales, list) else audio_scales
                        # Decode to audio first, then apply scale
                        reconstructed_audio = self.model.decoder(embeddings)
                        if scale is not None:
                            reconstructed_audio = reconstructed_audio * scale.view(-1, 1, 1)
                        if padding_mask is not None and padding_mask.shape[-1] < reconstructed_audio.shape[-1]:
                            reconstructed_audio = reconstructed_audio[..., :padding_mask.shape[-1]]
                else:
                    reconstructed_audio = self.model.decoder(embeddings)
            else:
                # HuggingFace 24kHz model
                # Decode from embeddings
                reconstructed_audio = self.model.decoder(embeddings)
            
            return reconstructed_audio
    
    def get_compression_ratio(self, audio_length: int) -> float:
        """
        Calculate the compression ratio for a given audio length.
        
        Args:
            audio_length: Length of audio in samples
            
        Returns:
            Compression ratio (audio_samples / embedding_frames)
        """
        embedding_frames = audio_length // (self.sample_rate // self.frame_rate)
        return audio_length / embedding_frames if embedding_frames > 0 else 0
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_type': f'EnCodec {self.model_type}',
            'sample_rate': self.sample_rate,
            'frame_rate': self.frame_rate,
            'device': self.device,
            'is_hf_model': self.is_hf_model,
            'encoder_type': type(self.model.encoder).__name__,
            'decoder_type': type(self.model.decoder).__name__,
            'quantizer_type': type(self.model.quantizer).__name__
        }


if __name__ == "__main__":
    # Simple test if run directly
    print("Testing EncodecProcessor...")
    
    # Test both 24kHz and 48kHz models
    for sr in [24000, 48000]:
        print(f"\n{'='*50}")
        print(f"Testing {sr}Hz model")
        print(f"{'='*50}")
        
        processor = EncodecProcessor(sr=sr)
        
        # Create test signal
        duration = 2.0
        t = torch.linspace(0, duration, int(processor.sample_rate * duration))
        test_audio = 0.5 * torch.sin(2 * np.pi * 440 * t)  # 440Hz sine wave
        
        print(f"\nTest audio: {duration}s sine wave at 440Hz")
        print(f"Audio shape: {test_audio.shape}")
        
        # Test continuous encoding/decoding
        print("\n=== Testing Continuous Path ===")
        cont_embeddings = processor.encode_audio_emb(test_audio)
        cont_reconstructed = processor.decode_emb_audio(cont_embeddings)
        
        print(f"Continuous embeddings shape: {cont_embeddings.shape}")
        print(f"Continuous reconstructed shape: {cont_reconstructed.shape}")
        
        # Test quantized encoding/decoding
        print("\n=== Testing Quantized Path ===")
        codes_result = processor.encode_audio_codes(test_audio, kbps=6.0)
        if processor.is_hf_model:
            codes, metadata = codes_result
            # Test the full pipeline: codes -> embeddings -> audio
            quant_embeddings = processor.decode_codes_emb(codes, metadata)
            quant_reconstructed_from_emb = processor.decode_emb_audio(quant_embeddings)
            # Also test direct codes -> audio
            quant_reconstructed = processor.decode_codes_audio(codes, metadata)
            print(f"Quantized embeddings shape: {quant_embeddings.shape}")
            print(f"Quantized reconstructed from embeddings shape: {quant_reconstructed_from_emb.shape}")
        else:
            codes, _ = codes_result
            quant_embeddings = processor.decode_codes_emb(codes)
            quant_reconstructed = processor.decode_emb_audio(quant_embeddings)
        
        print(f"Audio codes shape: {codes.shape}")
        print(f"Quantized reconstructed shape: {quant_reconstructed.shape}")
        
        print(f"\n=== Model Info ===")
        info = processor.get_model_info()
        for key, value in info.items():
            print(f"{key}: {value}")
    
    print("\n✓ All tests completed successfully!")
