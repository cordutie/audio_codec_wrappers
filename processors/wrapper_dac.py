"""
DAC Audio Processor Class

A wrapper class for DAC (Descript Audio Codec) models that provides easy access to:
- Continuous embeddings (pre-quantization)
- Quantized embeddings 
- Audio reconstruction from both types

This class handles the conversion between audio and DAC representations,
supporting both the continuous path and quantized path (standard DAC).
"""

import torch
import dac
from audiotools import AudioSignal
import numpy as np
from typing import Optional, Union


class DACProcessor:
    """
    A processor for DAC models with support for both continuous and quantized embeddings.
    
    This class provides an easy-to-use interface for:
    - Converting audio to continuous embeddings (pre-quantization)
    - Converting audio to quantized embeddings (standard DAC)
    - Reconstructing audio from both embedding types
    - Support for 16kHz, 24kHz, and 44.1kHz models
    """
    
    def __init__(self, sr: int = 44100, device: Optional[str] = None):
        """
        Initialize the DAC processor.
        
        Args:
            sr: Sample rate (16000, 24000, or 44100)
            device: Device to run the model on ('cuda', 'cpu', or None for auto-detect)
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.target_sr = sr
        
        print(f"Initializing DAC processor on device: {self.device}")
        
        # Determine model type from sample rate
        if sr in [16000, 24000, 44100]:
            model_type = f"{sr // 1000}khz"
            print(f"Loading DAC {model_type} model...")
            model_path = dac.utils.download(model_type=model_type)
            self.model = dac.DAC.load(model_path)
            self.model = self.model.to(self.device)
            self.model.eval()
            self.sample_rate = self.model.sample_rate
            self.model_type = model_type
            print(f"✓ DAC {model_type} model loaded")
        else:
            raise ValueError(f"Unsupported sample rate: {sr}. Supported rates: 16000, 24000, 44100")
        
        print(f"✓ Sample rate: {self.sample_rate} Hz")
        print(f"✓ Latent dimension: {self.model.latent_dim}")
        print(f"✓ Codebooks: {self.model.n_codebooks}")
        print(f"✓ Codebook size: {self.model.codebook_size}")
    
    def _prepare_audio_signal(self, audio: Union[torch.Tensor, np.ndarray, AudioSignal], 
                              sample_rate: Optional[int] = None) -> AudioSignal:
        """
        Prepare audio as AudioSignal for DAC processing.
        
        Args:
            audio: Audio data as tensor, numpy array, or AudioSignal
            sample_rate: Sample rate of input audio (will resample if needed)
            
        Returns:
            AudioSignal ready for DAC processing
        """
        # If already an AudioSignal, just ensure it's on the right device
        if isinstance(audio, AudioSignal):
            if sample_rate is not None and sample_rate != self.sample_rate:
                audio = audio.resample(self.sample_rate)
            return audio.to(self.device)
        
        # Convert to tensor if needed
        if isinstance(audio, np.ndarray):
            audio = torch.tensor(audio, dtype=torch.float32)
        
        # Ensure correct shape: [batch, channels, time]
        if audio.dim() == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)  # [time] -> [1, 1, time]
        elif audio.dim() == 2:
            if audio.shape[0] > audio.shape[1]:
                # Likely [time, channels] -> transpose to [channels, time]
                audio = audio.T.unsqueeze(0)  # -> [1, channels, time]
            else:
                # Likely [channels, time] -> add batch dimension
                audio = audio.unsqueeze(0)  # -> [1, channels, time]
        
        # Create AudioSignal
        signal_sr = sample_rate if sample_rate is not None else self.sample_rate
        audio_signal = AudioSignal(audio, sample_rate=signal_sr)
        
        # Resample if needed
        if signal_sr != self.sample_rate:
            print(f"Resampling from {signal_sr}Hz to {self.sample_rate}Hz")
            audio_signal = audio_signal.resample(self.sample_rate)
        
        return audio_signal.to(self.device)
    
    def audio_to_latents(self, audio: Union[torch.Tensor, np.ndarray, AudioSignal], 
                        sample_rate: Optional[int] = None):
        """
        Convert audio to continuous (pre-quantization) latent embeddings.
        This corresponds to the encoder output before quantization.
        
        Args:
            audio: Audio as tensor, numpy array, or AudioSignal
            sample_rate: Sample rate of input audio (will resample if needed)
        
        Returns:
            (latents, metadata) where latents is the raw encoder output [batch, channels, time_frames]
            and metadata contains preprocessing information
        """
        with torch.no_grad():
            audio_signal = self._prepare_audio_signal(audio, sample_rate)
            
            # Preprocess: padding for hop_length alignment
            preprocessed = self.model.preprocess(audio_signal.audio_data, audio_signal.sample_rate)
            
            # Encode to raw latents
            raw_latents = self.model.encoder(preprocessed)
            
            metadata = {
                'original_length': audio_signal.audio_data.shape[-1],
                'preprocessed_shape': preprocessed.shape
            }
            
            return raw_latents, metadata
    
    def latents_to_codes(self, latents: torch.Tensor, latents_meta: Optional[dict] = None):
        """
        Convert continuous latent embeddings to discrete codes via quantization.
        
        Args:
            latents: Raw latent embeddings from encoder [batch, channels, time_frames]
            latents_meta: Optional metadata (not used for DAC, included for API consistency)
        
        Returns:
            (codes, metadata) where codes is [batch, num_codebooks, time_frames]
            and metadata contains quantized latents and other info
        """
        with torch.no_grad():
            # DAC quantizer returns: z_quantized, codes, latents, commitment_loss, codebook_loss
            z_quantized, codes, _, commitment_loss, codebook_loss = self.model.quantizer(latents)
            
            metadata = {
                'quantized_latents': z_quantized,
                'commitment_loss': commitment_loss,
                'codebook_loss': codebook_loss
            }
            
            return codes, metadata
    
    def codes_to_latents(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Decode discrete codes to quantized embeddings (latents).
        
        Args:
            codes: Discrete codes [batch, num_codebooks, time_frames]
        
        Returns:
            Quantized embeddings tensor [batch, channels, time_frames]
        """
        with torch.no_grad():
            # Use the quantizer's built-in from_codes method
            # from_codes returns: z_q (quantized latents), z_p (projected), codes_out
            z_q, z_p, codes_out = self.model.quantizer.from_codes(codes)
            return z_q
    
    def decode_latents_audio(self, embeddings: torch.Tensor, metadata: Optional[dict] = None) -> torch.Tensor:
        """
        Decode embeddings back to audio.
        Works for both continuous (raw) and quantized embeddings.
        
        Args:
            embeddings: Tensor of shape [batch, channels, time_frames]
            metadata: Optional metadata (not used for DAC, included for API consistency)
            
        Returns:
            Reconstructed audio tensor of shape [batch, channels, time]
        """
        with torch.no_grad():
            reconstructed_audio = self.model.decoder(embeddings)
            return reconstructed_audio
    
    # ===== HIGH-LEVEL CONVENIENCE FUNCTIONS (matching EnCodec API) =====
    
    def encode_audio_emb(self, audio: Union[torch.Tensor, np.ndarray, AudioSignal], 
                        sample_rate: Optional[int] = None) -> torch.Tensor:
        """
        Encode audio to continuous embeddings (pre-quantization).
        
        Args:
            audio: Audio as tensor, numpy array, or AudioSignal
            sample_rate: Sample rate of input audio (will resample if needed)
            
        Returns:
            Continuous embeddings tensor of shape [batch, channels, time_frames]
        """
        latents, _ = self.audio_to_latents(audio, sample_rate)
        return latents
    
    def encode_audio_codes(self, audio: Union[torch.Tensor, np.ndarray, AudioSignal],
                          sample_rate: Optional[int] = None) -> tuple:
        """
        Encode audio to quantized codes (standard DAC with discrete codes).
        
        Args:
            audio: Audio as tensor, numpy array, or AudioSignal
            sample_rate: Sample rate of input audio (will resample if needed)
            
        Returns:
            (codes, metadata) where codes is [batch, num_codebooks, time_frames]
        """
        latents, latents_meta = self.audio_to_latents(audio, sample_rate)
        codes, codes_meta = self.latents_to_codes(latents, latents_meta)
        return codes, codes_meta
    
    def decode_codes_emb(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Decode discrete audio codes back to quantized embeddings.
        
        Args:
            codes: Discrete codes tensor [batch, num_codebooks, time_frames]
            
        Returns:
            Quantized embeddings tensor of shape [batch, channels, time_frames]
        """
        return self.codes_to_latents(codes)
    
    def decode_codes_audio(self, codes: torch.Tensor, metadata: Optional[dict] = None) -> torch.Tensor:
        """
        Decode discrete audio codes directly to audio using the complete quantized pipeline.
        
        Args:
            codes: Discrete codes tensor [batch, num_codebooks, time_frames]
            metadata: Optional metadata (not used for DAC, included for API consistency)
            
        Returns:
            Reconstructed audio tensor
        """
        latents = self.codes_to_latents(codes)
        audio = self.decode_latents_audio(latents, metadata)
        return audio
    
    # ===== UTILITY FUNCTIONS =====
    
    def get_compression_ratio(self, audio_length: int) -> float:
        """
        Calculate the compression ratio for a given audio length.
        
        Args:
            audio_length: Length of audio in samples
            
        Returns:
            Compression ratio (audio_samples / embedding_frames)
        """
        # DAC uses downsampling based on encoder stride
        # Typical encoder downsampling is ~320 for 44.1kHz model
        hop_length = self.model.encoder_rates[0] if hasattr(self.model, 'encoder_rates') else 512
        embedding_frames = audio_length // hop_length
        return audio_length / embedding_frames if embedding_frames > 0 else 0
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_type': f'DAC {self.model_type}',
            'sample_rate': self.sample_rate,
            'device': str(self.device),
            'latent_dim': self.model.latent_dim,
            'n_codebooks': self.model.n_codebooks,
            'codebook_size': self.model.codebook_size,
            'codebook_dim': self.model.codebook_dim,
            'encoder_type': type(self.model.encoder).__name__,
            'decoder_type': type(self.model.decoder).__name__,
            'quantizer_type': type(self.model.quantizer).__name__
        }


if __name__ == "__main__":
    # Simple test if run directly
    print("Testing DACProcessor...")
    
    # Test 44.1kHz model (you can also test 16000 or 24000)
    for sr in [44100]:
        print(f"\n{'='*50}")
        print(f"Testing {sr}Hz model")
        print(f"{'='*50}")
        
        processor = DACProcessor(sr=sr)
        
        # Create test signal
        duration = 2.0
        t = torch.linspace(0, duration, int(processor.sample_rate * duration))
        test_audio = 0.5 * torch.sin(2 * np.pi * 440 * t)  # 440Hz sine wave
        
        print(f"\nTest audio: {duration}s sine wave at 440Hz")
        print(f"Audio shape: {test_audio.shape}")
        
        # Test continuous encoding/decoding
        print("\n=== Testing Continuous Path ===")
        cont_embeddings = processor.encode_audio_emb(test_audio)
        cont_reconstructed = processor.decode_latents_audio(cont_embeddings)
        
        print(f"Continuous embeddings shape: {cont_embeddings.shape}")
        print(f"Continuous reconstructed shape: {cont_reconstructed.shape}")
        
        # Test quantized encoding/decoding
        print("\n=== Testing Quantized Path ===")
        codes, metadata = processor.encode_audio_codes(test_audio)
        quant_embeddings = processor.decode_codes_emb(codes)
        quant_reconstructed = processor.decode_codes_audio(codes, metadata)
        
        print(f"Audio codes shape: {codes.shape}")
        print(f"Quantized embeddings shape: {quant_embeddings.shape}")
        print(f"Quantized reconstructed shape: {quant_reconstructed.shape}")
        
        # Test step-by-step atomic functions
        print("\n=== Testing Atomic Functions ===")
        latents, latents_meta = processor.audio_to_latents(test_audio)
        codes2, codes_meta = processor.latents_to_codes(latents, latents_meta)
        latents2 = processor.codes_to_latents(codes2)
        audio_final = processor.decode_latents_audio(latents2)
        
        print(f"Latents shape: {latents.shape}")
        print(f"Codes shape: {codes2.shape}")
        print(f"Reconstructed latents shape: {latents2.shape}")
        print(f"Final audio shape: {audio_final.shape}")
        
        # Verify codes→latents reconstruction is correct
        print("\n=== Verification ===")
        # Check if quantized latents from metadata match reconstructed latents
        quantized_from_meta = metadata['quantized_latents']
        reconstruction_error = torch.nn.functional.mse_loss(latents2, quantized_from_meta)
        print(f"Codes→Latents reconstruction MSE: {reconstruction_error.item():.10f}")
        is_perfect = torch.allclose(latents2, quantized_from_meta, atol=1e-6)
        print(f"Perfect reconstruction: {is_perfect}")
        
        print(f"\n=== Model Info ===")
        info = processor.get_model_info()
        for key, value in info.items():
            print(f"{key}: {value}")
    
    print("\n✓ All tests completed successfully!")
