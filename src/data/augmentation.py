import numpy as np
import librosa
from typing import Dict, Tuple


class AudioAugmenter:
    """Audio data augmentation techniques."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.aug_config = config['data']['augmentation']
        
    def time_stretch(self, audio: np.ndarray, rate: float = None) -> np.ndarray:
        """Apply time stretching to audio."""
        if rate is None:
            rate = np.random.uniform(*self.aug_config['time_stretch'])
        
        stretched = librosa.effects.time_stretch(audio, rate=rate)
        return stretched
    
    def pitch_shift(self, audio: np.ndarray, sr: int, n_steps: float = None) -> np.ndarray:
        """Apply pitch shifting to audio."""
        if n_steps is None:
            n_steps = np.random.uniform(*self.aug_config['pitch_shift'])
        
        shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
        return shifted
    
    def add_noise(self, audio: np.ndarray, noise_factor: float = None) -> np.ndarray:
        """Add random noise to audio."""
        if noise_factor is None:
            noise_factor = self.aug_config['noise_factor']
        
        noise = np.random.randn(len(audio))
        augmented = audio + noise_factor * noise
        
        augmented = np.clip(augmented, -1.0, 1.0)
        return augmented
    
    def time_shift(self, audio: np.ndarray, shift_max: float = None) -> np.ndarray:
        """Apply time shifting to audio."""
        if shift_max is None:
            shift_max = self.aug_config['time_shift']
        
        shift = int(np.random.uniform(-shift_max, shift_max) * len(audio))
        
        if shift > 0:
            augmented = np.pad(audio, (shift, 0), mode='constant')[:-shift]
        elif shift < 0:
            augmented = np.pad(audio, (0, -shift), mode='constant')[-shift:]
        else:
            augmented = audio
        
        return augmented
    
    def augment_audio(
        self,
        audio: np.ndarray,
        sr: int,
        methods: list = None
    ) -> np.ndarray:
        """Apply multiple augmentation methods."""
        if not self.aug_config['enabled']:
            return audio
        
        if methods is None:
            methods = ['time_stretch', 'pitch_shift', 'noise', 'time_shift']
        
        augmented = audio.copy()
        
        if 'time_stretch' in methods and np.random.rand() > 0.5:
            augmented = self.time_stretch(augmented)
        
        if 'pitch_shift' in methods and np.random.rand() > 0.5:
            augmented = self.pitch_shift(augmented, sr)
        
        if 'noise' in methods and np.random.rand() > 0.5:
            augmented = self.add_noise(augmented)
        
        if 'time_shift' in methods and np.random.rand() > 0.5:
            augmented = self.time_shift(augmented)
        
        return augmented


class ImageAugmenter:
    """Image augmentation for spectrograms."""
    
    def __init__(self):
        pass
    
    def horizontal_flip(self, image: np.ndarray) -> np.ndarray:
        """Flip image horizontally (time reversal)."""
        return np.fliplr(image)
    
    def vertical_flip(self, image: np.ndarray) -> np.ndarray:
        """Flip image vertically (frequency reversal)."""
        return np.flipud(image)
    
    def add_noise(self, image: np.ndarray, noise_factor: float = 0.01) -> np.ndarray:
        """Add random noise to image."""
        noise = np.random.randn(*image.shape) * noise_factor
        augmented = image + noise
        return augmented
    
    def random_crop(
        self,
        image: np.ndarray,
        crop_size: Tuple[int, int]
    ) -> np.ndarray:
        """Randomly crop image."""
        h, w = image.shape
        crop_h, crop_w = crop_size
        
        if h <= crop_h or w <= crop_w:
            return image
        
        top = np.random.randint(0, h - crop_h)
        left = np.random.randint(0, w - crop_w)
        
        cropped = image[top:top+crop_h, left:left+crop_w]
        return cropped
    
    def augment_image(
        self,
        image: np.ndarray,
        methods: list = None
    ) -> np.ndarray:
        """Apply multiple augmentation methods."""
        if methods is None:
            methods = ['noise']
        
        augmented = image.copy()
        
        if 'horizontal_flip' in methods and np.random.rand() > 0.5:
            augmented = self.horizontal_flip(augmented)
        
        if 'noise' in methods and np.random.rand() > 0.5:
            augmented = self.add_noise(augmented)
        
        return augmented
