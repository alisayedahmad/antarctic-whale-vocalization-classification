import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import cv2
from datetime import datetime
from tqdm import tqdm
import pickle


class AudioPreprocessor:
    """Handles audio file loading and spectrogram generation."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.sr = config['data']['spectrogram']['sample_rate']
        self.nfft = config['data']['spectrogram']['nfft']
        self.hop_length = config['data']['spectrogram']['hop_length']
        self.win_length = config['data']['spectrogram']['win_length']
        self.n_mels = config['data']['spectrogram']['n_mels']
        self.fmin = config['data']['spectrogram']['fmin']
        self.fmax = config['data']['spectrogram']['fmax']
        
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file."""
        audio, sr = librosa.load(audio_path, sr=self.sr)
        return audio, sr
    
    def compute_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Compute mel spectrogram from audio signal."""
        spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_fft=self.nfft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax
        )
        spec_db = librosa.power_to_db(spec, ref=np.max)
        return spec_db
    
    def extract_patch(
        self,
        spectrogram: np.ndarray,
        low_freq: float,
        high_freq: float,
        start_time: float,
        end_time: float
    ) -> np.ndarray:
        """Extract spectrogram patch based on time-frequency coordinates."""
        time_to_frame = lambda t: int(t * self.sr / self.hop_length)
        freq_to_bin = lambda f: int(f * self.n_mels / self.fmax)
        
        start_frame = max(0, time_to_frame(start_time))
        end_frame = min(spectrogram.shape[1], time_to_frame(end_time))
        low_bin = max(0, freq_to_bin(low_freq))
        high_bin = min(spectrogram.shape[0], freq_to_bin(high_freq))
        
        patch = spectrogram[low_bin:high_bin, start_frame:end_frame]
        return patch


class AnnotationParser:
    """Parses annotation CSV files."""
    
    @staticmethod
    def load_annotations(csv_path: str) -> pd.DataFrame:
        """Load annotation CSV file."""
        df = pd.read_csv(csv_path)
        return df
    
    @staticmethod
    def parse_datetime(dt_str: str) -> datetime:
        """Parse datetime string from annotation."""
        return pd.to_datetime(dt_str)
    
    def get_time_frequency_bounds(self, row: pd.Series) -> Dict:
        """Extract time and frequency bounds from annotation row."""
        start_dt = self.parse_datetime(row['start_datetime'])
        end_dt = self.parse_datetime(row['end_datetime'])
        
        duration = (end_dt - start_dt).total_seconds()
        
        return {
            'low_freq': row['low_frequency'],
            'high_freq': row['high_frequency'],
            'start_time': 0,
            'end_time': duration,
            'annotation': row['annotation']
        }


class ImagePreprocessor:
    """Handles image preprocessing and normalization."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.target_height = config['data']['image']['target_height']
        self.target_width = config['data']['image']['target_width']
        self.normalization = config['data']['image']['normalization']
    
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target dimensions."""
        if image.shape[0] == 0 or image.shape[1] == 0:
            return np.zeros((self.target_height, self.target_width))
        
        resized = cv2.resize(
            image,
            (self.target_width, self.target_height),
            interpolation=cv2.INTER_LINEAR
        )
        return resized
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image values."""
        if self.normalization == 'minmax':
            img_min, img_max = image.min(), image.max()
            if img_max - img_min > 0:
                normalized = (image - img_min) / (img_max - img_min)
            else:
                normalized = image
        elif self.normalization == 'standard':
            mean, std = image.mean(), image.std()
            if std > 0:
                normalized = (image - mean) / std
            else:
                normalized = image - mean
        else:
            normalized = image
        
        return normalized
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Apply full preprocessing pipeline."""
        resized = self.resize_image(image)
        normalized = self.normalize_image(resized)
        return normalized


class DatasetBuilder:
    """Builds processed dataset from raw audio and annotations."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.audio_preprocessor = AudioPreprocessor(config)
        self.annotation_parser = AnnotationParser()
        self.image_preprocessor = ImagePreprocessor(config)
        
    def process_single_file(
        self,
        audio_path: Path,
        annotation_row: pd.Series
    ) -> Tuple[np.ndarray, str]:
        """Process single audio file and extract annotated patch."""
        audio, _ = self.audio_preprocessor.load_audio(str(audio_path))
        spectrogram = self.audio_preprocessor.compute_spectrogram(audio)
        
        bounds = self.annotation_parser.get_time_frequency_bounds(annotation_row)
        patch = self.audio_preprocessor.extract_patch(
            spectrogram,
            bounds['low_freq'],
            bounds['high_freq'],
            bounds['start_time'],
            bounds['end_time']
        )
        
        processed_patch = self.image_preprocessor.preprocess(patch)
        label = bounds['annotation']
        
        return processed_patch, label
    
    def build_dataset(
        self,
        data_split: str = 'train'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build complete dataset for train or test split."""
        raw_path = Path(self.config['data']['raw_data_path'])
        split_path = raw_path / data_split
        
        audio_path = split_path / 'audio'
        annotation_path = split_path / 'annotations'
        
        images = []
        labels = []
        
        annotation_files = list(annotation_path.glob('*.csv'))
        
        for ann_file in tqdm(annotation_files, desc=f"Processing {data_split} data"):
            df = self.annotation_parser.load_annotations(str(ann_file))
            dataset_name = ann_file.stem
            audio_dir = audio_path / dataset_name
            
            for _, row in df.iterrows():
                audio_file = audio_dir / row['filename']
                
                if audio_file.exists():
                    try:
                        patch, label = self.process_single_file(audio_file, row)
                        images.append(patch)
                        labels.append(label)
                    except Exception as e:
                        print(f"Error processing {audio_file}: {e}")
                        continue
        
        X = np.array(images)
        y = np.array(labels)
        
        return X, y
    
    def save_processed_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        split: str
    ):
        """Save processed data to disk."""
        output_path = Path(self.config['data']['processed_data_path']) / split
        output_path.mkdir(parents=True, exist_ok=True)
        
        np.save(output_path / 'X.npy', X)
        np.save(output_path / 'y.npy', y)
        
        with open(output_path / 'label_mapping.pkl', 'wb') as f:
            unique_labels = np.unique(y)
            label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
            pickle.dump(label_to_idx, f)
        
        print(f"Saved {split} data: {X.shape[0]} samples")
    
    def load_processed_data(
        self,
        split: str
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Load processed data from disk."""
        data_path = Path(self.config['data']['processed_data_path']) / split
        
        X = np.load(data_path / 'X.npy')
        y = np.load(data_path / 'y.npy')
        
        with open(data_path / 'label_mapping.pkl', 'rb') as f:
            label_mapping = pickle.load(f)
        
        return X, y, label_mapping


class LabelEncoder:
    """Encodes labels according to class mappings."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.original_classes = config['data']['classes']['original']
        self.merged_mapping = config['data']['classes']['merged']
        
    def merge_labels(self, labels: np.ndarray) -> np.ndarray:
        """Merge original labels into broader categories."""
        merged_labels = np.empty(len(labels), dtype=object)
        
        for idx, label in enumerate(labels):
            for merged_class, original_list in self.merged_mapping.items():
                if label in original_list:
                    merged_labels[idx] = merged_class
                    break
        
        return merged_labels
    
    def encode_labels(self, labels: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Convert string labels to integer indices."""
        unique_labels = np.unique(labels)
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        
        encoded = np.array([label_to_idx[label] for label in labels])
        
        return encoded, label_to_idx
    
    def decode_labels(
        self,
        encoded_labels: np.ndarray,
        label_mapping: Dict
    ) -> np.ndarray:
        """Convert integer indices back to string labels."""
        idx_to_label = {idx: label for label, idx in label_mapping.items()}
        decoded = np.array([idx_to_label[idx] for idx in encoded_labels])
        
        return decoded
