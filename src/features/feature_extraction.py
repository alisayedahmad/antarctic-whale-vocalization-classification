import numpy as np
import librosa
from typing import Dict, List
from scipy import stats


class FeatureExtractor:
    """Extracts various features from spectrogram images."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.mfcc_coeff = config['features']['mfcc_coefficients']
        self.chroma_bins = config['features']['chroma_bins']
        
    def extract_statistical_features(self, image: np.ndarray) -> np.ndarray:
        """Extract statistical features from image."""
        features = []
        
        flat_image = image.flatten()
        
        features.extend([
            np.mean(flat_image),
            np.std(flat_image),
            np.median(flat_image),
            np.max(flat_image),
            np.min(flat_image),
            stats.skew(flat_image),
            stats.kurtosis(flat_image)
        ])
        
        percentiles = np.percentile(flat_image, [25, 50, 75])
        features.extend(percentiles)
        
        row_means = np.mean(image, axis=1)
        col_means = np.mean(image, axis=0)
        features.extend([
            np.mean(row_means),
            np.std(row_means),
            np.mean(col_means),
            np.std(col_means)
        ])
        
        return np.array(features)
    
    def extract_spectral_features(self, image: np.ndarray) -> np.ndarray:
        """Extract spectral features from image."""
        features = []
        
        spectral_centroid = np.mean(np.sum(image * np.arange(image.shape[0])[:, None], axis=0) / 
                                    (np.sum(image, axis=0) + 1e-10))
        features.append(spectral_centroid)
        
        freq_bins = np.arange(image.shape[0])
        spectral_spread = np.sqrt(np.mean(
            np.sum(((freq_bins[:, None] - spectral_centroid) ** 2) * image, axis=0) /
            (np.sum(image, axis=0) + 1e-10)
        ))
        features.append(spectral_spread)
        
        energy_cumsum = np.cumsum(np.sum(image, axis=1))
        total_energy = energy_cumsum[-1]
        rolloff_idx = np.where(energy_cumsum >= 0.85 * total_energy)[0]
        spectral_rolloff = rolloff_idx[0] if len(rolloff_idx) > 0 else image.shape[0]
        features.append(spectral_rolloff)
        
        spectral_flux = np.mean(np.sqrt(np.sum(np.diff(image, axis=1) ** 2, axis=0)))
        features.append(spectral_flux)
        
        return np.array(features)
    
    def extract_temporal_features(self, image: np.ndarray) -> np.ndarray:
        """Extract temporal features from image."""
        features = []
        
        energy_contour = np.sum(image, axis=0)
        features.extend([
            np.mean(energy_contour),
            np.std(energy_contour),
            np.max(energy_contour),
            np.min(energy_contour)
        ])
        
        zero_crossings = np.sum(np.abs(np.diff(np.sign(energy_contour - np.mean(energy_contour)))) > 0)
        features.append(zero_crossings)
        
        features.append(image.shape[1])
        
        attack_time = np.argmax(energy_contour)
        features.append(attack_time / image.shape[1])
        
        return np.array(features)
    
    def extract_texture_features(self, image: np.ndarray) -> np.ndarray:
        """Extract texture features using GLCM-inspired statistics."""
        features = []
        
        h_diff = np.diff(image, axis=1)
        v_diff = np.diff(image, axis=0)
        
        features.extend([
            np.mean(np.abs(h_diff)),
            np.std(h_diff),
            np.mean(np.abs(v_diff)),
            np.std(v_diff)
        ])
        
        h_energy = np.sum(h_diff ** 2)
        v_energy = np.sum(v_diff ** 2)
        features.extend([h_energy, v_energy])
        
        return np.array(features)
    
    def extract_shape_features(self, image: np.ndarray) -> np.ndarray:
        """Extract shape-related features."""
        features = []
        
        threshold = np.mean(image) + 0.5 * np.std(image)
        binary = (image > threshold).astype(int)
        
        height, width = binary.shape
        features.extend([height, width, height / (width + 1e-10)])
        
        if np.sum(binary) > 0:
            y_coords, x_coords = np.where(binary == 1)
            centroid_y = np.mean(y_coords)
            centroid_x = np.mean(x_coords)
            features.extend([centroid_y / height, centroid_x / width])
            
            extent = np.sum(binary) / (height * width)
            features.append(extent)
        else:
            features.extend([0.5, 0.5, 0.0])
        
        return np.array(features)
    
    def extract_all_features(self, image: np.ndarray) -> np.ndarray:
        """Extract all feature types and concatenate."""
        feature_vectors = []
        
        if 'statistical' in self.config['features']:
            feature_vectors.append(self.extract_statistical_features(image))
        
        if 'spectral' in self.config['features']:
            feature_vectors.append(self.extract_spectral_features(image))
        
        if 'temporal' in self.config['features']:
            feature_vectors.append(self.extract_temporal_features(image))
        
        feature_vectors.append(self.extract_texture_features(image))
        feature_vectors.append(self.extract_shape_features(image))
        
        all_features = np.concatenate(feature_vectors)
        
        return all_features
    
    def extract_batch_features(self, images: np.ndarray) -> np.ndarray:
        """Extract features for a batch of images."""
        features_list = []
        
        for image in images:
            features = self.extract_all_features(image)
            features_list.append(features)
        
        return np.array(features_list)


class FeatureSelector:
    """Performs feature selection and dimensionality reduction."""
    
    def __init__(self, method: str = 'variance'):
        self.method = method
        self.selected_indices = None
        
    def fit(self, X: np.ndarray, y: np.ndarray = None, threshold: float = 0.01):
        """Fit feature selector."""
        if self.method == 'variance':
            variances = np.var(X, axis=0)
            self.selected_indices = np.where(variances > threshold)[0]
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features using selected indices."""
        if self.selected_indices is not None:
            return X[:, self.selected_indices]
        return X
    
    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray = None,
        threshold: float = 0.01
    ) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X, y, threshold)
        return self.transform(X)
