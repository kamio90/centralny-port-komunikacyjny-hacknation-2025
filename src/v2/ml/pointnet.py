"""
PointNet - Deep Learning dla Chmur Punktow

Implementacja architektury PointNet dla klasyfikacji punktow w chmurach LiDAR.
Ref: Qi et al. "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation"

HackNation 2025 - CPK Chmura+
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Sprawdz czy PyTorch jest dostepny
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. PointNet will use fallback mode.")


@dataclass
class PointNetConfig:
    """Konfiguracja modelu PointNet"""
    n_classes: int = 10
    input_channels: int = 9  # xyz + rgb + intensity + normals
    hidden_dims: List[int] = None  # [64, 128, 256]
    dropout: float = 0.3
    use_bn: bool = True
    use_tnet: bool = True  # Spatial transformer

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [64, 128, 256]


if TORCH_AVAILABLE:

    class TNet(nn.Module):
        """
        Spatial Transformer Network dla PointNet

        Uczy sie transformacji affinicznej punktow aby model byl
        niezmienczy wzgledem rotacji i translacji.
        """

        def __init__(self, k: int = 3, use_bn: bool = True):
            super().__init__()
            self.k = k

            # Shared MLP
            self.conv1 = nn.Conv1d(k, 64, 1)
            self.conv2 = nn.Conv1d(64, 128, 1)
            self.conv3 = nn.Conv1d(128, 1024, 1)

            # Fully connected
            self.fc1 = nn.Linear(1024, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, k * k)

            # Batch norm
            self.use_bn = use_bn
            if use_bn:
                self.bn1 = nn.BatchNorm1d(64)
                self.bn2 = nn.BatchNorm1d(128)
                self.bn3 = nn.BatchNorm1d(1024)
                self.bn4 = nn.BatchNorm1d(512)
                self.bn5 = nn.BatchNorm1d(256)

        def forward(self, x):
            batch_size = x.size(0)

            # Shared MLP
            if self.use_bn:
                x = F.relu(self.bn1(self.conv1(x)))
                x = F.relu(self.bn2(self.conv2(x)))
                x = F.relu(self.bn3(self.conv3(x)))
            else:
                x = F.relu(self.conv1(x))
                x = F.relu(self.conv2(x))
                x = F.relu(self.conv3(x))

            # Max pooling (symmetric function)
            x = torch.max(x, 2)[0]

            # FC layers
            if self.use_bn:
                x = F.relu(self.bn4(self.fc1(x)))
                x = F.relu(self.bn5(self.fc2(x)))
            else:
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))

            x = self.fc3(x)

            # Dodaj macierz identycznosci (uczymy sie residual transform)
            identity = torch.eye(self.k, device=x.device).view(1, self.k * self.k).repeat(batch_size, 1)
            x = x + identity
            x = x.view(-1, self.k, self.k)

            return x


    class PointNetEncoder(nn.Module):
        """
        Encoder PointNet - ekstrakcja globalnych i lokalnych cech
        """

        def __init__(self, config: PointNetConfig):
            super().__init__()
            self.config = config

            # Input transform (3x3 dla xyz)
            if config.use_tnet:
                self.input_transform = TNet(k=3, use_bn=config.use_bn)
                self.feature_transform = TNet(k=64, use_bn=config.use_bn)
            else:
                self.input_transform = None
                self.feature_transform = None

            # Shared MLP 1
            self.conv1 = nn.Conv1d(config.input_channels, 64, 1)
            self.conv2 = nn.Conv1d(64, 64, 1)

            # Shared MLP 2
            dims = config.hidden_dims
            self.conv3 = nn.Conv1d(64, dims[0], 1)
            self.conv4 = nn.Conv1d(dims[0], dims[1], 1)
            self.conv5 = nn.Conv1d(dims[1], dims[2], 1)

            # Batch norm
            if config.use_bn:
                self.bn1 = nn.BatchNorm1d(64)
                self.bn2 = nn.BatchNorm1d(64)
                self.bn3 = nn.BatchNorm1d(dims[0])
                self.bn4 = nn.BatchNorm1d(dims[1])
                self.bn5 = nn.BatchNorm1d(dims[2])

        def forward(self, x):
            """
            Args:
                x: (B, C, N) - batch, channels, points
            Returns:
                local_features: (B, 64, N)
                global_features: (B, hidden_dims[-1])
            """
            batch_size = x.size(0)
            n_points = x.size(2)

            # Input transform
            if self.input_transform is not None:
                xyz = x[:, :3, :]  # Tylko xyz
                trans = self.input_transform(xyz)
                xyz = torch.bmm(trans, xyz)
                x = torch.cat([xyz, x[:, 3:, :]], dim=1)

            # Shared MLP 1
            if self.config.use_bn:
                x = F.relu(self.bn1(self.conv1(x)))
                x = F.relu(self.bn2(self.conv2(x)))
            else:
                x = F.relu(self.conv1(x))
                x = F.relu(self.conv2(x))

            # Feature transform
            if self.feature_transform is not None:
                trans_feat = self.feature_transform(x)
                x = torch.bmm(trans_feat, x)

            local_features = x  # (B, 64, N)

            # Shared MLP 2
            if self.config.use_bn:
                x = F.relu(self.bn3(self.conv3(x)))
                x = F.relu(self.bn4(self.conv4(x)))
                x = self.bn5(self.conv5(x))
            else:
                x = F.relu(self.conv3(x))
                x = F.relu(self.conv4(x))
                x = self.conv5(x)

            # Max pooling - global features
            global_features = torch.max(x, 2)[0]  # (B, hidden_dims[-1])

            return local_features, global_features


    class PointNetSegmentation(nn.Module):
        """
        PointNet dla segmentacji semantycznej (klasyfikacja per-punkt)

        Laczy lokalne i globalne cechy do klasyfikacji kazdego punktu.
        """

        def __init__(self, config: PointNetConfig):
            super().__init__()
            self.config = config
            self.encoder = PointNetEncoder(config)

            # Decoder - segmentation head
            # Concat local (64) + global (hidden_dims[-1]) features
            concat_dim = 64 + config.hidden_dims[-1]

            self.conv1 = nn.Conv1d(concat_dim, 256, 1)
            self.conv2 = nn.Conv1d(256, 128, 1)
            self.conv3 = nn.Conv1d(128, 64, 1)
            self.conv4 = nn.Conv1d(64, config.n_classes, 1)

            if config.use_bn:
                self.bn1 = nn.BatchNorm1d(256)
                self.bn2 = nn.BatchNorm1d(128)
                self.bn3 = nn.BatchNorm1d(64)

            self.dropout = nn.Dropout(config.dropout)

        def forward(self, x):
            """
            Args:
                x: (B, C, N) - batch, channels, points
            Returns:
                logits: (B, n_classes, N)
            """
            batch_size = x.size(0)
            n_points = x.size(2)

            # Encode
            local_feat, global_feat = self.encoder(x)

            # Expand global features i polacz z lokalnymi
            global_feat = global_feat.unsqueeze(2).repeat(1, 1, n_points)
            x = torch.cat([local_feat, global_feat], dim=1)

            # Decode
            if self.config.use_bn:
                x = F.relu(self.bn1(self.conv1(x)))
                x = F.relu(self.bn2(self.conv2(x)))
                x = self.dropout(x)
                x = F.relu(self.bn3(self.conv3(x)))
            else:
                x = F.relu(self.conv1(x))
                x = F.relu(self.conv2(x))
                x = self.dropout(x)
                x = F.relu(self.conv3(x))

            x = self.conv4(x)

            return x


    class PointCloudDataset(Dataset):
        """Dataset dla chmur punktow"""

        def __init__(
            self,
            coords: np.ndarray,
            labels: np.ndarray,
            colors: Optional[np.ndarray] = None,
            intensity: Optional[np.ndarray] = None,
            normals: Optional[np.ndarray] = None,
            block_size: int = 4096,
            augment: bool = False
        ):
            self.coords = coords.astype(np.float32)
            self.labels = labels.astype(np.int64)
            self.colors = colors.astype(np.float32) / 255.0 if colors is not None else None
            self.intensity = intensity.astype(np.float32) if intensity is not None else None
            self.normals = normals.astype(np.float32) if normals is not None else None
            self.block_size = block_size
            self.augment = augment

            # Normalizuj wspolrzedne do [-1, 1]
            self.coords_min = self.coords.min(axis=0)
            self.coords_max = self.coords.max(axis=0)
            self.coords_range = self.coords_max - self.coords_min
            self.coords_range[self.coords_range == 0] = 1

            n_points = len(coords)
            self.n_blocks = (n_points + block_size - 1) // block_size

        def __len__(self):
            return self.n_blocks

        def __getitem__(self, idx):
            start = idx * self.block_size
            end = min(start + self.block_size, len(self.coords))

            # Pobierz punkty
            coords = self.coords[start:end].copy()
            labels = self.labels[start:end]

            # Normalizuj wspolrzedne
            coords = (coords - self.coords_min) / self.coords_range * 2 - 1

            # Augmentacja
            if self.augment:
                coords = self._augment(coords)

            # Zbuduj tensor wejsciowy
            features = [coords]

            if self.colors is not None:
                features.append(self.colors[start:end])

            if self.intensity is not None:
                intensity = self.intensity[start:end]
                if intensity.ndim == 1:
                    intensity = intensity.reshape(-1, 1)
                features.append(intensity)

            if self.normals is not None:
                features.append(self.normals[start:end])

            x = np.concatenate(features, axis=1)

            # Pad jesli za malo punktow
            actual_size = end - start
            if actual_size < self.block_size:
                pad_size = self.block_size - actual_size
                x = np.pad(x, ((0, pad_size), (0, 0)), mode='constant')
                labels = np.pad(labels, (0, pad_size), mode='constant', constant_values=-1)

            # Transpose do (C, N) dla Conv1d
            x = x.T

            return torch.from_numpy(x), torch.from_numpy(labels)

        def _augment(self, coords):
            """Augmentacja danych"""
            # Random rotation wokol Z
            theta = np.random.uniform(0, 2 * np.pi)
            c, s = np.cos(theta), np.sin(theta)
            rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            coords = coords @ rot.T

            # Random jitter
            coords += np.random.normal(0, 0.01, coords.shape)

            # Random scale
            scale = np.random.uniform(0.9, 1.1)
            coords *= scale

            return coords.astype(np.float32)


class PointNetTrainer:
    """
    Trainer dla modelu PointNet
    """

    def __init__(
        self,
        config: PointNetConfig,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        device: str = None
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for PointNet training")

        self.config = config
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

        # Model
        self.model = PointNetSegmentation(config).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=20, gamma=0.5
        )

        # Class weights (mogą być aktualizowane)
        self.class_weights = None

    def train(
        self,
        coords: np.ndarray,
        labels: np.ndarray,
        colors: Optional[np.ndarray] = None,
        intensity: Optional[np.ndarray] = None,
        normals: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 8,
        val_split: float = 0.1,
        progress_callback=None
    ) -> Dict:
        """
        Trenuje model PointNet

        Args:
            coords: (N, 3) wspolrzedne punktow
            labels: (N,) etykiety klas
            colors: (N, 3) kolory RGB
            intensity: (N,) intensywnosc
            normals: (N, 3) wektory normalne
            epochs: liczba epok
            batch_size: rozmiar batcha
            val_split: procent danych walidacyjnych
            progress_callback: callback(step, pct, msg)

        Returns:
            Dict z metrykami treningu
        """
        n_points = len(coords)

        # Oblicz wagi klas (dla niezbalansowanych danych)
        unique_classes, counts = np.unique(labels, return_counts=True)
        class_weights = 1.0 / counts
        class_weights = class_weights / class_weights.sum() * len(unique_classes)

        # Mapuj klasy na 0..n-1
        label_mapping = {c: i for i, c in enumerate(unique_classes)}
        labels_mapped = np.array([label_mapping[l] for l in labels])

        # Weight tensor
        weight_tensor = torch.zeros(self.config.n_classes, device=self.device)
        for i, c in enumerate(unique_classes):
            weight_tensor[label_mapping[c]] = class_weights[i]

        # Split train/val
        indices = np.random.permutation(n_points)
        val_size = int(n_points * val_split)
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]

        # Datasets
        train_dataset = PointCloudDataset(
            coords[train_indices], labels_mapped[train_indices],
            colors[train_indices] if colors is not None else None,
            intensity[train_indices] if intensity is not None else None,
            normals[train_indices] if normals is not None else None,
            block_size=4096, augment=True
        )

        val_dataset = PointCloudDataset(
            coords[val_indices], labels_mapped[val_indices],
            colors[val_indices] if colors is not None else None,
            intensity[val_indices] if intensity is not None else None,
            normals[val_indices] if normals is not None else None,
            block_size=4096, augment=False
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        # Loss function
        criterion = nn.CrossEntropyLoss(weight=weight_tensor, ignore_index=-1)

        # Training loop
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        best_val_acc = 0
        best_state = None

        start_time = time.time()

        for epoch in range(epochs):
            # Train
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(batch_x)

                # Reshape dla loss
                logits = logits.permute(0, 2, 1).contiguous().view(-1, self.config.n_classes)
                batch_y = batch_y.view(-1)

                loss = criterion(logits, batch_y)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

                # Accuracy
                mask = batch_y != -1
                preds = logits.argmax(dim=1)
                train_correct += (preds[mask] == batch_y[mask]).sum().item()
                train_total += mask.sum().item()

            train_loss /= len(train_loader)
            train_acc = train_correct / train_total if train_total > 0 else 0

            # Validation
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)

                    logits = self.model(batch_x)
                    logits = logits.permute(0, 2, 1).contiguous().view(-1, self.config.n_classes)
                    batch_y = batch_y.view(-1)

                    loss = criterion(logits, batch_y)
                    val_loss += loss.item()

                    mask = batch_y != -1
                    preds = logits.argmax(dim=1)
                    val_correct += (preds[mask] == batch_y[mask]).sum().item()
                    val_total += mask.sum().item()

            val_loss /= len(val_loader)
            val_acc = val_correct / val_total if val_total > 0 else 0

            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)

            # Best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = self.model.state_dict().copy()

            # Scheduler
            self.scheduler.step()

            # Progress
            if progress_callback:
                pct = int((epoch + 1) / epochs * 100)
                progress_callback(
                    "PointNet Training",
                    pct,
                    f"Epoch {epoch+1}/{epochs}: Train={train_acc:.1%}, Val={val_acc:.1%}"
                )

            logger.info(
                f"Epoch {epoch+1}/{epochs}: "
                f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
            )

        # Przywroc najlepszy model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        elapsed = time.time() - start_time

        # Zapisz mapowanie klas
        self.label_mapping = label_mapping
        self.reverse_mapping = {v: k for k, v in label_mapping.items()}

        return {
            'history': history,
            'best_val_acc': best_val_acc,
            'training_time': elapsed,
            'n_classes': len(unique_classes),
            'classes': unique_classes.tolist()
        }

    def predict(
        self,
        coords: np.ndarray,
        colors: Optional[np.ndarray] = None,
        intensity: Optional[np.ndarray] = None,
        normals: Optional[np.ndarray] = None,
        batch_size: int = 8,
        return_confidence: bool = True,
        progress_callback=None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predykcja na nowych danych
        """
        self.model.eval()

        # Dummy labels
        dummy_labels = np.zeros(len(coords), dtype=np.int64)

        dataset = PointCloudDataset(
            coords, dummy_labels,
            colors, intensity, normals,
            block_size=4096, augment=False
        )

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        all_preds = []
        all_probs = []

        with torch.no_grad():
            for i, (batch_x, _) in enumerate(loader):
                batch_x = batch_x.to(self.device)

                logits = self.model(batch_x)
                probs = F.softmax(logits, dim=1)

                preds = logits.argmax(dim=1)

                all_preds.append(preds.cpu().numpy())
                if return_confidence:
                    all_probs.append(probs.max(dim=1)[0].cpu().numpy())

                if progress_callback:
                    pct = int((i + 1) / len(loader) * 100)
                    progress_callback("PointNet Inference", pct, f"Batch {i+1}/{len(loader)}")

        # Sklej i przywroc oryginalne etykiety
        predictions = np.concatenate(all_preds, axis=0).reshape(-1)[:len(coords)]
        predictions = np.array([self.reverse_mapping.get(p, p) for p in predictions])

        confidence = None
        if return_confidence:
            confidence = np.concatenate(all_probs, axis=0).reshape(-1)[:len(coords)]

        return predictions, confidence

    def save(self, path: str):
        """Zapisz model"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'model_state': self.model.state_dict(),
            'config': self.config,
            'label_mapping': getattr(self, 'label_mapping', {}),
            'reverse_mapping': getattr(self, 'reverse_mapping', {})
        }, path)

        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str, device: str = None) -> 'PointNetTrainer':
        """Wczytaj model"""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required to load PointNet model")

        checkpoint = torch.load(path, map_location='cpu')

        config = checkpoint['config']
        trainer = cls(config, device=device)
        trainer.model.load_state_dict(checkpoint['model_state'])
        trainer.label_mapping = checkpoint.get('label_mapping', {})
        trainer.reverse_mapping = checkpoint.get('reverse_mapping', {})

        logger.info(f"Model loaded from {path}")

        return trainer


# Convenience functions

def create_pointnet_model(
    n_classes: int,
    input_channels: int = 9,
    hidden_dims: List[int] = None,
    device: str = None
) -> PointNetTrainer:
    """
    Tworzy nowy model PointNet

    Args:
        n_classes: liczba klas do klasyfikacji
        input_channels: liczba kanalow wejsciowych (xyz + dodatkowe cechy)
        hidden_dims: wymiary warstw ukrytych
        device: urzadzenie (cuda/cpu)
    """
    config = PointNetConfig(
        n_classes=n_classes,
        input_channels=input_channels,
        hidden_dims=hidden_dims
    )
    return PointNetTrainer(config, device=device)


def is_torch_available() -> bool:
    """Sprawdza czy PyTorch jest dostepny"""
    return TORCH_AVAILABLE


def get_device_info() -> Dict:
    """Zwraca informacje o dostepnych urzadzeniach"""
    if not TORCH_AVAILABLE:
        return {'available': False, 'message': 'PyTorch not installed'}

    info = {
        'available': True,
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
    }

    if info['cuda_available']:
        info['current_device'] = torch.cuda.current_device()
        info['device_name'] = torch.cuda.get_device_name(0)
        info['memory_allocated'] = torch.cuda.memory_allocated(0)
        info['memory_cached'] = torch.cuda.memory_reserved(0)

    return info
