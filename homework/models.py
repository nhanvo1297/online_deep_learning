from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


# Loss function for classification
class ClassificationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (b, num_classes) raw scores from model
            target: (b,) integer labels
        """
        return self.loss_fn(logits, target)


# Loss function for detection (segmentation + depth)
class DetectionLoss(nn.Module):
    def __init__(self, lambda_depth: float = 0.2):
        super().__init__()
        # Weight classes: background=1.0, left_lane=2.0, right_lane=2.0
        # This penalizes misclassification of lanes more heavily
        class_weights = torch.tensor([1.0, 2.0, 2.0])
        self.segmentation_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.depth_loss = nn.L1Loss()
        self.lambda_depth = lambda_depth

    def forward(
        self, 
        logits: torch.Tensor, 
        depth: torch.Tensor,
        target_logits: torch.Tensor,
        target_depth: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits: (b, num_classes, h, w) segmentation logits
            depth: (b, h, w) predicted depth
            target_logits: (b, h, w) target segmentation labels
            target_depth: (b, h, w) target depth
        """
        seg_loss = self.segmentation_loss(logits, target_logits)
        depth_loss_val = self.depth_loss(depth, target_depth)
        return seg_loss + self.lambda_depth * depth_loss_val

class Classifier(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 6,
    ):
        """
        A convolutional network for image classification.

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        # Simple fully connected layers
        self.fc1 = nn.Linear(in_channels * 64 * 64, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, h, w) image

        Returns:
            tensor (b, num_classes) logits
        """
        # Normalize the input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # Flatten
        z = z.view(z.size(0), -1)
        
        # Fully connected layers
        z = self.relu(self.fc1(z))
        z = self.dropout(z)
        z = self.relu(self.fc2(z))
        z = self.dropout(z)
        logits = self.fc3(z)

        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Used for inference, returns class labels
        This is what the AccuracyMetric uses as input (this is what the grader will use!).
        You should not have to modify this function.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            pred (torch.LongTensor): class labels {0, 1, ..., 5} with shape (b, h, w)
        """
        return self(x).argmax(dim=1)


class Detector(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
    ):
        """
        A single model that performs segmentation and depth regression

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        # Down-sampling blocks
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16)
        )
        
        self.down2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32)
        )
        
        # Up-sampling blocks with skip connections
        # After up1: concatenate with down1 features (16 channels) -> 32 total channels
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16)
        )
        # Merge down1 skip (16 channels) with up1 output (16 channels) -> 32 channels
        self.merge1 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16)
        )
        
        # After up2: concatenate with original input features (3 channels) -> 19 total channels
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16)
        )
        # Merge input skip (3 channels) with up2 output (16 channels) -> 19 channels
        self.merge2 = nn.Sequential(
            nn.Conv2d(19, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16)
        )
        
        # Output heads
        self.segmentation_head = nn.Conv2d(16, num_classes, kernel_size=1)
        self.depth_head = nn.Conv2d(16, 1, kernel_size=1)
        

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used in training, takes an image and returns raw logits and raw depth.
        This is what the loss functions use as input.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.FloatTensor, torch.FloatTensor):
                - logits (b, num_classes, h, w)
                - depth (b, h, w)
        """
        # Normalize the input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # Down-sampling (save intermediate features for skip connections)
        down1_feat = self.down1(z)  # (b, 16, h/2, w/2)
        down2_feat = self.down2(down1_feat)  # (b, 32, h/4, w/4)
        
        # Up-sampling with skip connections
        up1_feat = self.up1(down2_feat)  # (b, 16, h/2, w/2)
        # Concatenate with down1 skip connection
        up1_feat = torch.cat([up1_feat, down1_feat], dim=1)  # (b, 32, h/2, w/2)
        up1_feat = self.merge1(up1_feat)  # (b, 16, h/2, w/2)
        
        up2_feat = self.up2(up1_feat)  # (b, 16, h, w)
        # Concatenate with input skip connection
        up2_feat = torch.cat([up2_feat, z], dim=1)  # (b, 19, h, w)
        up2_feat = self.merge2(up2_feat)  # (b, 16, h, w)
        
        # Output heads
        logits = self.segmentation_head(up2_feat)  # (b, 3, h, w)
        raw_depth = self.depth_head(up2_feat)      # (b, 1, h, w)
        raw_depth = raw_depth.squeeze(1)    # (b, h, w)

        return logits, raw_depth

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used for inference, takes an image and returns class labels and normalized depth.
        This is what the metrics use as input (this is what the grader will use!).

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.LongTensor, torch.FloatTensor):
                - pred: class labels {0, 1, 2} with shape (b, h, w)
                - depth: normalized depth [0, 1] with shape (b, h, w)
        """
        logits, raw_depth = self(x)
        pred = logits.argmax(dim=1)

        # Optional additional post-processing for depth only if needed
        depth = raw_depth

        return pred, depth


MODEL_FACTORY = {
    "classifier": Classifier,
    "detector": Detector,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def debug_model(batch_size: int = 1):
    """
    Test your model implementation

    Feel free to add additional checks to this function -
    this function is NOT used for grading
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_batch = torch.rand(batch_size, 3, 64, 64).to(device)

    print(f"Input shape: {sample_batch.shape}")

    model = load_model("classifier", in_channels=3, num_classes=6).to(device)
    output = model(sample_batch)

    # should output logits (b, num_classes)
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    debug_model()
