import pytest
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def sample_batch():
    batch_size = 2
    channels = 3
    height, width = 224, 224
    images = torch.randn(batch_size, channels, height, width)
    masks = torch.randint(0, 2, (batch_size, 1, height, width)).float()
    return images, masks


@pytest.fixture
def temp_image_dir(tmp_path):
    img_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    for i in range(4):
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        mask = np.random.randint(0, 2, (256, 256), dtype=np.uint8) * 255
        from PIL import Image
        Image.fromarray(img).save(img_dir / f"test_{i}.jpg")
        Image.fromarray(mask).save(mask_dir / f"test_{i}.png")

    return tmp_path