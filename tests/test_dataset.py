import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import (
    MedicalSegmentationDataset,
    get_train_transforms,
    get_val_transforms,
    get_test_transforms
)


class TestTransforms:
    def test_train_transforms(self):
        transform = get_train_transforms((224, 224), augment=True)
        assert transform is not None
    
    def test_train_transforms_no_augment(self):
        transform = get_train_transforms((224, 224), augment=False)
        assert transform is not None
    
    def test_val_transforms(self):
        transform = get_val_transforms((224, 224))
        assert transform is not None
    
    def test_test_transforms(self):
        transform = get_test_transforms((224, 224))
        assert transform is not None


class TestMedicalSegmentationDataset:
    def test_dataset_loading(self, temp_image_dir):
        transform = get_val_transforms((224, 224))
        dataset = MedicalSegmentationDataset(
            root_dir=str(temp_image_dir),
            split='train',
            transform=transform,
            image_size=(224, 224)
        )
        
        assert len(dataset) == 4
    
    def test_dataset_getitem(self, temp_image_dir):
        transform = get_val_transforms((224, 224))
        dataset = MedicalSegmentationDataset(
            root_dir=str(temp_image_dir),
            split='train',
            transform=transform,
            image_size=(224, 224)
        )
        
        sample = dataset[0]
        
        assert 'image' in sample
        assert 'mask' in sample
        assert 'filename' in sample
        assert sample['image'].shape == (3, 224, 224)
        assert sample['mask'].shape == (1, 224, 224)
    
    def test_dataset_multi_split(self, tmp_path):
        for split in ['train', 'val', 'test']:
            img_dir = tmp_path / split / 'images'
            mask_dir = tmp_path / split / 'masks'
            img_dir.mkdir(parents=True)
            mask_dir.mkdir(parents=True)
            
            from PIL import Image
            import numpy as np
            img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            mask = np.random.randint(0, 2, (256, 256), dtype=np.uint8) * 255
            Image.fromarray(img).save(img_dir / 'test.jpg')
            Image.fromarray(mask).save(mask_dir / 'test.png')
        
        dataset = MedicalSegmentationDataset(
            root_dir=str(tmp_path),
            split='test',
            transform=get_test_transforms((224, 224)),
            image_size=(224, 224)
        )
        
        assert len(dataset) == 1
    
    def test_dataset_no_split(self, tmp_path):
        img_dir = tmp_path / 'images'
        mask_dir = tmp_path / 'masks'
        img_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)
        
        from PIL import Image
        import numpy as np
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        mask = np.random.randint(0, 2, (256, 256), dtype=np.uint8) * 255
        Image.fromarray(img).save(img_dir / 'test.jpg')
        Image.fromarray(mask).save(mask_dir / 'test.png')
        
        dataset = MedicalSegmentationDataset(
            root_dir=str(tmp_path),
            split='train',
            transform=get_val_transforms((224, 224)),
            image_size=(224, 224)
        )
        
        assert len(dataset) == 1


class TestDatasetEdgeCases:
    def test_binary_mask_conversion(self, temp_image_dir):
        transform = get_val_transforms((224, 224))
        dataset = MedicalSegmentationDataset(
            root_dir=str(temp_image_dir),
            split='train',
            transform=transform,
            image_size=(224, 224)
        )
        
        sample = dataset[0]
        mask = sample['mask']
        
        assert mask.max() <= 1.0
        assert mask.min() >= 0.0
    
    def test_invalid_root_dir(self):
        with pytest.raises(FileNotFoundError):
            dataset = MedicalSegmentationDataset(
                root_dir='/invalid/path',
                split='train',
                transform=get_val_transforms((224, 224)),
                image_size=(224, 224)
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])