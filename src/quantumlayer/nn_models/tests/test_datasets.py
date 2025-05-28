import pytest
import numpy as np
from datasets import iris, spiral, mnist_digits
from datasets import DatasetMetadata

def _validate_metadata(metadata):
    """Helper function to test metadata structure"""
    assert isinstance(metadata, DatasetMetadata)
    assert isinstance(metadata.name, str)
    assert isinstance(metadata.description, str)
    assert isinstance(metadata.num_instances, int)
    assert isinstance(metadata.features, list)
    assert len(metadata.features) > 0
    
    if metadata.subset:
        assert metadata.subset in ['train', 'test', 'val']
    
    if metadata.num_features:
        assert isinstance(metadata.num_features, int)
        assert metadata.num_features > 0

def _validate_data(X, y, expected_shape=None):
    """Helper function to test data structure"""
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert len(X) == len(y)
    if expected_shape:
        assert X.shape[1] == expected_shape[1]

class TestIrisDataset:
    def test_train_data(self):
        X, y, metadata = iris.get_data_train()
        _validate_metadata(metadata)
        _validate_data(X, y, expected_shape=(None, 4))
        
        # Test specific to Iris train set
        assert metadata.subset == 'train'
        assert metadata.num_classes == 3
        assert len(metadata.features) == 4
        assert metadata.normalization.method == 'min-max'
        assert metadata.normalization.range == (0, 1)
        
        # Test normalization
        assert np.all(X >= 0) and np.all(X <= 1)

    def test_test_data(self):
        X, y, metadata = iris.get_data_test()
        _validate_metadata(metadata)
        _validate_data(X, y, expected_shape=(None, 4))
        assert metadata.subset == 'test'

class TestSpiralDataset:
    def test_default_parameters(self):
        X, y, metadata = spiral.get_data()
        _validate_metadata(metadata)
        _validate_data(X, y, expected_shape=(None, 10))
        
        assert len(np.unique(y)) == 3
        assert X.shape[1] == 10
        assert metadata.num_classes == 3
        assert metadata.num_features == 10
        assert 'non-linear' in metadata.characteristics
        
    def test_custom_parameters(self):
        num_instances = 2000
        num_features = 15
        num_classes = 4
        
        X, y, metadata = spiral.get_data(
            num_instances=num_instances,
            num_features=num_features,
            num_classes=num_classes
        )
        
        _validate_metadata(metadata)
        _validate_data(X, y, expected_shape=(None, num_features))
        
        assert len(np.unique(y)) == num_classes
        assert X.shape[1] == num_features
        assert metadata.num_classes == num_classes
        assert metadata.num_features == num_features
        
    def test_reproducibility(self):
        X1, y1, _ = spiral.get_data(random_seed=42)
        X2, y2, _ = spiral.get_data(random_seed=42)
        
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

class TestMNISTDataset:
    def test_original_train(self):
        X, y, metadata = mnist_digits.get_data_train_original()
        _validate_metadata(metadata)
        _validate_data(X, y)
        
        assert metadata.subset == 'train'
        assert metadata.num_classes == 10
        assert X.shape[1:] == (28, 28)  # Image dimensions
        assert np.all(X >= 0) and np.all(X <= 255)
        
    def test_original_test(self):
        X, y, metadata = mnist_digits.get_data_test_original()
        _validate_metadata(metadata)
        _validate_data(X, y)
        
        assert metadata.subset == 'test'
        assert metadata.num_classes == 10
        assert X.shape[1:] == (28, 28)
        
    def test_percevalquest_train(self):
        X, y, metadata = mnist_digits.get_data_train_percevalquest()
        _validate_metadata(metadata)
        _validate_data(X, y)
        
        assert metadata.subset == 'train'
        assert metadata.name == "MNIST Subset for First Perceval Quest"
        assert metadata.num_classes == 10
        assert len(X) == 6000  # Specific to Perceval Quest train set
        
    def test_percevalquest_test(self):
        X, y, metadata = mnist_digits.get_data_test_percevalquest()
        _validate_metadata(metadata)
        _validate_data(X, y)
        
        assert metadata.subset == 'val'
        assert metadata.name == "MNIST Subset for First Perceval Quest"
        assert len(X) == 600  # Specific to Perceval Quest validation set

if __name__ == "__main__":
    pytest.main([__file__])