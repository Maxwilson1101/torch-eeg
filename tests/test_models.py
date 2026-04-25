import unittest
import torch
from models import BandSpatialCNN, TopoCNN, FactorizedCNN


class TestBandSpatialCNN(unittest.TestCase):
    def test_output_shape(self):
        out = BandSpatialCNN()(torch.randn(4, 62, 5))
        assert out.shape == (4, 5)

    def test_backward(self):
        model = BandSpatialCNN()
        loss = model(torch.randn(4, 62, 5)).sum()
        loss.backward()
        assert any(p.grad is not None for p in model.parameters())


class TestTopoCNN(unittest.TestCase):
    def test_output_shape(self):
        out = TopoCNN()(torch.randn(4, 62, 5))
        assert out.shape == (4, 5)

    def test_backward(self):
        model = TopoCNN()
        loss = model(torch.randn(4, 62, 5)).sum()
        loss.backward()
        assert any(p.grad is not None for p in model.parameters())


class TestFactorizedCNN(unittest.TestCase):
    def test_output_shape(self):
        out = FactorizedCNN()(torch.randn(4, 62, 5))
        assert out.shape == (4, 5)

    def test_backward(self):
        model = FactorizedCNN()
        loss = model(torch.randn(4, 62, 5)).sum()
        loss.backward()
        assert any(p.grad is not None for p in model.parameters())


if __name__ == "__main__":
    unittest.main()
