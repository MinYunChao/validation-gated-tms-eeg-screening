import torch

from tms_eeg_screening.config import ModelConfig
from tms_eeg_screening.model import CleanSTGNN


def test_model_forward_and_gradients():
    config = ModelConfig(
        patch_size=32,
        embedding_dim=16,
        graph_hidden_dim=12,
        latent_dim=10,
        window_sizes=[2],
        window_strides=[1],
        dropout=0.0,
    )
    model = CleanSTGNN(config)
    signal = torch.randn(5, 8, 128)
    logits, latent = model(signal, return_features=True)
    assert logits.shape == (5, 2)
    assert latent.shape == (5, 10)
    logits.sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert model.trainable_parameters > 0
