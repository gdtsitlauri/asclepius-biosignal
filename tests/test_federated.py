"""Tests for Module 7 Federated Learning."""
import numpy as np
import pytest
import torch

from asclepius.module7_federated.federated import (
    FederatedClient, FederatedServer, fedavg_aggregate,
    DPGaussianMechanism, simulate_hospital_split,
)
from asclepius.module1_per_signal.models import build_model


@pytest.fixture
def simple_model():
    return build_model("cnn1d", in_channels=2, n_classes=2, hidden=32)


def make_data(n=100, ch=2, win=64, n_cls=2):
    X = np.random.randn(n, ch, win).astype(np.float32)
    y = np.random.randint(0, n_cls, n).astype(np.int64)
    return X, y


def test_fedavg(simple_model):
    import copy
    clients = [copy.deepcopy(simple_model) for _ in range(3)]
    sizes = [100, 200, 150]
    aggregated = fedavg_aggregate(simple_model, clients, sizes)
    assert aggregated is simple_model


def test_dp_mechanism(simple_model):
    X, y = make_data(16)
    criterion = torch.nn.CrossEntropyLoss()
    out = simple_model(torch.from_numpy(X))
    loss = criterion(out, torch.from_numpy(y))
    loss.backward()
    dp = DPGaussianMechanism(noise_multiplier=1.1, max_grad_norm=1.0)
    dp.clip_and_noise(simple_model)


def test_hospital_split():
    X, y = make_data(500, n_cls=3)
    splits = simulate_hospital_split(X, y, n_hospitals=5)
    assert len(splits) == 5
    for Xh, yh in splits:
        assert len(Xh) > 0


def test_federated_server_mini(simple_model):
    X, y = make_data(200)
    splits = simulate_hospital_split(X, y, n_hospitals=3)
    clients = [
        FederatedClient(i, Xh[20:], yh[20:], Xh[:20], yh[:20], local_epochs=1)
        for i, (Xh, yh) in enumerate(splits)
    ]
    server = FederatedServer(
        global_model=simple_model,
        clients=clients,
        rounds=2,
        dp_enabled=True,
        noise_multiplier=0.5,
    )
    result = server.run()
    assert "final_metrics" in result
    assert result["n_rounds"] == 2
