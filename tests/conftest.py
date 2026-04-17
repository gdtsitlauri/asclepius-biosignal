import pytest
import numpy as np
import torch
from asclepius.utils import set_seed

@pytest.fixture(autouse=True)
def fix_seed():
    set_seed(42)
