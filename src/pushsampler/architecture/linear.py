import torch
import torch.nn as nn


class LinearMap(nn.Module):
    """
    Affine mapping architecture.

    Parameters
    ----------
    fan_in: Dimension of input.
    fan_out: Dimension of output.
    device: Computation device
    dtype: Type of parameters. Default is None, falling back to torch.float (32 bits).
    """

    def __init__(
        self,
        fan_in: int = 1,
        fan_out: int = 1,
        device: str = "cpu",
        dtype: type = None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.fc = nn.Linear(fan_in, fan_out, bias=True, **factory_kwargs)

    def forward(self, x: torch.Tensor):
        return self.fc(x)
