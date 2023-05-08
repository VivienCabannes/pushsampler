"""
Once again this does not work really well when we modify the value of mu and sigma, or when the network is not linear.
We could look at the loss landscape when `linear_net` is True, and understand a bit better the descent dynamics.
"""

import torch
import torch.optim as optim

from pushsampler.utils import set_all_seed
from pushsampler.architecture import LinearMap, MLP


def gaussian_cf(t: torch.Tensor, mu: float, sigma: float) -> torch.Tensor:
    """
    Compute Gaussian characteristic function

    Parameters
    ----------
    t: variable of the characteristic function of size `T`
    mu: mean of the Gaussian
    sigma: standard deviation of the Gaussian

    Returns
    -------
    out: tensor of size `2 x T`, corresponding to the real and imaginary part of the characteristic function
    """
    out = torch.stack((torch.cos(t * mu), torch.sin(t * mu)))
    out *= torch.exp(-0.5 * t**2 * sigma**2)
    return out


set_all_seed(42)

mu, sigma = 3, 2
# mu, sigma = 0, 20

nb_epochs = 5000
batch_size = 100
contrastive = True

linear_net = True

if linear_net:
    net = LinearMap(fan_in=1, fan_out=1)
else:
    net = MLP([1, 100, 100, 100, 1], layer_norm="none")
# optimizer = optim.SGD(net.parameters(), lr=1e-1, momentum=0)
optimizer = optim.SGD(net.parameters(), lr=1e-2, momentum=0)
# Linear scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=nb_epochs // 3, gamma=0.1)

# x = torch.rand(2 * batch_size, 1)
# t = torch.linspace(0, 1, 100)[1:]
for i in range(nb_epochs):
    optimizer.zero_grad()

    x = torch.randn(2 * batch_size, 1)

    # Generative moment differences
    # out = mu + sigma * x
    out = net(x)
    t = torch.linspace(-1, 1, 100)
    out = torch.stack((torch.cos(t * out), torch.sin(t * out)), dim=1)
    out = out - gaussian_cf(t, mu, sigma)

    if not contrastive:
        # Minibatch type
        out = torch.mean(out, dim=0)
        loss = out**2
    else:
        # Constrastive type
        out_1, out_2 = torch.mean(out[:batch_size], dim=0), torch.mean(
            out[batch_size:], dim=0
        )
        loss = out_1 * out_2

    # Average error
    loss = torch.mean(loss)

    if loss.isnan() or loss.isinf():
        continue
    if i % (nb_epochs // 100) == 0:
        with torch.no_grad():
            y = net(torch.linspace(0, 1, 2).unsqueeze(1))
        other_measure = ((y.min() - mu) ** 2 + (y.max() - mu - sigma) ** 2) / 2
        print(
            f"Epoch {i:5} - Loss: {loss.item():.1e} "
            f"- Distance: {other_measure.item():.1e}"
        )

    loss.backward()
    optimizer.step()
