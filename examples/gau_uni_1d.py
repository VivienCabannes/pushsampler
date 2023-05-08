import matplotlib.pyplot as plt
import numpy as np

from scipy.special import erf
import torch
import torch.optim as optim

from pushsampler.utils import set_all_seed
from pushsampler.architecture import LinearMap, MLP

plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{mathtools}")
plt.rc("font", size=10, family="serif", serif="cm")
plt.rc("figure", figsize=(2, 1.5))


def uniform_cf(t: torch.Tensor, a: float, b: float) -> torch.Tensor:
    """
    Compute uniform characteristic function

    Parameters
    ----------
    t: variable of the characteristic function of size `T`
    a: segment lower extremity
    b: segment upper extremity

    Returns
    -------
    out: tensor of size `2 x T`, corresponding to the real and imaginary part of the characteristic function
    """

    out = torch.stack(
        (torch.sin(t * b) - torch.sin(t * a), torch.cos(t * a) - torch.cos(t * b))
    )
    # out = torch.sin(t * b) - torch.sin(t * a)
    out /= t * (b - a)
    return out


set_all_seed(42)

a, b = 0, 1

nb_epochs = 5000
batch_size = 10
contrastive = True

linear_net = False

if linear_net:
    net = LinearMap(fan_in=1, fan_out=1)
else:
    net = MLP([1, 200, 200, 2000, 200, 1], layer_norm="none")
    # net = MLP([1, 200, 200, 20, 1], layer_norm="none")

# optimizer = optim.SGD(net.parameters(), lr=1e-1, momentum=0)
optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
# Linear scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=nb_epochs // 3, gamma=0.1)

# x = torch.rand(2 * batch_size, 1)
t = torch.linspace(0.01, 10, 100)
for i in range(nb_epochs):
    optimizer.zero_grad()

    x = torch.randn(2 * batch_size, 1)

    # Generative moment differences
    out = net(x)
    out = torch.exp(out) / (1 + torch.exp(out))

    # t = (2 * torch.rand(100) - 1)
    out = torch.stack((torch.cos(t * out), torch.sin(t * out)), dim=1)
    # out = torch.cos(t * out)
    out = out - uniform_cf(t, a, b)

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
        print(f"Epoch {i:5} - Loss is NaN or Inf")
        continue
    if i % (nb_epochs // 100) == 0:
        print(f"Epoch {i:5} - Loss: {loss.item():.1e}")

    loss.backward()
    optimizer.step()


x = torch.linspace(-5, 5, 100).unsqueeze(1)
net.eval()
with torch.no_grad():
    y = net(x).squeeze(1).numpy()
    y = np.exp(y) / (1 + np.exp(y))
x = x.numpy().squeeze()

# CDF
fig, ax = plt.subplots()
a, = ax.plot(x / np.sqrt(2), 2 * y - 1)
b, = ax.plot(x / np.sqrt(2), erf(x / np.sqrt(2)), "--")
ax.grid()
fig.savefig("cdf.pdf", bbox_inches="tight")

fig, ax = plt.subplots()
ax.legend([a, b], ["learned CDF", "ground truth"])
ax.axis("off")
fig.savefig("cdf_leg.pdf", bbox_inches="tight")

# # Quantile function
# fig, ax = plt.subplots(figsize=(10, 5))
# ax.plot(y, x)
# ax.grid()
