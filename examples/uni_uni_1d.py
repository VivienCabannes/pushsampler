"""
NOTE - TODO
-----------
This does not work so well when we modify the value of a and b.
It would be worth looking at the loss landscape when `linear_net` is True, and understanding the descent dynamics.
"""
import torch
import torch.optim as optim
import torch.nn.functional as F

from pushsampler.utils import set_all_seed
from pushsampler.architecture import LinearMap, MLP


set_all_seed(42)
linear_net = False


def uniform_mgf(t, a, b):
    out = torch.exp(b * t) - torch.exp(a * t)
    out /= t * (b - a)
    return out


a, b = 3, 5

nb_epochs = 5000
batch_size = 10
contrastive = True

if linear_net:
    net = LinearMap(fan_in=1, fan_out=1)
else:
    net = MLP([1, 100, 100, 1], layer_norm="none")
optimizer = optim.SGD(net.parameters(), lr=1e-2, momentum=0)
# Linear scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=nb_epochs // 3, gamma=0.1)

T = torch.linspace(0, 1, 100)[1:]
for i in range(nb_epochs):
    optimizer.zero_grad()

    x = torch.rand(2 * batch_size, 1)

    # Generative moment differences
    out = net(x)
    with torch.no_grad():
        thres = 1 / (1 + F.relu(out.max()))
    t = thres * T
    out = torch.exp(t * net(x))
    out = out - uniform_mgf(t, a, b)

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
            with torch.no_grad():
                y = net(torch.linspace(0, 1, 2).unsqueeze(1))
            # tmp = torch.linspace(0, 1, 100)[1:]
            test_loss = torch.mean(
                (uniform_mgf(t, y[0], y[1]) - uniform_mgf(t, a, b)) ** 2
            )
            other_measure = torch.sqrt(((y.min() - a) ** 2 + (y.max() - b) ** 2) / 2)
            print(
                f"Epoch {i:5} - Loss: {loss.item():.1e} - Test loss: {test_loss.item():.1e}"
                f"- Distance: {other_measure.item():.1e}"
            )

    loss.backward()
    optimizer.step()

with torch.no_grad():
    y = net(torch.linspace(0, 1, 2).unsqueeze(1))
print(y)
