# Optimizing Heat ODE: Physics-Informed Neural ODEs for Spectrogram Classification

A PyTorch Lightning implementation of Physics-Informed Neural Networks (PINNs) combining the heat equation PDE with Neural ODEs for audio spectrogram classification.

## Overview

This project implements a differentiable PDE solver that models heat diffusion on spectrogram data using Neural ODEs. The approach treats spectrogram frequency evolution as a spatiotemporal diffusion process, learning physical parameters (thermal diffusivity, heat sources) end-to-end via gradient descent.

**Key Innovation**: Replace traditional CNN feature extraction with a physics-constrained ODE layer that enforces smooth, physically-interpretable temporal evolution of frequency features.

## Architecture

```
Input Spectrogram (128×60)
↓
Initial Condition u₀ (batch×freq×time)
↓
Heat ODE Solver (Neural ODE + PDE constraint)
↓
Learnable Linear Projection (weight matrix)
↓
Classification Output (15 classes)
```

## Core Components

### 1. Data Pipeline

#### Spectrogram Processing

* Resizes images to 128×128 using OpenCV
* Extracts frequency sums along time axis
* Creates balanced positive/negative pairs for training

```python
def get_sum(img_path):
    """Extract normalized frequency features from spectrogram"""
    image = resize(imread(img_path, 0), (128, 128)).astype(float32)
    return sum(image, axis=1) / (256 * 128)
```

#### Dataset Structure

```python
class EasyLoder(Dataset):
    def __init__(self, x, y, device):
        self.x = torch.from_numpy(x).to(device, dtype=torch.float32)
        self.y = torch.from_numpy(y).to(device, dtype=torch.long)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
```

---

### 2. Physics Model: Heat Equation PDE

The heat equation with a learnable source term:

[
\frac{\partial u}{\partial t} = \alpha \nabla^2 u + Q(t)
]

Where:

* ( \alpha ): learnable thermal diffusivity (per frequency band)
* ( Q(t) = Q_0 e^{-\lambda t} ): time-decaying heat source
* ( \nabla^2 u ): Laplacian computed via Finite Difference Method

```python
class Heat_Eq_PDE(torch.nn.Module):
    def __init__(self, dx, num_freqs):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.ones(num_freqs) * 0.1)
        self.Q0 = torch.nn.Parameter(torch.ones(num_freqs) * 1)
        self.lamb = torch.nn.Parameter(torch.ones(num_freqs) * 0.001)
        self.dx = dx

    def laplace_FDM(self, u, delta):
        """Finite Difference Method for Laplacian (centered difference)"""
        u = torch.nn.functional.pad(u, (1, 1), mode='replicate')
        du = (u[:, :, 2:] - 2 * u[:, :, 1:-1] + u[:, :, :-2]) / (delta**2)
        return du

    def forward(self, t, u):
        u = self.alpha[:, None] * u
        du2_dx2 = self.laplace_FDM(u, self.dx)
        du_dt = du2_dx2 + self.Q(t)
        return du_dt
```

---

### 3. Neural ODE Integration

Uses `torchdiffeq` for adaptive ODE solving with adjoint method for memory-efficient backpropagation.

```python
class HeatODE(pl.LightningModule):
    def __init__(self, ode_params, input_shape=[128, 60], num_classes=15):
        super().__init__()
        self.time = torch.linspace(
            ode_params['time']['min'], 
            ode_params['time']['max'], 
            steps=ode_params['time']['time_steps']
        )
        self.heat_model = Heat_Eq_PDE(self.dx, input_shape[0])
        self.weight = torch.nn.Parameter(
            torch.randn(input_shape[1], input_shape[0])
        )

    def forward(self, u0):
        solution = odeint(
            self.heat_model,
            u0,
            self.time,
            rtol=self.rtol, 
            atol=self.atol, 
            method=self.method
        )[-1]
        solution = solution @ self.weight
        return torch.sum(solution, dim=0)
```

---

### 4. Gaussian Filtering (Preprocessing)

```python
class GaussianFilter(pl.LightningModule):
    def __init__(self, sigma_list, mode='reflect'):
        self.sigma_list = sigma_list
        self.kernel_dict = {}
        for sigma in sigma_list:
            kernel_size = max(1, int(2 * ceil(3 * sigma) + 1))
            kernel = precompute_gaussian_kernel(kernel_size, sigma)
            self.kernel_dict[sigma] = kernel
```

---

## Configuration

```python
ode_params = {
    "time": {
        "min": 0.0, 
        "max": 1.0, 
        "time_steps": 60
    },
    "dx": 0.1,
    "sol": {
        "rtol": 1e-5,
        "atol": 1e-12,
        "method": "rk4"
    },
    "sigmas": [0.5, 1.0, 1.5, 2.0, 2.5]
}
```

---

## Training

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
criterion = torch.nn.CrossEntropyLoss()

def train(dataset, epoch, num_epochs):
    train_bar = tqdm(dataset)
    for image, label in train_bar:
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
```

---

## Dependencies

```bash
pip install torchdiffeq torchsummary pytorch-lightning
```

---

## Applications

* Audio Classification
* Physics-Constrained ML
* Scientific ML

---

## Citation

* Neural ODEs (NeurIPS 2018)
* Physics-Informed Learning (2017)
