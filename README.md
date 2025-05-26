# PyTorch CUDA Template

A modern Python template for GPU-accelerated machine learning projects using PyTorch 2.7+ with CUDA 12.8 support.

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7%2B-red.svg)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Features

- **Modern PyTorch**: Latest PyTorch 2.7+ with optimized CUDA 12.8 support
- **GPU-Ready**: Pre-configured for CUDA acceleration with automatic fallback to CPU
- **Development Tools**: Integrated linting, formatting, testing, and type checking
- **Flexible Dependencies**: Optional dependency groups for different use cases
- **Best Practices**: Following modern Python packaging and development standards

## 📋 Requirements

- **Python**: 3.11 or higher
- **CUDA**: 12.8 (for GPU acceleration)
- **GPU**: Compatible NVIDIA GPU (optional, falls back to CPU)

## 🛠️ Installation

### Quick Start

```bash
# Clone the template
git clone https://github.com/bjoernbethge/torch-cuda.git
cd torch-cuda

# Install with uv (recommended)
uv sync

```

### Installation Options

Choose the dependencies that match your needs:

```bash
# Basic installation (PyTorch v2.7.0+cu128 only)
uv sync

# Development environment
uv sync --extra dev

# Data science toolkit Machine learning utilities
uv sync --extra extras

# Everything included
uv sync --extra all
```

### Manual PyTorch Installation

If you need a specific PyTorch configuration:

```bash
# Install PyTorch with CUDA 12.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install PyTorch CPU-only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## 🏗️ Project Structure

```
torch-cuda/
├── src/
│   └── torch_cuda/          # Main package
│       ├── __init__.py
│       ├── models/           # Model definitions
│       ├── data/            # Data loading and processing
│       ├── training/        # Training utilities
│       └── utils/           # Helper functions
├── tests/                   # Test suite
├── notebooks/               # Jupyter notebooks
├── scripts/                 # Training and inference scripts
├── configs/                 # Configuration files
├── pyproject.toml          # Project configuration
└── README.md               # This file
```

## 🚀 Quick Start Guide

### 1. Verify CUDA Installation

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Device count: {torch.cuda.device_count()}")
```

### 2. Basic Usage Example

```python
import torch
import torch.nn as nn

# Automatically use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create a simple model
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
).to(device)

# Create sample data
x = torch.randn(32, 10).to(device)
y = model(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {y.shape}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
```

### 3. Training Example

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = nn.Linear(10, 1).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Sample data
X = torch.randn(1000, 10)
y = torch.randn(1000, 1)
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop
model.train()
for epoch in range(10):
    total_loss = 0
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/10, Loss: {total_loss/len(dataloader):.4f}")
```

## 🧪 Development

### Setup Development Environment

```bash
# Install development dependencies
uv sync --extra dev

# Install pre-commit hooks
pre-commit install
```

### Code Quality Tools

```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint code
flake8 src/ tests/
ruff check src/ tests/

# Type checking
mypy src/

# Run tests
pytest

# Run tests with coverage
pytest --cov=src --cov-report=html
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_models.py

# Run with verbose output
pytest -v
```

## 📊 Performance Tips

### CUDA Optimization

1. **Use appropriate data types**:
   ```python
   # Use float16 for memory efficiency (with mixed precision)
   model = model.half()  # or use torch.cuda.amp
   ```

2. **Optimize data loading**:
   ```python
   dataloader = DataLoader(
       dataset, 
       batch_size=32, 
       num_workers=4,  # Parallel data loading
       pin_memory=True  # Faster GPU transfer
   )
   ```

3. **Enable optimizations**:
   ```python
   # Compile model for better performance (PyTorch 2.0+)
   model = torch.compile(model)
   
   # Enable cuDNN benchmarking
   torch.backends.cudnn.benchmark = True
   ```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`pytest`)
6. Format your code (`black . && isort .`)
7. Commit your changes (`git commit -m 'Add amazing feature'`)
8. Push to the branch (`git push origin feature/amazing-feature`)
9. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [PyTorch Team](https://pytorch.org/) for the incredible framework
- [NVIDIA](https://developer.nvidia.com/cuda-toolkit) for CUDA toolkit
- [Astral](https://astral.sh/) for the amazing uv package manager

## 🆘 Troubleshooting

### Common Issues

**CUDA Out of Memory**:
```python
# Reduce batch size or use gradient accumulation
# Clear cache periodically
torch.cuda.empty_cache()
```

**Slow Data Loading**:
```python
# Increase num_workers in DataLoader
# Use pin_memory=True for GPU training
# Consider using faster storage (SSD)
```

**Import Errors**:
```bash
# Ensure proper installation
uv sync --extra all
# Or reinstall PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### Getting Help

- Check the [Issues](https://github.com/bjoernbethge/torch-cuda/issues) page
- Review [PyTorch Documentation](https://pytorch.org/docs/)
- Visit [PyTorch Forums](https://discuss.pytorch.org/)

---

**Happy coding with PyTorch! 🔥**