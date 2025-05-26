# ⚡ PyTorch CUDA Template

<div align="center">

<!-- Badges -->
[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7%2B-red.svg)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![uv](https://img.shields.io/badge/uv-enabled-brightgreen.svg)](https://github.com/astral-sh/uv)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<!-- Logo/Header -->
<h3>🚀 A blazing-fast Python template for GPU-accelerated machine learning</h3>

*Harness the full power of modern PyTorch with CUDA 12.8 acceleration* 🔥

</div>

---

## 🌟 Overview

**PyTorch CUDA Template** provides everything you need to jumpstart your GPU-accelerated machine learning projects. Built with modern Python packaging standards and optimized for **PyTorch 2.7+** with **CUDA 12.8** support, this template eliminates setup friction so you can focus on building amazing models.

### 🎯 Key Features

- 🔥 **Cutting-Edge PyTorch** - Latest PyTorch 2.7+ with optimized CUDA 12.8 support
- ⚡ **GPU-Ready Architecture** - Pre-configured CUDA acceleration with intelligent CPU fallback
- 🛠️ **Modern Development Stack** - Integrated linting, formatting, testing, and type checking
- 📊 **ML Ops Ready** - MLflow experiment tracking and Polars for high-performance data processing
- 🚀 **Lightning-Fast Setup** - Powered by `uv` for blazing-fast dependency resolution
- 🏗️ **Production-Ready Structure** - Following modern Python packaging best practices

---

## 📋 Requirements

- 🐍 **Python** ≥ 3.11
- 🎮 **CUDA** 12.8 (for GPU acceleration)
- 💻 **GPU** Compatible NVIDIA GPU (optional, gracefully falls back to CPU)
- ⚡ **uv** Package manager (recommended for fastest installs)

---

## 🚀 Installation

### ⚡ Lightning-Fast Setup

```bash
# Clone the template
git clone https://github.com/bjoernbethge/torch-cuda.git
cd torch-cuda

# Install everything with uv (recommended)
uv sync
```

### 🎛️ Customized Installation

Choose exactly what you need:

```bash
# 🔥 Basic PyTorch setup
uv sync

# 🧪 Development environment (testing, linting, formatting)
uv sync --extra dev

# 📊 ML Ops toolkit (MLflow, Polars, Plotly, profiling tools)
uv sync --extra extras

# 🌟 Everything included (the full experience)
uv sync --extra all
```

---

## 🚀 Quick Start Guide

### 1. 🔍 Verify Your GPU Setup

```python
import torch

print(f"🔥 PyTorch version: {torch.__version__}")
print(f"⚡ CUDA available: {torch.cuda.is_available()}")
print(f"🎮 CUDA version: {torch.version.cuda}")
print(f"💻 GPU count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"🚀 Current GPU: {torch.cuda.get_device_name()}")
    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### 2. 🧠 Create Your First Model

```python
import torch
import torch.nn as nn

# 🎯 Automatically detect best device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# 🧠 Build a neural network
class SimpleNet(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)

# 🚀 Instantiate and move to GPU
model = SimpleNet().to(device)

# 📊 Model info
total_params = sum(p.numel() for p in model.parameters())
print(f"🧠 Model parameters: {total_params:,}")

# 🎯 Test forward pass
sample_input = torch.randn(32, 784).to(device)
output = model(sample_input)
print(f"📊 Input shape: {sample_input.shape}")
print(f"📈 Output shape: {output.shape}")
```

### 3. 🏋️ Train with MLflow Tracking

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import mlflow
import mlflow.pytorch

# 📊 Initialize MLflow experiment
mlflow.set_experiment("pytorch-cuda-training")
mlflow.start_run()

# 🎯 Setup training environment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleNet().to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()

# 📈 Log hyperparameters
mlflow.log_params({
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 10,
    "device": str(device),
    "model_params": sum(p.numel() for p in model.parameters())
})

# 📊 Create sample dataset
X = torch.randn(1000, 784)
y = torch.randint(0, 10, (1000,))
dataset = TensorDataset(X, y)
dataloader = DataLoader(
    dataset, 
    batch_size=32, 
    shuffle=True,
    num_workers=4,  # 🚀 Parallel data loading
    pin_memory=True  # ⚡ Faster GPU transfer
)

# 🏋️ Training loop with MLflow logging
model.train()
for epoch in range(10):
    epoch_loss = 0
    correct_predictions = 0
    
    pbar = tqdm(dataloader, desc=f"🏋️ Epoch {epoch+1}/10")
    
    for batch_x, batch_y in pbar:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        pred = outputs.argmax(dim=1)
        correct_predictions += (pred == batch_y).sum().item()
        
        pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    # 📊 Log metrics to MLflow
    avg_loss = epoch_loss / len(dataloader)
    accuracy = correct_predictions / len(dataset)
    
    mlflow.log_metrics({
        "loss": avg_loss,
        "accuracy": accuracy,
        "epoch": epoch + 1
    })
    
    print(f"🎯 Epoch {epoch+1}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.3f}")

# 💾 Save model
mlflow.pytorch.log_model(model, "model")
mlflow.end_run()
```

### 4. 📊 High-Performance Data Processing with Polars

```python
import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader

# 📊 Create and process data with Polars (much faster than pandas)
def create_sample_dataset():
    """Create a sample dataset using Polars for high-performance processing"""
    
    # 🚀 Generate sample data with Polars
    df = pl.DataFrame({
        "feature_1": pl.Series([i * 0.1 for i in range(10000)]),
        "feature_2": pl.Series([i * 0.2 + 1 for i in range(10000)]),
        "feature_3": pl.Series([i * 0.05 - 0.5 for i in range(10000)]),
        "target": pl.Series([i % 3 for i in range(10000)])
    })
    
    # 📈 High-performance data transformations
    processed_df = (
        df
        .with_columns([
            # 🔄 Feature engineering
            ((pl.col("feature_1") * pl.col("feature_2")).alias("interaction_1")),
            (pl.col("feature_3").pow(2).alias("feature_3_squared")),
            # 📊 Normalization
            ((pl.col("feature_1") - pl.col("feature_1").mean()) / pl.col("feature_1").std()).alias("feature_1_norm"),
            ((pl.col("feature_2") - pl.col("feature_2").mean()) / pl.col("feature_2").std()).alias("feature_2_norm")
        ])
        .filter(pl.col("feature_1") > 0.5)  # 🎯 Fast filtering
    )
    
    print(f"📊 Processed {len(processed_df)} samples")
    return processed_df

# 🎯 Custom Dataset class for Polars integration
class PolarsDataset(Dataset):
    def __init__(self, df: pl.DataFrame, feature_cols: list, target_col: str):
        self.features = torch.tensor(df.select(feature_cols).to_numpy(), dtype=torch.float32)
        self.targets = torch.tensor(df.select(target_col).to_numpy().flatten(), dtype=torch.long)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# 🚀 Use the high-performance dataset
df = create_sample_dataset()
feature_cols = ["feature_1_norm", "feature_2_norm", "feature_3_squared", "interaction_1"]

dataset = PolarsDataset(df, feature_cols, "target")
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, pin_memory=True)

print(f"✅ Created dataset with {len(dataset)} samples and {len(feature_cols)} features")
```

### 5. 📈 Interactive Visualization with Plotly

```python
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import numpy as np

def visualize_training_metrics(losses, accuracies, gpu_utilization=None):
    """Create interactive training visualizations"""
    
    # 📊 Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('🏋️ Training Loss', '🎯 Accuracy', '⚡ GPU Utilization', '📈 Learning Curve'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    epochs = list(range(1, len(losses) + 1))
    
    # 📉 Loss curve
    fig.add_trace(
        go.Scatter(x=epochs, y=losses, mode='lines+markers', name='Loss', line=dict(color='red')),
        row=1, col=1
    )
    
    # 🎯 Accuracy curve
    fig.add_trace(
        go.Scatter(x=epochs, y=accuracies, mode='lines+markers', name='Accuracy', line=dict(color='green')),
        row=1, col=2
    )
    
    # ⚡ GPU utilization (if available)
    if gpu_utilization:
        fig.add_trace(
            go.Scatter(x=epochs, y=gpu_utilization, mode='lines+markers', name='GPU %', line=dict(color='blue')),
            row=2, col=1
        )
    
    # 📈 Combined learning curve
    fig.add_trace(
        go.Scatter(x=epochs, y=losses, mode='lines', name='Loss (normalized)', line=dict(color='red', dash='dot')),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=accuracies, mode='lines', name='Accuracy', line=dict(color='green')),
        row=2, col=2
    )
    
    # 🎨 Update layout
    fig.update_layout(
        title="🚀 PyTorch CUDA Training Dashboard",
        showlegend=True,
        height=600
    )
    
    return fig

# 📊 Example usage
sample_losses = [2.3, 1.8, 1.4, 1.1, 0.9, 0.7, 0.6, 0.5, 0.4, 0.35]
sample_accuracies = [0.1, 0.3, 0.5, 0.65, 0.75, 0.82, 0.87, 0.91, 0.94, 0.96]
sample_gpu_util = [85, 87, 90, 88, 92, 89, 91, 88, 90, 87]

fig = visualize_training_metrics(sample_losses, sample_accuracies, sample_gpu_util)
fig.show()  # 🎯 Interactive visualization in browser
```

### 6. ⚡ Performance Monitoring with GPU Profiling

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import psutil
import time

def profile_training_step(model, data_loader, device):
    """Profile training performance with detailed GPU metrics"""
    
    # 🔍 Start profiling
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        
        model.train()
        for i, (batch_x, batch_y) in enumerate(data_loader):
            if i >= 5:  # Profile first 5 batches
                break
                
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            with record_function("forward_pass"):
                outputs = model(batch_x)
                loss = torch.nn.functional.cross_entropy(outputs, batch_y)
            
            with record_function("backward_pass"):
                loss.backward()
            
            with record_function("optimizer_step"):
                torch.optim.Adam(model.parameters()).step()
    
    # 📊 Print profiling results
    print("🔥 GPU Profiling Results:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    # 💾 Export for visualization
    prof.export_chrome_trace("trace.json")
    print("📈 Trace exported to trace.json - open in chrome://tracing")

def monitor_system_resources():
    """Monitor CPU, memory, and GPU usage"""
    
    # 💻 System resources
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    
    print(f"💻 CPU Usage: {cpu_percent}%")
    print(f"💾 RAM Usage: {memory.percent}% ({memory.used / 1e9:.1f}GB / {memory.total / 1e9:.1f}GB)")
    
    # 🎮 GPU resources
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1e9
        gpu_cached = torch.cuda.memory_reserved() / 1e9
        gpu_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print(f"🎮 GPU Memory: {gpu_memory:.1f}GB allocated, {gpu_cached:.1f}GB cached, {gpu_total:.1f}GB total")
        print(f"📊 GPU Utilization: {(gpu_memory/gpu_total)*100:.1f}%")

# 🚀 Example usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleNet().to(device)

# Monitor during training
monitor_system_resources()
```

---

## 🧪 Development Workflow

### 🛠️ Setup Development Environment

```bash
# 📦 Install all development tools
uv sync --extra dev

# 🪝 Setup pre-commit hooks for code quality
pre-commit install

# 🧪 Verify everything works
pytest --version && black --version && mypy --version
```

### ✨ Code Quality Arsenal

```bash
# 🎨 Format your code beautifully
black src/ tests/
isort src/ tests/

# 🔍 Lint and catch issues
ruff check src/ tests/

# 🎯 Type checking for better code
mypy src/

# 🧪 Run comprehensive tests
pytest

# 📊 Test coverage analysis
pytest --cov=src --cov-report=html
```

---

## 🚀 Performance Optimization Guide

### ⚡ GPU Memory Optimization

```python
# 💾 Monitor GPU memory usage
def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        cached = torch.cuda.memory_reserved() / 1e9
        print(f"💾 GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")

# 🧹 Memory cleanup strategies
def cleanup_gpu_memory():
    """Clean up GPU memory periodically"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# 📊 Gradient accumulation for large effective batch sizes
accumulation_steps = 4
for i, (batch_x, batch_y) in enumerate(dataloader):
    outputs = model(batch_x)
    loss = criterion(outputs, batch_y) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 🔥 Training Acceleration

```python
# ⚡ DataLoader optimization
dataloader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=min(8, os.cpu_count()),  # Optimal worker count
    pin_memory=True,  # Faster GPU transfer
    persistent_workers=True,  # Keep workers alive
    prefetch_factor=2  # Prefetch batches
)

# 🚀 Model compilation (PyTorch 2.0+)
model = torch.compile(
    model, 
    mode="max-autotune",  # Maximum optimization
    dynamic=False  # Static shapes for better optimization
)

# 💡 Learning rate scheduling
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.01,
    steps_per_epoch=len(dataloader),
    epochs=num_epochs,
    pct_start=0.3,  # 30% warmup
    anneal_strategy='cos'
)
```

---

## 🤝 Contributing

We welcome contributions from the community! Here's how to get involved:

### 🛠️ Development Setup

1. **🍴 Fork** the repository on GitHub
2. **📥 Clone** your fork: `git clone https://github.com/yourusername/torch-cuda.git`
3. **📦 Install** in development mode: `uv sync --extra dev`
4. **🌿 Create** a feature branch: `git checkout -b feature/amazing-feature`
5. **✨ Make** your changes and add comprehensive tests
6. **🧪 Run** the test suite: `pytest`
7. **🎨 Format** your code: `black . && isort .`
8. **📝 Commit** your changes: `git commit -m 'Add amazing feature'`
9. **🚀 Push** to your branch: `git push origin feature/amazing-feature`
10. **🔄 Submit** a Pull Request

---

## 🆘 Troubleshooting

### 🔥 Common CUDA Issues

**❌ CUDA Out of Memory**
```python
# 💡 Solutions:
# 1. Reduce batch size
batch_size = 16  # Instead of 64

# 2. Use gradient accumulation
accumulation_steps = 4

# 3. Enable mixed precision
from torch.cuda.amp import autocast
with autocast():
    outputs = model(inputs)

# 4. Clear cache periodically
torch.cuda.empty_cache()
```

**🐌 Slow Training Performance**
```python
# 💡 Performance boosters:
# 1. Optimize DataLoader
dataloader = DataLoader(
    dataset,
    num_workers=4,      # Parallel loading
    pin_memory=True,    # Faster GPU transfer
    persistent_workers=True  # Keep workers alive
)

# 2. Enable optimizations
torch.backends.cudnn.benchmark = True
model = torch.compile(model)

# 3. Use appropriate batch sizes
# Sweet spot is usually 32-128 depending on model size
```

**🚫 Installation Issues**
```bash
# 🔄 Refresh installation
uv sync --extra all

# 🧹 Clean cache and reinstall
uv cache clean && uv sync

# 🎯 Verify uv configuration
uv tree
```

### 🆘 Getting Help

- 🐛 **Issues**: Check our [GitHub Issues](https://github.com/bjoernbethge/torch-cuda/issues)
- 📚 **Documentation**: [PyTorch Official Docs](https://pytorch.org/docs/)
- 💬 **Community**: [PyTorch Forums](https://discuss.pytorch.org/)
- 📧 **Contact**: [bjoern.bethge@gmail.com](mailto:bjoern.bethge@gmail.com)

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- 🔥 **PyTorch Team** - For creating the most amazing deep learning framework
- 🎮 **NVIDIA** - For CUDA toolkit and GPU computing revolution  
- ⚡ **Astral Team** - For the blazing-fast `uv` package manager
- 📊 **Polars Team** - For lightning-fast data processing
- 🌟 **Open Source Community** - For continuous inspiration and collaboration

---

## 📞 Connect & Links

<div align="center">

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/bjoernbethge/torch-cuda)
[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:bjoern.bethge@gmail.com)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/bjoernbethge)

**Made with ❤️ and ⚡ GPU acceleration**

</div>

---

<div align="center">
<sub>Built with 🔥 PyTorch • Accelerated by ⚡ CUDA • Powered by 🚀 uv & Modern Python</sub>
</div>