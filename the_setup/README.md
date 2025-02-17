# The Setup: Your First Step into AI 🚀

Welcome to The Neural Gear! This is Lesson 1, where we set up our AI development environment.

## Why Rust + PyTorch? 🦀

- **Speed**: Rust's zero-cost abstractions
- **Safety**: Memory safety without garbage collection
- **Performance**: Direct access to PyTorch's C++ backend via libtorch
- **Reliability**: Strong type system and compile-time checks

## Quick Setup Guide 🛠️

### 1. Install Rust
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustc --version  # Verify installation
```

### 2. Install libtorch
1. Download from [pytorch.org](https://pytorch.org/) (Select: C++/LibTorch, Your OS, CPU/GPU)
2. Set environment variables:
```bash
# Add to .bashrc or .zshrc
export LIBTORCH=/path/to/libtorch
export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH
```

### 3. Verify Setup
```rust
use tch::Tensor;

fn main() {
    let tensor = Tensor::f_from_slice(&[1.0f32, 2.0, 3.0]).unwrap();
    println!("Success! Tensor: {:?}", tensor);
}
```

## Next Steps 🎯
In Lesson 2, we'll dive into tensors - the building blocks of neural networks.

## Need Help? 🤝
- Check environment variables
- Ensure libtorch matches your OS
- Verify Rust toolchain is up to date
