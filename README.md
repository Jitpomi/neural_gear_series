# Neural Gear: Learning Journey 🧠

A simple yet powerful demonstration of neural network fundamentals using Rust and PyTorch (tch-rs).

## Learning Process Visualization

```mermaid
flowchart TD
    A[Training Data] -->|1. Feed Forward| B[Neural Network]
    B -->|2. Make Prediction| C[Predicted Output]
    D[Expected Output] -->|3. Compare| E[Calculate Loss]
    C -->|3. Compare| E
    E -->|4. Backpropagate| B
    B -->|5. Update Weights| B
    
    style A fill:#f9f,stroke:#333,stroke-width:4px
    style B fill:#bbf,stroke:#333,stroke-width:4px
    style C fill:#dfd,stroke:#333,stroke-width:4px
    style D fill:#f96,stroke:#333,stroke-width:4px
    style E fill:#ff9,stroke:#333,stroke-width:4px
```

## What's Happening?

1. **Training Data** (🎲): Numbers we want our AI to learn from
   - Input: `[1.0, 2.0, 3.0, 4.0]`
   - Each number gets stacked into a tower shape

2. **Neural Network** (🧠): A simple linear model that learns the pattern
   - Tries to learn the recipe: `output = input × 2 + 1`
   - Has weights and biases that it adjusts during training

3. **Learning Process** (📚):
   - Makes predictions
   - Compares with correct answers
   - Updates its understanding
   - Repeats 500 times to get better!

## Running the Project

```bash
cargo run
```

Watch as the AI learns! The "loss" number gets smaller as it gets better at predictions.

## Dependencies
- Rust
- tch (PyTorch for Rust)
