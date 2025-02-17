use tch::{nn, Device, Kind, Tensor};
use tch::nn::Module;

fn main() {
    println!("\n1️⃣ Tensors: The Number Wizards 🧙‍♂️");
    create_tensor();

    println!("\n2️⃣ Computational Graphs: AI's Map 🗺️");
    build_computational_graph();

    println!("\n3️⃣ Autograd: AI's Memory 🧠");
    learn_from_mistakes();

    println!("\n4️⃣ Model Serialization: Saving AI's Brain 💾");
    save_model();

    println!("\n5️⃣ Using the Trained Model: AI Remembers! 🤖");
    let prediction = predict(4.0);
    println!("When input is 4.0, AI predicts: {}", prediction);
    println!("Expected output (4.0 × 2 + 1): 9.0");
}

/// 1️⃣ Tensors: The Number Wizards 🧙‍♂️
/// Creates a simple tensor to store numbers
fn create_tensor() {
    let tensor = Tensor::from_slice(&[2.0, 3.0]);  // 1D Tensor
    println!("Tensor: {:?}", tensor);
}

/// 2️⃣ Computational Graphs: AI's Map 🗺️
/// Builds a graph that multiplies by 2 and adds 1
fn build_computational_graph() {
    let x = Tensor::from_slice(&[2.0, 3.0]).requires_grad_(true);
    let y = &x * 2.0 + 1.0;  // AI follows this path
    println!("Computational Graph Output: {:?}", y);
}

/// 3️⃣ Autograd: AI's Memory 🧠
/// Demonstrates how AI learns from mistakes using autograd
fn learn_from_mistakes() {
    let x = Tensor::from_slice(&[2.0, 3.0]).requires_grad_(true);
    let y = &x * 3.0 + 2.0;  // Multiply by 3, then add 2
    let loss = y.sum(Kind::Float); // AI calculates how wrong it was
    loss.backward();  // AI learns from the mistake
    println!("Gradient: {:?}", x.grad());
}

/// 4️⃣ Model Serialization: Saving AI's Brain 💾
/// Creates and saves a model that multiplies by 2 and adds 1
fn save_model() {
    let vs = nn::VarStore::new(Device::Cpu);
    let linear = nn::linear(vs.root(), 1, 1, Default::default());
    
    // Initialize the weights to multiply by 2 and add 1
    tch::no_grad(|| {
        let params = vs.trainable_variables();
        for mut param in params {
            let size = param.size();
            if size.len() == 2 {  // Weight matrix
                param.copy_(&Tensor::from_slice(&[2.0]).reshape(&[1, 1]));
            } else {  // Bias vector
                param.copy_(&Tensor::from_slice(&[1.0]));
            }
        }
    });

    vs.save("model.ot").unwrap();
    println!("Model saved with weights that multiply by 2 and add 1!");
}

/// 5️⃣ Using the Trained Model: AI Remembers!
/// Loads the saved model and uses it to make predictions
fn predict(val: f32) -> f64 {
    let mut vs = nn::VarStore::new(Device::Cpu);
    let linear = nn::linear(vs.root(), 1, 1, Default::default());
    vs.load("model.ot").unwrap();  // Load the saved AI brain

    // Create input tensor with value 4.0
    let input = Tensor::from_slice(&[val]).reshape(&[-1, 1]);
    let output = linear.forward(&input);

    // Get the actual value from the tensor
    output.double_value(&[0, 0])
}
