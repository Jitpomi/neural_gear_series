use tch::{nn, Device,  Tensor, Reduction};
use tch::nn::{Module, OptimizerConfig};

fn main() {
    println!("\n1️⃣ Tensors: The Number Wizards 🧙‍♂️");
    let x = create_training_data();
    println!("Training data (x): {:?}", x);
    let y = create_target_data(&x);
    println!("Target data (y): {:?}", y);

    println!("\n2️⃣ Computational Graphs: AI's Map 🗺️");
    let mut model = build_model();
    let initial_pred = model.forward(&x);
    println!("Initial prediction: {:?}", initial_pred);

    println!("\n3️⃣ Autograd: AI's Memory 🧠");
    train_model(&mut model, &x, &y);

    println!("\n4️⃣ Model Serialization: Saving AI's Brain 💾");
    save_model(&model);

    println!("\n5️⃣ Using the Trained Model: AI Remembers! 🤖");
    let test_input = 4.0;
    let prediction = predict(test_input);
    println!("When input is {}, AI predicts: {}", test_input, prediction);
    println!("Expected output ({} × 2 + 1): {}", test_input, test_input * 2.0 + 1.0);
}

/// Create training data tensor with gradient tracking
fn create_training_data() -> Tensor {
    Tensor::from_slice(&[1.0f32, 2.0f32, 3.0f32, 4.0f32]).reshape(&[-1, 1]).requires_grad_(true)
}

/// Create target data following y = 2x + 1
fn create_target_data(x: &Tensor) -> Tensor {
    x * 2.0 + 1.0
}

/// Build a linear model that we'll train to learn y = 2x + 1
#[derive(Debug)]
struct Model {
    vs: nn::VarStore,
    linear: nn::Linear,
}

impl Module for Model {
    fn forward(&self, xs: &Tensor) -> Tensor {
        self.linear.forward(xs)
    }
}

fn build_model() -> Model {
    let vs = nn::VarStore::new(Device::Cpu);
    let linear = nn::linear(vs.root(), 1, 1, Default::default());
    Model { vs, linear }
}

/// Train the model using gradient descent
fn train_model(model: &mut Model, x: &Tensor, y: &Tensor) {
    let learning_rate = 0.1;
    let mut opt = nn::Adam::default().build(&model.vs, learning_rate).unwrap();
    
    for epoch in 0..100 {
        let prediction = model.forward(x);
        let loss = prediction.mse_loss(y, Reduction::Mean);
        
        opt.backward_step(&loss);
        
        if epoch % 20 == 0 {
            println!("Epoch {}: Loss = {:?}", epoch, loss.double_value(&[]));
        }
    }
    
    // Show final parameters
    let mut vs_map = model.vs.variables();
    if let Some(w) = vs_map.get("linear.weight") {
        println!("Final weight: {:?}", w.copy());
    }
    if let Some(b) = vs_map.get("linear.bias") {
        println!("Final bias: {:?}", b.copy());
    }
}

/// Save the trained model
fn save_model(model: &Model) {
    model.vs.save("model.ot").unwrap();
    println!("Model saved with its trained parameters! 🧠");
}

/// Load and use the trained model
fn predict(val: f32) -> f64 {
    let mut vs = nn::VarStore::new(Device::Cpu);
    let linear = nn::linear(vs.root(), 1, 1, Default::default());
    vs.load("model.ot").unwrap();
    
    let input = Tensor::from_slice(&[val]).reshape(&[-1, 1]);
    let output = linear.forward(&input);
    output.double_value(&[0, 0])
}
