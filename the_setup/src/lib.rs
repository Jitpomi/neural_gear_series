//! # The Setup
//! Welcome to Lesson 1 of The Neural Gear series!
//! This module provides a simple verification of your AI development environment setup.

use tch::Tensor;

/// Verifies that the tch-rs setup is working correctly by creating a simple tensor
pub fn verify_setup() -> Result<Tensor, Box<dyn std::error::Error>> {
    // Create a simple tensor
    let tensor = Tensor::f_from_slice(&[1.0f32, 2.0, 3.0])?;
    println!("✅ Tensor created successfully!");
    println!("🔍 Tensor details: {:?}", tensor);
    
    Ok(tensor)
}

pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }

    #[test]
    fn test_verify_setup() {
        let result = verify_setup();
        assert!(result.is_ok(), "Setup verification failed!");
        
        let tensor = result.unwrap();
        assert_eq!(tensor.size(), &[3], "Tensor should have size [3]");
    }
}
