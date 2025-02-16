use tch::Tensor;

fn main() {
    let t = Tensor::from_slice(&[1, 2, 3]);

    println!("{:?}",t);
}