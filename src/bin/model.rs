extern crate dbscan;
use dbscan::Model;

/// Transpose a matrix of An,m into a matrix of Am,n
pub fn transpose<T, I, V>(v: V) -> Vec<Vec<T>>
where
    T: Copy,
    I: AsRef<[T]>,
    V: AsRef<[I]>,
{
    let len = v.as_ref()[0].as_ref().len();
    let mut outer = Vec::new();
    for j in 0..len {
        outer.push(v.as_ref().iter().map(|x| x.as_ref()[j]).collect());
    }
    outer
}

fn main() {
    let v = transpose(vec![
        vec![1.0f64, 1.2, 0.8, 3.7, 3.9, 3.6, 10.],
        vec![1.1, 0.8, 1.0, 4.0, 3.9, 4.1, 10.],
    ]);

    println!("{:?}", &v);
    let c = Model::new(0.5, 2).run(&v);
    println!("{:?}", c);
}
