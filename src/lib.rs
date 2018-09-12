extern crate myriad;
use myriad::mpmc::{self, Receiver, Sender};

/// Calculate euclidean distance between two vectors
#[inline]
pub fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .fold(0f64, |acc, (x, y)| acc + (x - y).powi(2))
        .sqrt()
}

pub struct Model {
    eps: f64,
    mpt: usize,
    c: Vec<Option<usize>>,
    v: Vec<bool>,
}

type Matrix = Vec<Vec<f64>>;

impl Model {
    pub fn new(eps: f64, min_points: usize) -> Model {
        Model {
            eps,
            mpt: min_points,
            c: Vec::new(),
            v: Vec::new(),
        }
    }

    fn expand_cluster(
        &mut self,
        population: &Matrix,
        index: usize,
        neighbors: &[usize],
        cluster: usize,
    ) {
        self.c[index] = Some(cluster);
        for &n_idx in neighbors {
            // Have we previously visited this point?
            let v = self.v[n_idx];
            if !v {
                self.v[n_idx] = true;

                // What about neighbors of this neighbor? Are they close enough to add into
                // the current cluster? If so, recurse and add them.
                let nn = self.range_query(&population[n_idx], population, euclidean_distance);
                if nn.len() >= self.mpt {
                    self.expand_cluster(population, n_idx, &nn, cluster);
                }
            }
        }
    }

    fn range_query<F>(&self, sample: &[f64], population: &Matrix, distance: F) -> Vec<usize>
    where
        F: Fn(&[f64], &[f64]) -> f64,
    {
        let mut neighbors = Vec::new();
        for (i, p) in population.iter().enumerate() {
            if distance(sample, p) < self.eps {
                neighbors.push(i);
            }
        }
        neighbors
    }

    pub fn run(mut self, population: &Matrix) -> Vec<Option<usize>> {
        self.c = (0..population.len()).map(|_| None).collect();
        self.v = (0..population.len()).map(|_| false).collect();

        let mut cluster = 0;
        for (idx, sample) in population.iter().enumerate() {
            let v = self.v[idx];
            if !v {
                self.v[idx] = true;
                let n = self.range_query(sample, population, euclidean_distance);
                if n.len() >= self.mpt {
                    self.expand_cluster(population, idx, &n, cluster);
                    cluster += 1;
                }
            }
        }
        self.c
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cluster() {
        let model = Model::new(1.0, 3);
        let inputs = vec![
            vec![1.0, 1.1],
            vec![1.2, 0.8],
            vec![0.8, 1.0],
            vec![3.7, 4.0],
            vec![3.9, 3.9],
            vec![3.6, 4.1],
            vec![10.0, 10.0],
        ];
        let output = model.run(&inputs);
        assert_eq!(
            output,
            vec![Some(0), Some(0), Some(0), Some(1), Some(1), Some(1), None]
        );
    }

    #[test]
    fn range_query() {
        let model = Model::new(1.0, 3);
        let inputs = vec![vec![1.0, 1.0], vec![1.1, 1.9], vec![3.0, 3.0]];
        let neighbours = model.range_query(&[1.0, 1.0], &inputs, euclidean_distance);

        assert!(neighbours.len() == 2);
    }

    #[test]
    fn range_query_small_eps() {
        let model = Model::new(0.01, 3);
        let inputs = vec![vec![1.0, 1.0], vec![1.1, 1.9], vec![3.0, 3.0]];
        let neighbours = model.range_query(&[1.0, 1.0], &inputs, euclidean_distance);

        assert!(neighbours.len() == 1);
    }
}
