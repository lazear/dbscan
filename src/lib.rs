/// Calculate euclidean distance between two vectors
#[inline]
fn euclidean_distance<T>(a: &[T], b: &[T]) -> f64
where
    f64: From<T>,
    T: Copy,
{
    a.iter()
        .zip(b.iter())
        .fold(0f64, |acc, (&x, &y)| {
            acc + (f64::from(x) - f64::from(y)).powi(2)
        }).sqrt()
}

/// Cluster datapoints using the DBSCAN algorithm
///
/// # Arguments
/// * `eps` - maximum distance between datapoints within a cluster
/// * `min_points` - minimum number of datapoints to make a cluster
/// * `input` - a Vec<Vec<f64>> of datapoints, organized by row
pub fn cluster<T>(eps: f64, min_points: usize, input: &Vec<Vec<T>>) -> Vec<Option<usize>>
where
    T: Copy,
    f64: From<T>,
{
    Model::new(eps, min_points).run(input)
}

/// DBSCAN parameters
pub struct Model<T> 
where
    T: Copy,
    f64: From<T>,
    {
    /// Epsilon value - maximum distance between points in a cluster
    pub eps: f64,
    /// Minimum number of points in a cluster
    pub mpt: usize,

    distance: fn(&[T], &[T]) -> f64,
    c: Vec<Option<usize>>,
    v: Vec<bool>,
}

impl<T> Model<T> 
where
    T: Copy,
    f64: From<T>,
{
    /// Create a new `Model` with a set of parameters
    ///
    /// # Arguments
    /// * `eps` - maximum distance between datapoints within a cluster
    /// * `min_points` - minimum number of datapoints to make a cluster
    pub fn new(eps: f64, min_points: usize) -> Model<T> {
        Model {
            eps,
            mpt: min_points,
            c: Vec::new(),
            v: Vec::new(),
            distance: euclidean_distance,
        }
    }

    /// Change the function used to calculate distance between points
    pub fn set_distance_fn<F>(mut self, func:  fn(&[T], &[T]) -> f64) -> Model<T> 
    {
        self.distance = func;
        self
    }

    fn expand_cluster(
        &mut self,
        population: &Vec<Vec<T>>,
        index: usize,
        neighbors: &[usize],
        cluster: usize,
    )
    {
        self.c[index] = Some(cluster);
        for &n_idx in neighbors {
            // Have we previously visited this point?
            let v = self.v[n_idx];
            if !v {
                self.v[n_idx] = true;

                // What about neighbors of this neighbor? Are they close enough to add into
                // the current cluster? If so, recurse and add them.
                let nn = self.range_query(&population[n_idx], population);
                if nn.len() >= self.mpt {
                    self.expand_cluster(population, n_idx, &nn, cluster);
                }
            }
        }
    }

    #[inline]
    fn range_query(&self, sample: &[T], population: &Vec<Vec<T>>) -> Vec<usize>
    {
        let mut neighbors = Vec::new();
        for (i, p) in population.iter().enumerate() {
            if (self.distance)(sample, p) < self.eps {
                neighbors.push(i);
            }
        }
        neighbors
    }

    /// Run the DBSCAN algorithm on a given population of datapoints.
    /// 
    /// A vector of `Option<usize>` is returned, where each element
    /// corresponds to a row in the input matrix. `Some(usize)` represents
    /// cluster membership, and `None` represents noise/outliers
    ///
    /// # Arguments
    /// * `population` - a matrix of datapoints, organized by rows
    /// 
    /// # Example
    /// 
    /// ```rust
    /// use dbscan::Model;
    /// 
    /// let model = Model::new(1.0, 3);
    /// let inputs = vec![
    ///     vec![1.0, 1.1],
    ///     vec![1.2, 0.8],
    ///     vec![0.8, 1.0],
    ///     vec![3.7, 4.0],
    ///     vec![3.9, 3.9],
    ///     vec![3.6, 4.1],
    ///     vec![10.0, 10.0],
    /// ];
    /// let output = model.run(&inputs);
    /// assert_eq!(
    ///     output,
    ///     vec![Some(0), Some(0), Some(0), Some(1), Some(1), Some(1), None]
    /// );
    /// ```
    pub fn run(mut self, population: &Vec<Vec<T>>) -> Vec<Option<usize>>
    {
        self.c = (0..population.len()).map(|_| None).collect();
        self.v = (0..population.len()).map(|_| false).collect();

        let mut cluster = 0;
        for (idx, sample) in population.iter().enumerate() {
            let v = self.v[idx];
            if !v {
                self.v[idx] = true;
                let n = self.range_query(sample, population);
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
        let neighbours = model.range_query(&[1.0, 1.0], &inputs);

        assert!(neighbours.len() == 2);
    }

    #[test]
    fn range_query_small_eps() {
        let model = Model::new(0.01, 3);
        let inputs = vec![vec![1.0, 1.0], vec![1.1, 1.9], vec![3.0, 3.0]];
        let neighbours = model.range_query(&[1.0, 1.0], &inputs);

        assert!(neighbours.len() == 1);
    }
}
