use anyhow::{Error, Result};
use config::Config;
use nalgebra::Vector6;
use pyo3::prelude::*;
use rstar::{PointDistance, RTree, RTreeObject, AABB};
use serde::{Deserialize, Serialize};
use rand::Rng;

#[derive(Debug, Clone)]
pub struct RRTNode {
    pub cost: f32,
    pub d2land: f32,
    pub state: Vector6<f32>,
}

impl RTreeObject for RRTNode {
    type Envelope = AABB<[f32; 2]>;

    fn envelope(&self) -> Self::Envelope {
        AABB::from_point([self.state[0], self.state[1]])
    }
}

impl PointDistance for RRTNode {
    fn distance_2(&self, point: &[f32; 2]) -> f32 {
        let x = self.state[0] - point[0];
        let y = self.state[1] - point[1];
        x * x + y * y
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RRTParams {
    pub max_iter: usize,
    pub max_time: f32,
    pub max_nodes: usize,
    pub step_size: f32,
    pub alpha: f32,
}

impl RRTParams {
    pub fn from_json_value(json: serde_json::Value) -> Self {
        let cfg = serde_json::from_value(json).unwrap();
        cfg
    }

    pub fn from_file(filename: &str) -> Self {
        let cfg = Config::builder()
            .add_source(config::File::with_name(filename))
            .build()
            .unwrap()
            .try_deserialize::<RRTParams>()
            .unwrap();
        cfg
    }

    pub fn to_file(&self, filename: &str) {
        let serialized_cfg = serde_json::to_string(&self).unwrap();
        println!("{}", serialized_cfg);
        serde_json::to_writer_pretty(std::fs::File::create(filename).unwrap(), &self).unwrap();
    }
}

#[pyclass]
pub struct RRT {
    pub params: RRTParams,
    pub x_init: Vector6<f32>,
    pub x_goal: Vector6<f32>,
    pub tree: RTree<RRTNode>,
}

// impl IntoPy<PyObject> for RRT {
//     fn into_py(self, py: Python<'_>) -> PyObject {
//         // delegates to i32's IntoPy implementation.
//         self.into_py(py)
//     }
// }

#[pymethods]
impl RRT {
    #[new]
    pub fn new(params: RRTParams, x_init: Vector6<f32>, x_goal: Vector6<f32>) -> Self {
        let mut tree = RTree::new();
        tree.insert(RRTNode {
            cost: 0.0,
            d2land: 0.0,
            state: x_init,
        });
        Self {
            params,
            x_init,
            x_goal,
            tree,
        }
    }

    pub fn grow_towards_goal(flag: bool) -> PyResult<Vec<Vector6<f32>>> {

    }

    pub fn set_init_state(&mut self, x_init: Vector6<f32>) {
        self.x_init = x_init;
    }

    pub fn set_goal_state(&mut self, x_goal: Vector6<f32>) {
        self.x_goal = x_goal;
    }

    pub fn step(&mut self) -> Result<()> {
        let x_rand = self.sample();
        let x_near = self.nearest_neighbor(&x_rand)?;
        let x_new = self.steer(&x_near, &x_rand)?;
        self.tree.insert(x_new);
        Ok(())
    }

    pub fn sample(&self) -> Vector6<f32> {
        let mut rng = rand::thread_rng();
        let x_rand =
    }
}
/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn rrt_rust(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<RRT>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rrtnode() {
        let node = RRTNode {
            cost: 0.0,
            d2land: 0.0,
            state: Vector6::zeros(),
        };
        assert_eq!(node.state, Vector6::zeros());
        assert_eq!(node.distance_2(&[1.0, 0.0]), 1.0);
    }

    #[test]
    fn test_tree() {
        let mut tree = RTree::new();
        tree.insert(RRTNode {
            cost: 0.0,
            d2land: 0.0,
            state: Vector6::zeros(),
        });

        tree.insert(RRTNode {
            cost: 2.0,
            d2land: 40.0,
            state: Vector6::new(50.0, 50.0, 0.0, 0.0, 0.0, 0.0),
        });

        let nearest = tree.nearest_neighbor(&[1.0, 0.0]).unwrap();
        assert_eq!(nearest.state, Vector6::zeros());
    }
}
