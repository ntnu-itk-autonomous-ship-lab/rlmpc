//! # RRT*
//! Contains main RRT* functionality
//!
use crate::enc_hazards::ENCHazards;
use crate::steering::{SimpleSteering, Steering};
use crate::utils;
use config::Config;
use nalgebra::{Vector3, Vector6};
use pyo3::prelude::*;
use pyo3::types::{PyList, PySlice};
use pyo3::FromPyObject;
use rand::SeedableRng;
use rand_chacha::ChaChaRng;
use rstar::{PointDistance, RTree, RTreeObject, AABB};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct RRTNode {
    pub cost: f32,
    pub d2land: f32,
    pub state: Vector6<f64>,
}

impl RRTNode {
    pub fn new(state: Vector6<f64>, cost: f32, d2land: f32) -> Self {
        Self {
            state,
            cost,
            d2land,
        }
    }

    pub fn point(&self) -> [f64; 2] {
        [self.state[0], self.state[1]]
    }
}

impl RTreeObject for RRTNode {
    type Envelope = AABB<[f64; 2]>;

    fn envelope(&self) -> Self::Envelope {
        AABB::from_point([self.state[0], self.state[1]])
    }
}

impl PointDistance for RRTNode {
    fn distance_2(&self, point: &[f64; 2]) -> f64 {
        let x = self.state[0] - point[0];
        let y = self.state[1] - point[1];
        x * x + y * y
    }
}

#[derive(FromPyObject, Serialize, Deserialize, Debug, Clone, Copy)]
pub struct RRTParams {
    pub max_iter: u64,
    pub max_nodes: u64,
    pub max_time: f64,
    pub step_size: f64,
    pub max_steering_time: f64,
    pub alpha: f64,
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
pub struct RRTStar {
    pub params: RRTParams,
    pub steering: SimpleSteering,
    pub x_init: Vector6<f64>,
    pub x_goal: Vector6<f64>,
    pub tree: RTree<RRTNode>,
    rng: ChaChaRng,
    pub enc: ENCHazards,
}

#[pymethods]
impl RRTStar {
    #[new]
    pub fn py_new(params: RRTParams) -> Self {
        println!("RRTStar initialized with params: {:?}", params);
        Self {
            params: params.clone(),
            steering: SimpleSteering::new(),
            x_init: Vector6::zeros(),
            x_goal: Vector6::zeros(),
            tree: RTree::new(),
            rng: ChaChaRng::from_entropy(),
            enc: ENCHazards::py_new(),
        }
    }

    pub fn set_init_state(&mut self, x_init: &PySlice) -> PyResult<()> {
        let x_init_slice = x_init.extract::<[f64; 6]>()?;
        self.x_init = Vector6::from(x_init_slice);
        self.tree.insert(RRTNode {
            cost: 0.0,
            d2land: 0.0,
            state: Vector6::from(x_init_slice),
        });
        Ok(())
    }

    pub fn set_goal_state(&mut self, x_goal: &PySlice) -> PyResult<()> {
        let x_goal_slice = x_goal.extract::<[f64; 6]>()?;
        self.x_goal = Vector6::from(x_goal_slice);
        Ok(())
    }

    pub fn update_parameters(&mut self, params: RRTParams) -> PyResult<()> {
        self.params = params.clone();
        Ok(())
    }

    pub fn transfer_enc_data(&mut self, enc_data: Vec<&PyAny>) -> PyResult<()> {
        self.enc.transfer_enc_data(enc_data)
    }

    #[allow(non_snake_case)]
    pub fn grow_towards_goal(
        &mut self,
        ownship_state: &PySlice,
        do_list: &PyList,
        flag: bool,
    ) -> PyResult<Vec<&PyAny>> {
        for i in 0..self.params.max_iter {
            let z_rand = self.sample()?;
            let z_nearest = self.nearest(&z_rand)?;
            let (x_new, u_new, t_new) = self.steer(&z_nearest, &z_rand)?;
        }
        Ok(vec![])
    }
}

impl RRTStar {
    pub fn nearest(&self, z_rand: &RRTNode) -> PyResult<RRTNode> {
        let nearest = self.tree.nearest_neighbor(&z_rand.point()).unwrap().clone();
        Ok(nearest)
    }

    pub fn steer(
        &self,
        z_nearest: &RRTNode,
        z_rand: &RRTNode,
    ) -> PyResult<(Vector6<f64>, Vec<Vector3<f64>>, f64)> {
        let (xs_array, u_array, t_new) = self.steering.steer(
            &z_nearest.state,
            &z_rand.state,
            self.params.step_size,
            self.params.max_steering_time,
        );
        Ok((xs_array.last().copied().unwrap(), u_array, t_new))
    }

    pub fn sample(&mut self) -> PyResult<RRTNode> {
        let p_rand = utils::sample_free_position(&self.enc, &mut self.rng);
        Ok(RRTNode {
            state: Vector6::new(p_rand[0], p_rand[1], 0.0, 0.0, 0.0, 0.0),
            cost: 0.0,
            d2land: 0.0,
        })
    }
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn rrt_star_rust(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<RRTStar>()?;
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

    #[test]
    fn test_sample() {
        let mut rrt = RRTStar::py_new(RRTParams {
            max_iter: 1000,
            max_nodes: 1000,
            max_time: 100.0,
            step_size: 1.0,
            alpha: 1.0,
        });
        let z_rand = rrt.sample().unwrap();
        assert_eq!(z_rand.state, Vector6::zeros());
    }
}
