//! # RRT*
//! Contains main RRT* functionality
//!
use crate::enc_hazards::ENCHazards;
use crate::steering::{SimpleSteering, Steering};
use crate::utils;
use config::Config;
use nalgebra::{Vector2, Vector3, Vector6};
use pyo3::prelude::*;
use pyo3::types::PySlice;
use pyo3::FromPyObject;
use rand_chacha::ChaChaRng;
use kd_tree::{KdTree, KdPoint};
use serde::{Deserialize, Serialize};
use typenum;
use rand::SeedableRng;

#[derive(Debug, Clone, PartialEq)]
pub struct RRTNode {
    pub id: usize,
    pub parent: Option<usize>,
    pub cost: f64,
    pub d2land: f64,
    pub state: Vector6<f64>,
}

impl RRTNode {
    pub fn new(id: usize, parent: Option<usize>, state: Vector6<f64>, cost: f64, d2land: f64) -> Self {
        Self {
            id,
            parent,
            cost,
            d2land,
            state,
        }
    }

    pub fn point(&self) -> [f64; 2] {
        [self.state[0], self.state[1]]
    }

    pub fn vec2d(&self) -> Vector2<f64> {
        Vector2::new(self.state[0], self.state[1])
    }

    pub fn dist2other(&self, other: &Self) -> f64 {
        let dx = self.state[0] - other.state[0];
        let dy = self.state[1] - other.state[1];
        f64::sqrt(dx * dx + dy * dy)
    }
}

impl KdPoint for RRTNode {
    type Scalar = f64;
    type Dim = typenum::U2;

    fn at(&self, k: usize) -> f64 {
        self.state[k]
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
    pub tree: KdTree<RRTNode>,
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
            tree: KdTree::build_by_ordered_float(vec![]),
            rng: ChaChaRng::from_entropy(),
            enc: ENCHazards::py_new(),
        }
    }

    pub fn set_init_state(&mut self, x_init: &PySlice) -> PyResult<()> {
        let x_init_slice = x_init.extract::<[f64; 6]>()?;
        self.x_init = Vector6::from(x_init_slice);
        self.tree.fill(RRTNode {
            id: 0,
            parent: None,
            cost: 0.0,
            d2land: 0.0,
            state: Vector6::from(x_init_slice),
        });
        self.tree.fill_with(f64::INFINITY, 1, self.params.max_nodes as usize);
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

    pub fn grow_towards_goal(
        &mut self,
        ownship_state: &PySlice,
        do_list: &PyList,
        flag: bool,
    ) -> PyResult<Vec<&PyAny>> {
        for i in 0..self.params.max_iter {
            let z_rand = self.sample()?;
            let z_nearest = self.nearest(&z_rand)?;
            let (xs_array, _, _, _) = self.steer(&z_nearest, &z_rand)?;
            let x_new = xs_array.last().copied().unwrap();
            let is_collision_free = self
                .enc
                .intersects_with_segment(&z_nearest.vec2d(), &Vector2::new(x_new[0], x_new[1]));
            if is_collision_free {
                let z_new = RRTNode::new(x_new, z_nearest.cost, 0.0);
                let Z_near = self.nearest_neighbors(&z_new)?;
                let z_min = self.choose_parent(&z_new, &Z_near)?;
                self.rewire(&z_min, &Z_near)?;
                self.tree.insert(z_min.clone());

                if self.reached_goal(&z_min) {
                    return Ok(self.reconstruct_path(&z_min));
                }
            }
        }
        Ok(vec![])
    }
}

#[allow(non_snake_case)]
impl RRTStar {
    pub fn reconstruct_path(&self, z: &RRTNode) -> Vec<&PyAny> {
        let mut path = vec![];
        let mut z = z;
        z.
        while z.parent.is_some() {
            let z_parent = z.parent.unwrap();
            path.push(z_parent);
            z = z_parent;
        }
        path.push(z);
        path.reverse();
        path
    }

    pub fn reached_goal(&self, z: &RRTNode) -> bool {
        let x = z.state[0];
        let y = z.state[1];
        let x_goal = self.x_goal[0];
        let y_goal = self.x_goal[1];
        let dist = ((x - x_goal) * (x - x_goal) + (y - y_goal) * (y - y_goal)).sqrt();
        dist < 20.0
    }

    pub fn rewire(&mut self, z_min: &RRTNode, Z_near: &Vec<RRTNode>) -> PyResult<()> {
        for z_near in Z_near {
            let (xs_array, _, _, _) = self.steer(&z_near, &z_min)?;
            let x_new = xs_array.last().copied().unwrap();
            let is_collision_free = self
                .enc
                .intersects_with_segment(&Vector2::new(x_new[0], x_new[1]), &z_min.vec2d());
            if is_collision_free {
                let path_length = utils::compute_path_length(&xs_array);
                let z_new = RRTNode::new(x_new, z_min.cost + path_length, 0.0);
                if z_new.cost < z_near.cost {
                    self.tree.remove(z_near);
                    self.tree.insert(z_new);
                }
            }
        }
        Ok(())
    }

    pub fn nearest(&self, z_rand: &RRTNode) -> PyResult<RRTNode> {
        let nearest = self.tree.nearest_neighbor(&z_rand.point()).unwrap().clone();
        Ok(nearest)
    }

    pub fn nearest_neighbors(&self, z_new: &RRTNode) -> PyResult<Vec<RRTNode>> {
        let mut Z_near = self
            .tree
            .nearest_neighbor_iter(&z_new.point())
            .take_while(|z| z.distance_2(&z_new.point()) < self.params.alpha.powi(2))
            .map(|z| z.clone())
            .collect::<Vec<_>>();
        Z_near.sort_by(|a, b| a.cost.partial_cmp(&b.cost).unwrap());
        Ok(Z_near)
    }

    /// Select parent here, and dont consider the z_nearest node in the iteration

    pub fn choose_parent(&self, z_new: &RRTNode, Z_near: &[RRTNode]) -> PyResult<RRTNode> {
        let mut z_min = z_new.clone();
        let mut z_nearest = Z_near[0].clone();
        for z_near in Z_near {
            let (xs_array, _, _, _) = self.steer(&z_near, &z_new)?;
            let x_new = xs_array.last().copied().unwrap();
            let is_collision_free = self
                .enc
                .intersects_with_segment(&z_near.vec2d(), &Vector2::new(x_new[0], x_new[1]));
            if is_collision_free {
                let path_length = utils::compute_path_length(&xs_array);
                let cost = z_near.cost + path_length;
                if cost < z_min.cost {
                    z_min = RRTNode::new(x_new, cost, 0.0);
                    z_nearest = z_near.clone();
                }
            }
        }
        Ok(z_min)
    }

    pub fn steer(
        &mut self,
        z_nearest: &RRTNode,
        z_rand: &RRTNode,
    ) -> PyResult<(Vec<Vector6<f64>>, Vec<Vector3<f64>>, Vec<(f64, f64)>, f64)> {
        let (xs_array, u_array, refs_array, t_new) = self.steering.steer(
            &z_nearest.state,
            &z_rand.state,
            self.params.step_size,
            self.params.max_steering_time,
        );
        Ok((xs_array, u_array, refs_array, t_new))
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
    fn test_sample() -> PyResult<()> {
        let mut rrt = RRTStar::py_new(RRTParams {
            max_iter: 1000,
            max_nodes: 1000,
            max_time: 100.0,
            step_size: 1.0,
            max_steering_time: 20.0,
            alpha: 1.0,
        });
        let z_rand = rrt.sample()?;
        assert_eq!(z_rand.state, Vector6::zeros());
        Ok(())
    }
}
