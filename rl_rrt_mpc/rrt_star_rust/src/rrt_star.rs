//! # RRT*
//! Contains main RRT* functionality
//!
use crate::enc_hazards::ENCHazards;
use crate::steering::{SimpleSteering, Steering};
use crate::utils;
use config::Config;
use id_tree::InsertBehavior::*;
use id_tree::*;
use nalgebra::{Vector2, Vector3, Vector6};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyList};
use pyo3::FromPyObject;
use rand::SeedableRng;
use rand_chacha::ChaChaRng;
use rstar::{PointDistance, RTree, RTreeObject, AABB};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq)]
pub struct RRTNode {
    pub id: Option<NodeId>,
    pub cost: f64,
    pub d2land: f64,
    pub state: Vector6<f64>,
    pub time: f64,
}

impl RRTNode {
    pub fn new(state: Vector6<f64>, cost: f64, d2land: f64, time: f64) -> Self {
        Self {
            id: None,
            state,
            cost,
            d2land,
            time,
        }
    }

    pub fn set_id(&mut self, id: NodeId) {
        self.id = Some(id);
    }

    pub fn point(&self) -> [f64; 2] {
        [self.state[0], self.state[1]]
    }

    pub fn vec2d(&self) -> Vector2<f64> {
        Vector2::new(self.state[0], self.state[1])
    }
}

impl RTreeObject for RRTNode {
    type Envelope = AABB<[f64; 2]>;

    fn envelope(&self) -> Self::Envelope {
        AABB::from_point(self.point())
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
    pub goal_radius: f64,
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
    pub c_best: f64,
    pub solutions: Vec<(Vec<Vector6<f64>>, Vec<f64>, f64)>, // (states, times, cost) for each solution
    pub params: RRTParams,
    pub steering: SimpleSteering,
    pub x_start: Vector6<f64>,
    pub x_goal: Vector6<f64>,
    pub rtree: RTree<RRTNode>,
    bookkeeping_tree: Tree<RRTNode>,
    rng: ChaChaRng,
    pub enc: ENCHazards,
}

#[pymethods]
impl RRTStar {
    #[new]
    pub fn py_new(params: RRTParams) -> Self {
        println!("RRTStar initialized with params: {:?}", params);
        Self {
            c_best: std::f64::INFINITY,
            solutions: Vec::new(),
            params: params.clone(),
            steering: SimpleSteering::new(),
            x_start: Vector6::zeros(),
            x_goal: Vector6::zeros(),
            rtree: RTree::new(),
            bookkeeping_tree: Tree::new(),
            rng: ChaChaRng::from_entropy(),
            enc: ENCHazards::py_new(),
        }
    }

    pub fn set_init_state(&mut self, x_start: &PyList) -> PyResult<()> {
        let x_start_vec = x_start.extract::<Vec<f64>>()?;
        self.x_start = Vector6::from_vec(x_start_vec);

        let root_node = Node::new(RRTNode {
            id: None,
            cost: 0.0,
            d2land: 0.0,
            state: self.x_start.clone(),
            time: 0.0,
        });
        let root_id = self.bookkeeping_tree.insert(root_node, AsRoot).unwrap();
        let root = self.bookkeeping_tree.get_mut(&root_id).unwrap();
        root.data_mut().set_id(root_id.clone());

        self.rtree.insert(RRTNode {
            id: Some(root_id),
            cost: 0.0,
            d2land: 0.0,
            state: self.x_start.clone(),
            time: 0.0,
        });
        Ok(())
    }

    pub fn set_goal_state(&mut self, x_goal: &PyList) -> PyResult<()> {
        let x_goal_vec = x_goal.extract::<Vec<f64>>()?;
        self.x_goal = Vector6::from_vec(x_goal_vec);
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
        ownship_state: &PyList,
        do_list: &PyList,
        flag: bool,
    ) -> PyResult<Vec<&PyAny>> {
        self.set_init_state(ownship_state)?;
        self.c_best = std::f64::INFINITY;
        self.solutions = Vec::new();
        for _ in 0..self.params.max_iter {
            //self.c_best = self.solutions.iter().fold(std::f64::INFINITY, |acc, x| (acc).min(x.2));
            println!("c_best: {}", self.c_best);
            self.print_tree()?;
            let z_rand = self.sample()?;
            let z_nearest = self.nearest(&z_rand)?;
            let (xs_array, _, _, t_new) = self.steer(&z_nearest, &z_rand)?;
            let x_new: Vector6<f64> = xs_array.last().copied().unwrap();
            if self.is_collision_free(&z_nearest, &x_new) {
                let path_length = utils::compute_path_length(&xs_array);
                let z_new = RRTNode::new(
                    x_new,
                    z_nearest.cost + path_length,
                    0.0,
                    z_nearest.time + t_new,
                );
                let Z_near = self.nearest_neighbors(&z_new)?;
                println!("Z_near: {:?}", Z_near);
                let (z_new, z_parent) = self.choose_parent(&z_new, &Z_near)?;
                println!("z_new: {:?} \nz_parent: {:?}", z_new, z_parent);
                self.insert(&z_new, &z_parent)?;
                self.rewire(&z_new, &Z_near)?;
                if self.reached_goal(&z_new) {
                    let soln = self.reconstruct_trajectory(&z_new);
                    self.solutions.push(soln.clone());
                    self.c_best = self.c_best.min(soln.2);
                }
            }
        }
        Ok(vec![])
    }
}

#[allow(non_snake_case)]
impl RRTStar {
    pub fn reconstruct_trajectory(&self, z: &RRTNode) -> (Vec<Vector6<f64>>, Vec<f64>, f64) {
        let mut states = vec![z.state];
        let mut times = vec![z.time];
        let mut z_current = self.bookkeeping_tree.get(&z.clone().id.unwrap()).unwrap();
        let cost = z_current.data().cost;
        while z_current.parent().is_some() {
            println!("State: {:?}", z_current.data().state);
            let parent_id = z_current.parent().unwrap();
            let z_parent = self.bookkeeping_tree.get(&parent_id).unwrap();
            states.push(z_parent.data().state);
            times.push(z_parent.data().time);
            z_current = z_parent;
        }
        states.push(z_current.data().state);
        times.push(z_current.data().time);
        states.reverse();
        times.reverse();
        (states, times, cost)
    }

    pub fn is_collision_free(&self, z_nearest: &RRTNode, x_new: &Vector6<f64>) -> bool {
        if self.enc.is_empty() {
            return true;
        }
        let is_collision_free = self
            .enc
            .intersects_with_segment(&z_nearest.vec2d(), &Vector2::new(x_new[0], x_new[1]));
        is_collision_free
    }

    pub fn reached_goal(&self, z: &RRTNode) -> bool {
        let x = z.state[0];
        let y = z.state[1];
        let x_goal = self.x_goal[0];
        let y_goal = self.x_goal[1];
        let dist_squared = (x - x_goal).powi(2) + (y - y_goal).powi(2);
        dist_squared < self.params.goal_radius.powi(2)
    }

    /// Inserts a new node into the tree, with the parent node being z_parent
    /// Since we have two trees (RTree for nearest neighbor search and Tree for keeping track of parents/children),
    /// we need to keep track of the node id in both trees. This is done by setting the id of the node in the Tree
    pub fn insert(&mut self, z: &RRTNode, z_parent: &RRTNode) -> PyResult<NodeId> {
        let z_parent_id = z_parent.clone().id.unwrap();
        let z_node = Node::new(z.clone());
        let z_id = self
            .bookkeeping_tree
            .insert(z_node, UnderNode(&z_parent_id))
            .unwrap();
        let z_node = self.bookkeeping_tree.get_mut(&z_id).unwrap();
        z_node.data_mut().set_id(z_id.clone());

        let mut z_copy = z.clone();
        z_copy.set_id(z_id.clone());
        self.rtree.insert(z_copy);

        Ok(z_id)
    }

    pub fn rewire(&mut self, z_new: &RRTNode, Z_near: &Vec<RRTNode>) -> PyResult<()> {
        let z_new_parent_id = self
            .bookkeeping_tree
            .get(&z_new.clone().id.unwrap())
            .unwrap()
            .parent()
            .unwrap();
        for z_near in Z_near.iter() {
            if z_new_parent_id.eq(&z_near.clone().id.unwrap()) {
                continue;
            }
            let (xs_array, _, _, t_new) = self.steer(&z_new, &z_near)?;
            let x_new_near: Vector6<f64> = xs_array.last().copied().unwrap();
            if self.is_collision_free(&z_new, &x_new_near) {
                let path_length = utils::compute_path_length(&xs_array);
                let z_new_near = RRTNode::new(
                    x_new_near,
                    z_new.cost + path_length,
                    0.0,
                    z_new.time + t_new,
                );
                if z_new_near.cost < z_near.cost {
                    let z_near_id = z_near.clone().id.unwrap();
                    self.transfer_node_data(&z_near_id, &z_new)?;

                    self.move_node(&z_near_id, &z_new.clone().id.unwrap())?;

                    self.rtree.remove(z_near);
                    self.rtree.insert(
                        self.bookkeeping_tree
                            .get(&z_near_id)
                            .unwrap()
                            .data()
                            .clone(),
                    );
                }
            }
        }
        Ok(())
    }

    fn transfer_node_data(&mut self, z_recipient_id: &NodeId, z_new: &RRTNode) -> PyResult<()> {
        let z_recipient = self.bookkeeping_tree.get_mut(&z_recipient_id).unwrap();
        z_recipient.data_mut().state = z_new.state;
        z_recipient.data_mut().cost = z_new.cost;
        z_recipient.data_mut().time = z_new.time;
        z_recipient.data_mut().d2land = z_new.d2land;
        Ok(())
    }

    fn move_node(&mut self, z_id: &NodeId, z_parent_id: &NodeId) -> PyResult<()> {
        self.bookkeeping_tree
            .move_node(&z_id, MoveBehavior::ToParent(&z_parent_id))
            .unwrap();
        Ok(())
    }

    pub fn nearest(&self, z_rand: &RRTNode) -> PyResult<RRTNode> {
        let nearest = self
            .rtree
            .nearest_neighbor(&z_rand.point())
            .unwrap()
            .clone();
        Ok(nearest)
    }

    pub fn nearest_neighbors(&self, z_new: &RRTNode) -> PyResult<Vec<RRTNode>> {
        let ball_radius = self.compute_ball_radius();
        if self.rtree.size() == 1 {
            let root_id = self.bookkeeping_tree.root_node_id().unwrap();
            let z = self.bookkeeping_tree.get(root_id).unwrap().data().clone();
            return Ok(vec![z]);
        }

        let mut Z_near = self
            .rtree
            .nearest_neighbor_iter(&z_new.point())
            .take_while(|z| z.distance_2(&z_new.point()) < ball_radius.powi(2))
            .map(|z| z.clone())
            .collect::<Vec<_>>();
        Z_near.sort_by(|a, b| a.cost.partial_cmp(&b.cost).unwrap());
        Ok(Z_near)
    }

    pub fn compute_ball_radius(&self) -> f64 {
        let dim = 2;
        let n = self.rtree.size() as f64;
        let ball_radius = self.params.alpha * (n.ln() / n).powi(dim);
        ball_radius
    }

    /// Select parent here as the one giving minimum cost
    pub fn choose_parent(
        &mut self,
        z_new: &RRTNode,
        Z_near: &Vec<RRTNode>,
    ) -> PyResult<(RRTNode, RRTNode)> {
        let mut z_new_ = z_new.clone(); // Contains the current minimum cost for the new node
        let mut z_parent = Z_near[0].clone();
        for z_near in Z_near {
            let (xs_array, _, _, t_new) = self.steer(&z_near, &z_new)?;
            let x_new: Vector6<f64> = xs_array.last().copied().unwrap();
            if self.is_collision_free(&z_near, &x_new) {
                let path_length = utils::compute_path_length(&xs_array);
                let cost = z_near.cost + path_length;
                if cost < z_new_.cost {
                    z_new_ = RRTNode::new(x_new, cost, 0.0, z_near.time + t_new);
                    z_parent = z_near.clone();
                }
            }
        }
        Ok((z_new_, z_parent))
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
        let mut p_rand = Vector2::zeros();
        if self.c_best < f64::INFINITY {
            let p_start: Vector2<f64> = self.x_start.fixed_rows::<2>(0).into();
            let p_goal: Vector2<f64> = self.x_goal.fixed_rows::<2>(0).into();
            p_rand =
                utils::informed_sample(&p_start, &p_goal, self.c_best, &self.enc, &mut self.rng);
        } else {
            p_rand = utils::uniform_sample(&self.enc, &mut self.rng);
        }
        Ok(RRTNode {
            id: None,
            state: Vector6::new(p_rand[0], p_rand[1], 0.0, 0.0, 0.0, 0.0),
            cost: 0.0,
            d2land: 0.0,
            time: 0.0,
        })
    }

    pub fn print_tree(&self) -> PyResult<()> {
        let mut s = String::new();
        self.bookkeeping_tree.write_formatted(&mut s).unwrap();
        println!("Tree: {}", s);
        Ok(())
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
            id: None,
            cost: 0.0,
            d2land: 0.0,
            state: Vector6::zeros(),
            time: 0.0,
        };
        assert_eq!(node.state, Vector6::zeros());
        assert_eq!(node.distance_2(&[1.0, 0.0]), 1.0);
    }

    #[test]
    fn test_rtree() {
        let mut tree = RTree::new();
        tree.insert(RRTNode {
            id: None,
            cost: 0.0,
            d2land: 0.0,
            state: Vector6::zeros(),
            time: 0.0,
        });

        tree.insert(RRTNode {
            id: None,
            cost: 2.0,
            d2land: 40.0,
            state: Vector6::new(50.0, 50.0, 0.0, 0.0, 0.0, 0.0),
            time: 20.0,
        });

        let nearest = tree.nearest_neighbor(&[1.0, 0.0]).unwrap();
        assert_eq!(nearest.state, Vector6::zeros());
    }

    #[test]
    fn test_bookkeeping_tree() {
        let mut tree = Tree::new();
        tree.insert(
            Node::new(RRTNode {
                id: None,
                cost: 0.0,
                d2land: 0.0,
                state: Vector6::zeros(),
                time: 0.0,
            }),
            AsRoot,
        )
        .unwrap();

        let root_id = tree.root_node_id().unwrap().clone();

        tree.get_mut(&root_id)
            .unwrap()
            .data_mut()
            .set_id(root_id.clone());

        println!("Tree root id: {:?}", root_id);

        let child1_id = tree
            .insert(
                Node::new(RRTNode {
                    id: None,
                    cost: 2.0,
                    d2land: 40.0,
                    state: Vector6::new(50.0, 50.0, 0.0, 0.0, 0.0, 0.0),
                    time: 20.0,
                }),
                UnderNode(&root_id),
            )
            .unwrap();

        let child1_child1_id = tree
            .insert(
                Node::new(RRTNode {
                    id: None,
                    cost: 2.0,
                    d2land: 40.0,
                    state: Vector6::new(1000.0, 50.0, 0.0, 0.0, 0.0, 0.0),
                    time: 40.0,
                }),
                UnderNode(&child1_id),
            )
            .unwrap();

        tree.get_mut(&child1_id)
            .unwrap()
            .data_mut()
            .set_id(child1_id.clone());
        tree.get_mut(&child1_child1_id)
            .unwrap()
            .data_mut()
            .set_id(child1_child1_id.clone());

        println!("Root child id: {:?}", child1_id);
        println!("Root child of child1 id: {:?}", child1_child1_id);

        let mut s = String::new();
        tree.write_formatted(&mut s).unwrap();
        println!("Tree: {}", s);
    }

    #[test]
    fn test_sample() -> PyResult<()> {
        let mut rrt = RRTStar::py_new(RRTParams {
            max_iter: 1000,
            max_nodes: 1000,
            goal_radius: 20.0,
            step_size: 1.0,
            max_steering_time: 20.0,
            alpha: 1.0,
        });
        let z_rand = rrt.sample()?;
        assert_eq!(z_rand.state, Vector6::zeros());
        Ok(())
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_choose_parent_and_insert() -> PyResult<()> {
        let mut rrt = RRTStar::py_new(RRTParams {
            max_iter: 1000,
            max_nodes: 1000,
            goal_radius: 20.0,
            step_size: 0.1,
            max_steering_time: 20.0,
            alpha: 1.0,
        });

        let x_start = [0.0, 0.0, 0.0, 5.0, 0.0, 0.0];
        let x_goal = [100.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        Python::with_gil(|py| -> PyResult<()> {
            let x_start_pyany = x_start.into_py(py);
            let x_start_py = x_start_pyany.as_ref(py).downcast::<PyList>().unwrap();
            rrt.set_init_state(&x_start_py)?;
            let x_goal_pyany = x_goal.into_py(py);
            let x_goal_py = x_goal_pyany.as_ref(py).downcast::<PyList>().unwrap();
            rrt.set_goal_state(x_goal_py)?;
            Ok(())
        })?;

        let z_new = RRTNode {
            id: None,
            cost: 100.0,
            d2land: 20.0,
            state: Vector6::new(100.0, 0.0, 0.0, 5.0, 0.0, 0.0),
            time: 20.0,
        };
        let Z_near = rrt.nearest_neighbors(&z_new)?;
        println!("Z_near: {:?}", Z_near);

        let (z_new, z_parent) = rrt.choose_parent(&z_new, &Z_near)?;
        println!("z_new: {:?}", z_new);
        println!("z_parent: {:?}", z_parent);

        rrt.print_tree()?;
        let z_new_id = rrt.insert(&z_new, &z_parent)?;
        rrt.print_tree()?;
        println!("z_new_id: {:?}", z_new_id);
        Ok(())
    }

    #[test]
    fn test_grow_towards_goal() -> PyResult<()> {
        let mut rrt = RRTStar::py_new(RRTParams {
            max_iter: 10,
            max_nodes: 100000,
            goal_radius: 20.0,
            step_size: 0.1,
            max_steering_time: 20.0,
            alpha: 1.0,
        });

        let x_start = [0.0, 0.0, 0.0, 5.0, 0.0, 0.0];
        let x_goal = [300.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        Python::with_gil(|py| -> PyResult<()> {
            let x_start_pyany = x_start.into_py(py);
            let x_start_py = x_start_pyany.as_ref(py).downcast::<PyList>().unwrap();
            let x_goal_pyany = x_goal.into_py(py);
            let x_goal_py = x_goal_pyany.as_ref(py).downcast::<PyList>().unwrap();
            rrt.set_goal_state(x_goal_py)?;

            let do_list = Vec::<[f64; 6]>::new().into_py(py);
            let do_list = do_list.as_ref(py).downcast::<PyList>().unwrap();
            rrt.grow_towards_goal(x_start_py, do_list, false)?;
            Ok(())
        })
    }
}
