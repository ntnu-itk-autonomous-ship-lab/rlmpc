//! # RRT*
//! Contains the main informed RRT* functionality
//!
use crate::common::{RRTNode, RRTResult};
use crate::enc_data::ENCData;
use crate::model::Telemetron;
use crate::steering::{SimpleSteering, Steering};
use crate::utils;
use config::Config;
use id_tree::InsertBehavior::*;
use id_tree::*;
use nalgebra::{Vector2, Vector3, Vector6};
use pyo3::conversion::ToPyObject;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};
use pyo3::FromPyObject;
use rand::SeedableRng;
use rand_chacha::ChaChaRng;
use rstar::{PointDistance, RTree};
use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(FromPyObject, Serialize, Deserialize, Debug, Clone, Copy)]
pub struct InformedRRTParams {
    pub max_nodes: u64,
    pub max_iter: u64,
    pub iter_between_direct_goal_growth: u64,
    pub min_node_dist: f64,
    pub goal_radius: f64,
    pub step_size: f64,
    pub max_steering_time: f64,
    pub steering_acceptance_radius: f64,
    pub max_nn_node_dist: f64, // nearest neighbor max radius parameter
    pub gamma: f64,            // nearest neighbor radius parameter
}

impl InformedRRTParams {
    pub fn default() -> Self {
        Self {
            max_nodes: 10000,
            max_iter: 100000,
            iter_between_direct_goal_growth: 100,
            min_node_dist: 10.0,
            goal_radius: 100.0,
            step_size: 0.2,
            max_steering_time: 20.0,
            steering_acceptance_radius: 5.0,
            max_nn_node_dist: 400.0,
            gamma: 200.0,
        }
    }

    pub fn from_json_value(json: serde_json::Value) -> Self {
        let cfg = serde_json::from_value(json).unwrap();
        cfg
    }

    pub fn from_file(filename: &str) -> Self {
        let cfg = Config::builder()
            .add_source(config::File::with_name(filename))
            .build()
            .unwrap()
            .try_deserialize::<InformedRRTParams>()
            .unwrap();
        cfg
    }

    pub fn to_file(&self, filename: &str) {
        let serialized_cfg = serde_json::to_string(&self).unwrap();
        println!("{}", serialized_cfg);
        serde_json::to_writer_pretty(std::fs::File::create(filename).unwrap(), &self).unwrap();
    }
}

#[allow(non_snake_case)]
#[pyclass]
pub struct InformedRRTStar {
    pub c_best: f64,
    pub z_best_parent: RRTNode,
    pub solutions: Vec<RRTResult>, // (states, times, cost) for each solution
    pub params: InformedRRTParams,
    pub steering: SimpleSteering<Telemetron>,
    pub xs_start: Vector6<f64>,
    pub xs_goal: Vector6<f64>,
    pub U_d: f64,
    pub num_nodes: u64,
    pub rtree: RTree<RRTNode>,
    bookkeeping_tree: Tree<RRTNode>,
    rng: ChaChaRng,
    pub enc: ENCData,
}

#[pymethods]
impl InformedRRTStar {
    #[new]
    pub fn py_new(params: InformedRRTParams) -> Self {
        println!("InformedRRTStar initialized with params: {:?}", params);
        Self {
            c_best: std::f64::INFINITY,
            z_best_parent: RRTNode::new(Vector6::zeros(), 0.0, 0.0, 0.0),
            solutions: Vec::new(),
            params: params.clone(),
            steering: SimpleSteering::new(),
            xs_start: Vector6::zeros(),
            xs_goal: Vector6::zeros(),
            U_d: 5.0,
            num_nodes: 0,
            rtree: RTree::new(),
            bookkeeping_tree: Tree::new(),
            rng: ChaChaRng::from_entropy(),
            enc: ENCData::py_new(),
        }
    }

    #[allow(non_snake_case)]
    fn set_speed_reference(&mut self, U_d: f64) -> PyResult<()> {
        Ok(self.U_d = U_d)
    }

    fn set_init_state(&mut self, xs_start: &PyList) -> PyResult<()> {
        let xs_start_vec = xs_start.extract::<Vec<f64>>()?;
        self.xs_start = Vector6::from_vec(xs_start_vec);

        let root_node = Node::new(RRTNode {
            id: None,
            cost: 0.0,
            d2land: 0.0,
            state: self.xs_start.clone(),
            time: 0.0,
        });
        let root_id = self.bookkeeping_tree.insert(root_node, AsRoot).unwrap();
        let root = self.bookkeeping_tree.get_mut(&root_id).unwrap();
        root.data_mut().set_id(root_id.clone());

        self.rtree.insert(RRTNode {
            id: Some(root_id),
            cost: 0.0,
            d2land: 0.0,
            state: self.xs_start.clone(),
            time: 0.0,
        });
        self.num_nodes += 1;
        Ok(())
    }

    fn set_goal_state(&mut self, xs_goal: &PyList) -> PyResult<()> {
        let xs_goal_vec = xs_goal.extract::<Vec<f64>>()?;
        self.xs_goal = Vector6::from_vec(xs_goal_vec);
        Ok(())
    }

    fn transfer_enc_hazards(&mut self, hazards: &PyAny) -> PyResult<()> {
        self.enc.transfer_enc_hazards(hazards)
    }

    fn transfer_safe_sea_triangulation(&mut self, safe_sea_triangulation: &PyList) -> PyResult<()> {
        self.enc
            .transfer_safe_sea_triangulation(safe_sea_triangulation)
    }

    fn get_tree_as_list_of_dicts(&self, py: Python<'_>) -> PyResult<PyObject> {
        let node_list = PyList::empty(py);
        let root_node_id = self.bookkeeping_tree.root_node_id().unwrap();
        let node_id_int: i64 = 0;
        let parent_id_int: i64 = -1;
        let mut total_num_nodes: i64 = 0;
        self.append_subtree_to_list(
            node_list,
            &root_node_id,
            node_id_int,
            parent_id_int,
            &mut total_num_nodes,
            py,
        )?;
        Ok(node_list.into_py(py))
    }

    #[allow(non_snake_case)]
    fn grow_towards_goal(
        &mut self,
        ownship_state: &PyList,
        U_d: f64,
        do_list: &PyList,
        py: Python<'_>,
    ) -> PyResult<PyObject> {
        let start = Instant::now();
        self.set_speed_reference(U_d)?;
        self.set_init_state(ownship_state)?;
        println!("Ownship state: {:?}", ownship_state);
        println!("Goal state: {:?}", self.xs_goal);
        println!("U_d: {:?}", U_d);
        println!("Do list: {:?}", do_list);

        self.c_best = std::f64::INFINITY;
        self.solutions = Vec::new();
        let mut z_new = self.get_root_node();
        let mut num_iter = 0;
        let goal_attempt_steering_time = 10.0 * 60.0;
        while self.num_nodes < self.params.max_nodes && num_iter < self.params.max_iter {
            if self.attempt_direct_goal_growth(num_iter, self.c_best, goal_attempt_steering_time)? {
                continue;
            }

            if self.goal_reachable(&z_new) {
                let accepted = self.attempt_goal_insertion(
                    &z_new,
                    self.c_best,
                    self.params.max_steering_time,
                )?;
                if accepted {
                    println!(
                        "Num iter: {} | Num nodes: {} | c_best: {}",
                        num_iter, self.num_nodes, self.c_best
                    );
                }
            }

            z_new = RRTNode::default();
            let z_rand = self.sample()?;
            let z_nearest = self.nearest(&z_rand)?;
            let (xs_array, _, _, t_new, _) = self.steer(
                &z_nearest,
                &z_rand,
                self.params.max_steering_time,
                self.params.steering_acceptance_radius,
            )?;
            let xs_new: Vector6<f64> = xs_array.last().copied().unwrap();

            if self.is_too_close(&xs_new) {
                continue;
            }
            if self.is_collision_free(&xs_array) {
                let path_length = utils::compute_path_length(&xs_array);
                z_new = RRTNode::new(
                    xs_new,
                    z_nearest.cost + path_length,
                    0.0,
                    z_nearest.time + t_new,
                );

                // utils::draw_current_situation(
                //     "current_situation.png",
                //     &xs_array,
                //     &self.bookkeeping_tree,
                //     &self.enc,
                // )
                // .unwrap();

                let Z_near = self.nearest_neighbors(&z_new)?;
                let (z_new_, z_parent) = self.choose_parent(&z_new, &z_nearest, &Z_near)?;
                z_new = z_new_;
                z_new = self.insert(&z_new, &z_parent)?;
                self.rewire(&z_new, &Z_near)?;
            }
            num_iter += 1;
        }
        let opt_soln = match self.extract_best_solution() {
            Ok(soln) => soln,
            Err(e) => {
                println!("No solution found. Error msg: {:?}", e);
                return Ok(PyList::empty(py).into_py(py));
            }
        };
        let duration = start.elapsed();
        println!("InformedRRTStar run time: {:?}", duration.as_secs());
        //self.draw_tree(Some(&opt_soln))?;
        Ok(opt_soln.to_object(py))
    }
}

#[allow(non_snake_case)]
impl InformedRRTStar {
    fn append_subtree_to_list(
        &self,
        list: &PyList,
        node_id: &NodeId,
        node_id_int: i64,
        parent_id_int: i64,
        total_num_nodes: &mut i64,
        py: Python<'_>,
    ) -> PyResult<()> {
        let node = self.bookkeeping_tree.get(node_id).unwrap();
        let node_data = node.data().clone();
        let node_dict = PyDict::new(py);
        node_dict.set_item("state", node_data.state.as_slice())?;
        node_dict.set_item("cost", node_data.cost)?;
        node_dict.set_item("d2land", node_data.d2land)?;
        node_dict.set_item("time", node_data.time)?;
        node_dict.set_item("id", node_id_int.clone())?;
        node_dict.set_item("parent_id", parent_id_int.clone())?;
        // println!(
        //     "Node ID: {} | Parent ID: {} | cost: {}",
        //     node_id_int, parent_id_int, node_data.cost
        // );

        *total_num_nodes += 1;
        list.append(node_dict)?;
        let mut children_ids = self.bookkeeping_tree.children_ids(node_id).unwrap();
        let mut child_node_id_int = *total_num_nodes;
        loop {
            let child_id = match children_ids.next() {
                Some(id) => id,
                None => break,
            };
            self.append_subtree_to_list(
                list,
                &child_id,
                child_node_id_int,
                node_id_int,
                total_num_nodes,
                py,
            )?;
            child_node_id_int = *total_num_nodes;
        }
        Ok(())
    }

    // Add a solution if one is found and is better than the current best
    pub fn add_solution(&mut self, z: &RRTNode, z_goal_attempt: &RRTNode) -> PyResult<()> {
        let z_goal_ = self.insert(&z_goal_attempt.clone(), &z)?;
        let soln = self.extract_solution(&z_goal_)?;
        self.solutions.push(soln.clone());
        self.c_best = self.c_best.min(soln.cost);
        self.z_best_parent = z.clone();
        println!("Solution found! Cost: {}", soln.cost);
        Ok(())
    }

    // Find a solution by backtracing from the input node
    pub fn extract_solution(&self, z: &RRTNode) -> PyResult<RRTResult> {
        let mut states: Vec<[f64; 6]> = vec![z.state.clone().into()];
        let mut times = vec![z.time];
        let mut z_current = self.bookkeeping_tree.get(&z.clone().id.unwrap()).unwrap();
        let cost = z_current.data().cost;
        while z_current.parent().is_some() {
            let parent_id = z_current.parent().unwrap();
            let z_parent = self.bookkeeping_tree.get(&parent_id).unwrap();
            states.push(z_parent.data().state.clone().into());
            times.push(z_parent.data().time);
            z_current = z_parent;
        }
        states.push(z_current.data().state.clone().into());
        times.push(z_current.data().time);
        states.reverse();
        times.reverse();
        Ok(RRTResult::new((states, vec![], vec![], times, cost)))
    }

    // Prune state nodes from the solution to make the trajectory smoother and more optimal wrt distance
    fn optimize_solution(&self, soln: &mut RRTResult) -> PyResult<()> {
        soln.save_to_json()?;
        if soln.states.len() < 2 {
            soln.states = vec![];
            return Ok(());
        }
        let mut states: Vec<[f64; 6]> = vec![soln.states.last().unwrap().clone()];
        let mut idx: usize = soln.states.len() - 1;
        while idx > 0 {
            for j in 0..idx {
                let xs_array = vec![
                    soln.states[j].clone().into(),
                    soln.states[idx].clone().into(),
                ];
                let is_collision_free = self.is_collision_free(&xs_array);
                if is_collision_free {
                    states.push(soln.states[j].clone());
                    idx = j;
                    break;
                }
            }
        }
        if states.len() > 1 {
            states.reverse();
            soln.states = states;
        }
        Ok(())
    }

    pub fn is_collision_free(&self, xs_array: &Vec<Vector6<f64>>) -> bool {
        if self.enc.is_empty() {
            return true;
        }
        let is_collision_free = !self.enc.intersects_with_trajectory(&xs_array);
        is_collision_free
    }

    pub fn is_too_close(&self, xs_new: &Vector6<f64>) -> bool {
        let nearest = self
            .rtree
            .nearest_neighbor_iter_with_distance_2(&[xs_new[0], xs_new[1]])
            .next();
        if nearest.is_none() {
            return false;
        }
        let tup = nearest.unwrap();
        let min_dist = self.params.min_node_dist;
        tup.1 <= min_dist.powi(2)
    }

    pub fn goal_reachable(&self, z: &RRTNode) -> bool {
        let x = z.state[0];
        let y = z.state[1];
        let x_goal = self.xs_goal[0];
        let y_goal = self.xs_goal[1];
        let dist_squared = (x - x_goal).powi(2) + (y - y_goal).powi(2);

        dist_squared < self.params.goal_radius.powi(2)
    }

    pub fn attempt_direct_goal_growth(
        &mut self,
        num_iter: u64,
        c_best: f64,
        max_steering_time: f64,
    ) -> PyResult<bool> {
        if num_iter % self.params.iter_between_direct_goal_growth != 0 {
            return Ok(false);
        }
        let z_goal = RRTNode::new(self.xs_goal.clone(), 0.0, 0.0, 0.0);
        let z_nearest = self.nearest(&z_goal)?;
        self.attempt_goal_insertion(&z_nearest, c_best, max_steering_time)
    }

    pub fn attempt_goal_insertion(
        &mut self,
        z: &RRTNode,
        c_best: f64,
        max_steering_time: f64,
    ) -> PyResult<bool> {
        if z.id == self.z_best_parent.id {
            println!("Attempted goal insertion with same node as best parent");
            return Ok(false);
        }
        let mut z_goal_ = RRTNode::new(self.xs_goal.clone(), 0.0, 0.0, 0.0);
        let (xs_array, _, _, t_new, reached) = self.steer(
            &z,
            &z_goal_,
            max_steering_time,
            self.params.steering_acceptance_radius,
        )?;
        let x_new: Vector6<f64> = xs_array.last().copied().unwrap();

        if !(self.is_collision_free(&xs_array) && reached) {
            return Ok(false);
        }
        let cost = z.cost + utils::compute_path_length(&xs_array);
        if cost >= c_best {
            println!(
                "Attempted goal insertion | cost : {} | c_best : {}",
                cost, c_best
            );
            return Ok(false);
        }
        z_goal_ = RRTNode::new(x_new, cost, 0.0, z.time + t_new);
        self.add_solution(&z, &z_goal_)?;
        Ok(true)
    }

    /// Inserts a new node into the tree, with the parent node being z_parent
    /// Since we have two trees (RTree for nearest neighbor search and Tree for keeping track of parents/children),
    /// we need to keep track of the node id in both trees. This is done by setting the id of the node in the Tree
    pub fn insert(&mut self, z: &RRTNode, z_parent: &RRTNode) -> PyResult<RRTNode> {
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
        self.rtree.insert(z_copy.clone());
        self.num_nodes += 1;
        Ok(z_copy)
    }

    fn get_parent_id(&self, z: &RRTNode) -> PyResult<NodeId> {
        let parent_id = self
            .bookkeeping_tree
            .get(&z.clone().id.unwrap())
            .unwrap()
            .parent()
            .unwrap();
        Ok(parent_id.clone())
    }

    pub fn rewire(&mut self, z_new: &RRTNode, Z_near: &Vec<RRTNode>) -> PyResult<()> {
        let z_new_parent_id = self.get_parent_id(&z_new)?;
        for z_near in Z_near.iter() {
            if z_new_parent_id == z_near.clone().id.unwrap() {
                continue;
            }
            let (xs_array, _, _, t_new, reached) = self.steer(
                &z_new,
                &z_near,
                10.0 * self.params.max_steering_time,
                self.params.steering_acceptance_radius,
            )?;
            let xs_new_near: Vector6<f64> = xs_array.last().copied().unwrap();
            if self.is_collision_free(&xs_array) && reached {
                let path_length = utils::compute_path_length(&xs_array);
                let z_new_near = RRTNode::new(
                    xs_new_near,
                    z_new.cost + path_length,
                    0.0,
                    z_new.time + t_new,
                );
                let z_near_id = z_near.clone().id.unwrap();
                if z_new_near.cost < z_near.cost {
                    self.transfer_node_data(&z_near_id, &z_new_near)?;

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

    pub fn nearest(&self, z_rand: &RRTNode) -> PyResult<RRTNode> {
        let nearest = self
            .rtree
            .nearest_neighbor(&z_rand.point())
            .unwrap()
            .clone();
        Ok(nearest)
    }

    pub fn nearest_neighbors(&self, z_new: &RRTNode) -> PyResult<Vec<RRTNode>> {
        let ball_radius = self.compute_nn_radius();
        if self.rtree.size() == 1 {
            let root_id = self.bookkeeping_tree.root_node_id().unwrap();
            let z = self.bookkeeping_tree.get(root_id).unwrap().data().clone();
            return Ok(vec![z]);
        }
        println!("NN radius: {}", ball_radius);

        let mut Z_near = self
            .rtree
            .nearest_neighbor_iter(&z_new.point())
            .take_while(|z| z.distance_2(&z_new.point()) < ball_radius.powi(2))
            .map(|z| z.clone())
            .collect::<Vec<_>>();
        Z_near.sort_by(|a, b| a.cost.partial_cmp(&b.cost).unwrap());
        Ok(Z_near)
    }

    /// Select parent here as the one giving minimum cost
    pub fn choose_parent(
        &mut self,
        z_new: &RRTNode,
        z_nearest: &RRTNode,
        Z_near: &Vec<RRTNode>,
    ) -> PyResult<(RRTNode, RRTNode)> {
        let mut z_new_ = z_new.clone(); // Contains the current minimum cost for the new node
        let z_nearest_id = z_nearest.clone().id.unwrap();
        let mut z_parent = z_nearest.clone();
        if Z_near.is_empty() {
            return Ok((z_new_, z_parent));
        }
        for z_near in Z_near {
            if z_near.id.clone().unwrap() == z_nearest_id {
                continue;
            }
            let (xs_array, _, _, t_new, reached) = self.steer(
                &z_near,
                &z_new,
                10.0 * self.params.max_steering_time,
                self.params.steering_acceptance_radius,
            )?;
            let xs_new: Vector6<f64> = xs_array.last().copied().unwrap();
            if self.is_collision_free(&xs_array) && !self.is_too_close(&xs_new) && reached {
                let path_length = utils::compute_path_length(&xs_array);
                let cost = z_near.cost + path_length;
                if cost < z_new_.cost {
                    z_new_ = RRTNode::new(xs_new, cost, 0.0, z_near.time + t_new);
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
        max_steering_time: f64,
        acceptance_radius: f64,
    ) -> PyResult<(
        Vec<Vector6<f64>>,
        Vec<Vector3<f64>>,
        Vec<(f64, f64)>,
        f64,
        bool,
    )> {
        let (xs_array, u_array, refs_array, t_array, reached) = self.steering.steer(
            &z_nearest.state,
            &z_rand.state,
            self.U_d,
            acceptance_radius,
            self.params.step_size,
            max_steering_time,
        );
        // self.draw_steering_results(
        //     &z_nearest.state,
        //     &z_rand.state,
        //     &refs_array,
        //     &xs_array,
        //     acceptance_radius,
        // )?;
        Ok((
            xs_array,
            u_array,
            refs_array,
            t_array.last().unwrap().clone(),
            reached,
        ))
    }

    pub fn steer_through_solution(&mut self, soln: &RRTResult) -> PyResult<RRTResult> {
        let mut xs_array: Vec<[f64; 6]> = Vec::new();
        let mut u_array: Vec<[f64; 3]> = Vec::new();
        let mut refs_array: Vec<(f64, f64)> = Vec::new();
        let mut t_array: Vec<f64> = Vec::new();
        if soln.states.len() < 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyException, _>(
                "Solution must have at least 2 states",
            ));
        }
        let mut xs_start = soln.states[0].clone();
        for xs_next in soln.states.iter().skip(1) {
            if xs_start == *xs_next {
                continue;
            }
            if xs_array.len() > 0 {
                xs_start = xs_array.last().unwrap().clone();
            }

            let (xs_array_, u_array_, refs_array_, t_array_, reached) = self.steering.steer(
                &xs_start.into(),
                &xs_next.clone().into(),
                self.U_d,
                self.params.steering_acceptance_radius * 3.0,
                self.params.step_size,
                10.0 * 60.0 * self.params.max_steering_time,
            );
            assert_eq!(reached, true);
            xs_array.extend(
                xs_array_
                    .iter()
                    .map(|x| [x[0], x[1], x[2], x[3], x[4], x[5]])
                    .collect::<Vec<[f64; 6]>>(),
            );
            u_array.extend(
                u_array_
                    .iter()
                    .map(|u| [u[0], u[1], u[2]])
                    .collect::<Vec<[f64; 3]>>(),
            );
            refs_array.extend(refs_array_);
            t_array.extend(
                t_array_
                    .iter()
                    .map(|t| {
                        if t_array.len() > 0 {
                            t + t_array.last().unwrap().clone()
                        } else {
                            *t
                        }
                    })
                    .collect::<Vec<f64>>(),
            );
        }
        let new_cost = utils::compute_path_length(
            &xs_array
                .iter()
                .map(|x| Vector6::from(*x))
                .collect::<Vec<Vector6<f64>>>(),
        );
        Ok(RRTResult {
            states: xs_array,
            inputs: u_array,
            references: refs_array,
            times: t_array,
            cost: new_cost,
        })
    }

    pub fn sample(&mut self) -> PyResult<RRTNode> {
        let mut p_rand = Vector2::zeros();
        let p_start: Vector2<f64> = self.xs_start.fixed_rows::<2>(0).into();
        let p_goal: Vector2<f64> = self.xs_goal.fixed_rows::<2>(0).into();
        // let mut map_bbox = self.enc.bbox.clone();
        let map_bbox = utils::bbox_from_corner_points(&p_start, &p_goal, 500.0, 5000.0);
        // println!("Map bbox: {:?}", map_bbox);
        loop {
            if self.c_best < f64::INFINITY {
                p_rand = utils::informed_sample(&p_start, &p_goal, self.c_best, &mut self.rng);
                //println!("Informed sample: {:?}", p_rand);
            } else if !self.enc.safe_sea_triangulation.is_empty() {
                p_rand = utils::sample_from_triangulation(
                    &self.enc.safe_sea_triangulation,
                    &mut self.rng,
                );
                //println!("Sampled from triangulation: {:?}", p_rand);
            } else {
                p_rand = utils::sample_from_bbox(&map_bbox, &mut self.rng);
            }
            //println!("Sampled: {:?}", p_rand);
            if !self.enc.inside_hazards(&p_rand) && self.enc.inside_bbox(&p_rand) {
                // println!("Sampled outside hazard");
                return Ok(RRTNode {
                    id: None,
                    state: Vector6::new(p_rand[0], p_rand[1], 0.0, 0.0, 0.0, 0.0),
                    cost: 0.0,
                    d2land: 0.0,
                    time: 0.0,
                });
            } else {
                println!("Sampled inside hazard");
            }
        }
    }

    pub fn draw_tree(&self, soln: Option<&RRTResult>) -> PyResult<()> {
        let p_start = self.xs_start.fixed_rows::<2>(0).into();
        let p_goal = self.xs_goal.fixed_rows::<2>(0).into();

        let xs_soln_array = match soln {
            Some(s) => Some(s.states.as_ref()),
            None => None,
        };
        let res = utils::draw_tree(
            "tree.png",
            &self.bookkeeping_tree,
            &p_start,
            &p_goal,
            xs_soln_array,
            &self.enc,
        );
        return res.map_err(|e| utils::map_err_to_pyerr(e));
    }

    fn draw_steering_results(
        &self,
        xs_start: &Vector6<f64>,
        xs_goal: &Vector6<f64>,
        refs_array: &Vec<(f64, f64)>,
        xs_array: &Vec<Vector6<f64>>,
        acceptance_radius: f64,
    ) -> PyResult<()> {
        let res = utils::draw_steering_results(
            &xs_start,
            &xs_goal,
            &refs_array,
            &xs_array,
            acceptance_radius,
        );
        return res.map_err(|e| utils::map_err_to_pyerr(e));
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

    /// Compute nearest neightbours radius as in RRT* by Karaman and Frazzoli, used for search and sampling
    fn compute_nn_radius(&self) -> f64 {
        let dim = 2;
        let n = self.rtree.size() as f64;
        let ball_radius = self.params.gamma * (n.ln() / n).powf(1.0 / dim as f64);
        ball_radius.min(self.params.max_nn_node_dist)
    }

    fn extract_best_solution(&mut self) -> PyResult<RRTResult> {
        let mut opt_soln = self.solutions.iter().fold(
            RRTResult::new((vec![], vec![], vec![], vec![], std::f64::INFINITY)),
            |acc, x| {
                if x.cost < acc.cost {
                    x.clone()
                } else {
                    acc
                }
            },
        );
        self.optimize_solution(&mut opt_soln)?;
        self.steer_through_solution(&opt_soln)
    }

    fn get_root_node(&self) -> RRTNode {
        let root_id = self.bookkeeping_tree.root_node_id().unwrap();
        let root_node = self.bookkeeping_tree.get(&root_id).unwrap();
        root_node.data().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample() -> PyResult<()> {
        let mut rrt = InformedRRTStar::py_new(InformedRRTParams {
            max_nodes: 1000,
            max_iter: 100000,
            iter_between_direct_goal_growth: 100,
            min_node_dist: 20.0,
            goal_radius: 100.0,
            step_size: 1.0,
            max_steering_time: 20.0,
            steering_acceptance_radius: 5.0,
            gamma: 200.0,
            max_nn_node_dist: 100.0,
        });
        let z_rand = rrt.sample()?;
        assert_eq!(z_rand.state, Vector6::zeros());
        Ok(())
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_choose_parent_and_insert() -> PyResult<()> {
        let mut rrt = InformedRRTStar::py_new(InformedRRTParams {
            max_nodes: 1000,
            max_iter: 100000,
            iter_between_direct_goal_growth: 100,
            min_node_dist: 50.0,
            goal_radius: 100.0,
            step_size: 0.1,
            max_steering_time: 20.0,
            steering_acceptance_radius: 5.0,
            gamma: 200.0,
            max_nn_node_dist: 150.0,
        });

        let xs_start = [0.0, 0.0, 0.0, 5.0, 0.0, 0.0];
        let xs_goal = [100.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        Python::with_gil(|py| -> PyResult<()> {
            let xs_start_pyany = xs_start.into_py(py);
            let xs_start_py = xs_start_pyany.as_ref(py).downcast::<PyList>().unwrap();
            rrt.set_init_state(&xs_start_py)?;
            let xs_goal_pyany = xs_goal.into_py(py);
            let xs_goal_py = xs_goal_pyany.as_ref(py).downcast::<PyList>().unwrap();
            rrt.set_goal_state(xs_goal_py)?;
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

        let (z_new, z_parent) = rrt.choose_parent(&z_new, &Z_near[0].clone(), &Z_near)?;
        println!("z_new: {:?}", z_new);
        println!("z_parent: {:?}", z_parent);

        let z_new_id = rrt.insert(&z_new, &z_parent)?;
        println!("z_new_id: {:?}", z_new_id);
        Ok(())
    }

    #[test]
    fn test_optimize_solution() -> PyResult<()> {
        let mut rrt = InformedRRTStar::py_new(InformedRRTParams {
            max_nodes: 1700,
            max_iter: 10000,
            iter_between_direct_goal_growth: 100,
            min_node_dist: 30.0,
            goal_radius: 600.0,
            step_size: 0.5,
            max_steering_time: 25.0,
            steering_acceptance_radius: 5.0,
            gamma: 1200.0,
            max_nn_node_dist: 200.0,
        });
        let mut soln = RRTResult {
            states: vec![],
            references: vec![],
            inputs: vec![],
            times: vec![],
            cost: 0.0,
        };
        rrt.enc.load_hazards_from_json()?;
        soln.load_from_json()?;
        println!("soln length: {}", soln.states.len());
        rrt.optimize_solution(&mut soln)?;
        println!("optimized soln length: {}", soln.states.len());
        soln = rrt.steer_through_solution(&soln)?;
        Python::with_gil(|py| -> PyResult<()> {
            let _soln_py = soln.to_object(py);
            Ok(())
        })?;
        Ok(())
    }
    #[test]
    fn test_grow_towards_goal() -> PyResult<()> {
        let mut rrt = InformedRRTStar::py_new(InformedRRTParams {
            max_nodes: 2000,
            max_iter: 10000,
            iter_between_direct_goal_growth: 100,
            min_node_dist: 10.0,
            goal_radius: 10.0,
            step_size: 0.5,
            max_steering_time: 15.0,
            steering_acceptance_radius: 5.0,
            gamma: 1200.0,
            max_nn_node_dist: 125.0,
        });
        let xs_start = [
            6581590.0,
            -33715.0,
            120.0 * std::f64::consts::PI / 180.0,
            4.0,
            0.0,
            0.0,
        ];
        let xs_goal = [
            6581780.0,
            -32670.0,
            -30.0 * std::f64::consts::PI / 180.0,
            0.0,
            0.0,
            0.0,
        ];
        // let xs_start = [
        //     6574280.0,
        //     -31824.0,
        //     -45.0 * std::f64::consts::PI / 180.0,
        //     5.0,
        //     0.0,
        //     0.0,
        // ];
        // let xs_goal = [6583580.0, -31824.0, 0.0, 0.0, 0.0, 0.0];
        rrt.enc.load_hazards_from_json()?;
        rrt.enc.load_safe_sea_triangulation_from_json()?;
        Python::with_gil(|py| -> PyResult<()> {
            let xs_start_pyany = xs_start.into_py(py);
            let xs_start_py = xs_start_pyany.as_ref(py).downcast::<PyList>().unwrap();
            let xs_goal_pyany = xs_goal.into_py(py);
            let xs_goal_py = xs_goal_pyany.as_ref(py).downcast::<PyList>().unwrap();
            rrt.set_goal_state(xs_goal_py)?;
            rrt.set_speed_reference(6.0)?;

            let do_list = Vec::<[f64; 6]>::new().into_py(py);
            let do_list = do_list.as_ref(py).downcast::<PyList>().unwrap();
            let result = rrt.grow_towards_goal(xs_start_py, 6.0, do_list, py)?;
            let pydict = result.as_ref(py).downcast::<PyDict>().unwrap();
            println!("rrtresult states: {:?}", pydict.get_item("states"));
            Ok(())
        })
    }
}
