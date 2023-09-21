//! # PQ-RRT*
//! Contains the main PQ-RRT* functionality
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
pub struct PQRRTParams {
    pub max_nodes: u64,
    pub max_iter: u64,
    pub iter_between_direct_goal_growth: u64,
    pub min_node_dist: f64,
    pub min_steering_time: f64,
    pub goal_radius: f64,
    pub step_size: f64,
    pub max_steering_time: f64,
    pub steering_acceptance_radius: f64,
    pub max_nn_node_dist: f64, // nearest neighbor max radius parameter
    pub gamma: f64,            // nearest neighbor radius parameter
    pub max_sample_adjustments: u64,
    pub lambda_sample_adjustment: f64, // sample adjustment parameter
    pub safe_distance: f64,            // safe distance to obstacles
    pub max_ancestry_level: u64, // max number of levels to go up in the ancestry tree when choosing parent in pq-rrt*
}

impl PQRRTParams {
    pub fn default() -> Self {
        Self {
            max_nodes: 10000,
            max_iter: 100000,
            iter_between_direct_goal_growth: 100,
            min_node_dist: 10.0,
            min_steering_time: 5.0,
            goal_radius: 100.0,
            step_size: 0.2,
            max_steering_time: 20.0,
            steering_acceptance_radius: 5.0,
            max_nn_node_dist: 400.0,
            gamma: 200.0,
            max_sample_adjustments: 50,
            lambda_sample_adjustment: 0.5,
            safe_distance: 25.0,
            max_ancestry_level: 4,
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
            .try_deserialize::<PQRRTParams>()
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
pub struct PQRRTStar {
    pub c_best: f64,
    pub z_best_parent: RRTNode,
    pub solutions: Vec<RRTResult>, // (states, times, cost) for each solution
    pub params: PQRRTParams,
    pub steering: SimpleSteering<Telemetron>,
    pub xs_start: Vector6<f64>,
    pub xs_goal: Vector6<f64>,
    pub U_d: f64,
    pub num_nodes: u64,
    pub num_iter: u64,
    pub rtree: RTree<RRTNode>,
    bookkeeping_tree: Tree<RRTNode>,
    rng: ChaChaRng,
    pub enc: ENCData,
}

#[pymethods]
impl PQRRTStar {
    #[new]
    pub fn py_new(params: PQRRTParams) -> Self {
        println!("PQRRTStar initialized with params: {:?}", params);
        Self {
            c_best: std::f64::INFINITY,
            z_best_parent: RRTNode::new(Vector6::zeros(), Vec::new(), Vec::new(), 0.0, 0.0, 0.0),
            solutions: Vec::new(),
            params: params.clone(),
            steering: SimpleSteering::new(),
            xs_start: Vector6::zeros(),
            xs_goal: Vector6::zeros(),
            U_d: 5.0,
            num_nodes: 0,
            num_iter: 0,
            rtree: RTree::new(),
            bookkeeping_tree: Tree::new(),
            rng: ChaChaRng::from_entropy(),
            enc: ENCData::py_new(),
        }
    }

    #[allow(non_snake_case)]
    pub fn set_speed_reference(&mut self, U_d: f64) -> PyResult<()> {
        Ok(self.U_d = U_d)
    }

    pub fn set_init_state(&mut self, xs_start: &PyList) -> PyResult<()> {
        let xs_start_vec = xs_start.extract::<Vec<f64>>()?;
        self.xs_start = Vector6::from_vec(xs_start_vec);

        let root_node = Node::new(RRTNode {
            id: None,
            cost: 0.0,
            d2land: 0.0,
            state: self.xs_start.clone(),
            trajectory: Vec::new(),
            controls: Vec::new(),
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
            trajectory: vec![self.xs_start.clone()],
            controls: Vec::new(),
            time: 0.0,
        });
        self.num_nodes += 1;
        Ok(())
    }

    pub fn set_goal_state(&mut self, xs_goal: &PyList) -> PyResult<()> {
        let xs_goal_vec = xs_goal.extract::<Vec<f64>>()?;
        self.xs_goal = Vector6::from_vec(xs_goal_vec);
        Ok(())
    }

    pub fn update_parameters(&mut self, params: PQRRTParams) -> PyResult<()> {
        self.params = params.clone();
        Ok(())
    }

    pub fn transfer_enc_hazards(&mut self, hazards: &PyAny) -> PyResult<()> {
        self.enc.transfer_enc_hazards(hazards)
    }

    pub fn transfer_safe_sea_triangulation(
        &mut self,
        safe_sea_triangulation: &PyList,
    ) -> PyResult<()> {
        self.enc
            .transfer_safe_sea_triangulation(safe_sea_triangulation)
    }

    pub fn get_tree_as_list_of_dicts(&self, py: Python<'_>) -> PyResult<PyObject> {
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
    pub fn grow_towards_goal(
        &mut self,
        ownship_state: &PyList,
        U_d: f64,
        do_list: &PyList,
        py: Python<'_>,
    ) -> PyResult<PyObject> {
        let start = Instant::now();
        self.set_speed_reference(U_d)?;
        self.set_init_state(ownship_state)?;
        // println!("Ownship state: {:?}", ownship_state);
        // println!("Goal state: {:?}", self.xs_goal);
        // println!("U_d: {:?}", U_d);
        // println!("Do list: {:?}", do_list);

        self.c_best = std::f64::INFINITY;
        self.solutions = Vec::new();

        let mut z_new = self.get_root_node();
        self.num_iter = 0;
        let goal_attempt_steering_time = 10.0 * 60.0;
        while self.num_nodes < self.params.max_nodes && self.num_iter < self.params.max_iter {
            self.attempt_direct_goal_growth(goal_attempt_steering_time)?;

            if self.goal_reachable(&z_new) {
                self.attempt_goal_insertion(&z_new, self.params.max_steering_time)?;
            }

            z_new = RRTNode::default();
            let z_rand = self.sample()?;
            let z_prand = self.adjust_sample(&z_rand)?;

            let z_nearest = self.nearest(&z_prand)?;
            let (xs_array, u_array, _, t_new, _) = self.steer(
                &z_nearest,
                &z_prand,
                self.params.max_steering_time,
                self.params.steering_acceptance_radius,
            )?;
            let xs_new: Vector6<f64> = xs_array.last().copied().unwrap();
            if self.is_collision_free(&xs_array)
                && t_new > self.params.min_steering_time
                && !self.is_too_close_to_neighbours(&xs_new, &None)
                && !self.went_full_loop_backwards(&xs_array)
            {
                let path_length = utils::compute_path_length(&xs_array);
                z_new = RRTNode::new(
                    xs_new,
                    xs_array.clone(),
                    u_array,
                    z_nearest.cost + path_length,
                    0.0,
                    z_nearest.time + t_new,
                );

                let mut Z_near = self.nearest_neighbors(&z_new)?;
                let Z_parents = self.ancestry(&Z_near)?;
                Z_near.extend(Z_parents);
                let (z_new_, z_parent) = self.choose_parent(&z_new, &z_nearest, &Z_near)?;
                z_new = z_new_;
                z_new = self.insert(&z_new, &z_parent)?;
                self.rewire(&z_new, &Z_near)?;
            }
            self.num_iter += 1;
            if self.num_iter % 5000 == 0 {
                println!(
                    "Num iter: {} | Num nodes: {} | c_best: {}",
                    self.num_iter, self.num_nodes, self.c_best
                );
            }
        }
        let opt_soln = match self.extract_best_solution() {
            Ok(soln) => soln,
            Err(e) => {
                println!("No solution found. Error msg: {:?}", e);
                return Ok(PyList::empty(py).into_py(py));
            }
        };
        let duration = start.elapsed();
        println!("PQRRT* runtime: {:?}", duration.as_millis() as f64 / 1000.0);
        //self.draw_tree(Some(&opt_soln))?;
        Ok(opt_soln.to_object(py))
    }
}

#[allow(non_snake_case)]
impl PQRRTStar {
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
        let pytrajectory = PyList::new(
            py,
            node_data
                .trajectory
                .clone()
                .into_iter()
                .map(|x| x.into())
                .collect::<Vec<[f64; 6]>>(),
        );

        node_dict.set_item("state", node_data.state.as_slice())?;
        node_dict.set_item("trajectory", pytrajectory)?;
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
        println!(
            "Solution Found! Num iter: {} | Num nodes: {} | c_best: {}",
            self.num_iter, self.num_nodes, self.c_best
        );
        Ok(())
    }

    // Find a solution by backtracking from the input node
    pub fn extract_solution(&self, z: &RRTNode) -> PyResult<RRTResult> {
        let mut z_current = self.bookkeeping_tree.get(&z.clone().id.unwrap()).unwrap();
        let cost = z_current.data().cost;
        let mut node_states: Vec<[f64; 6]> = vec![z_current.data().clone().state.into()];
        let mut trajectories: Vec<Vec<[f64; 6]>> =
            vec![z.trajectory.clone().into_iter().map(|x| x.into()).collect()];
        let mut controls: Vec<Vec<[f64; 3]>> =
            vec![z.controls.clone().into_iter().map(|x| x.into()).collect()];
        while z_current.parent().is_some() {
            let parent_id = z_current.parent().unwrap();
            let z_parent = self.bookkeeping_tree.get(&parent_id).unwrap();
            node_states.push(z_parent.data().state.clone().into());
            trajectories.push(
                z_parent
                    .data()
                    .trajectory
                    .clone()
                    .into_iter()
                    .map(|x| x.into())
                    .collect(),
            );
            controls.push(
                z_parent
                    .data()
                    .controls
                    .clone()
                    .into_iter()
                    .map(|x| x.into())
                    .collect(),
            );
            z_current = z_parent;
        }
        node_states.reverse();
        let states = trajectories
            .iter()
            .rev()
            .flatten()
            .map(|x| *x)
            .collect::<Vec<[f64; 6]>>();
        let inputs = controls
            .iter()
            .rev()
            .flatten()
            .map(|x| *x)
            .collect::<Vec<[f64; 3]>>();
        let times = Vec::from_iter((0..states.len()).map(|i| i as f64 * self.params.step_size));
        Ok(RRTResult::new((node_states, inputs, times, cost)))
    }

    // Prune state nodes from the solution to make the trajectory smoother and more optimal wrt distance
    fn optimize_solution(&mut self, soln: &mut RRTResult) -> PyResult<()> {
        //soln.save_to_json()?;
        if soln.states.len() < 2 {
            soln.states = vec![];
            return Ok(());
        }
        // let mut states: Vec<[f64; 6]> = vec![soln.states.last().unwrap().clone()];
        // let mut idx: usize = soln.states.len() - 1;
        // while idx > 0 {
        //     for j in 0..idx {
        //         let state_j = Vector6::from_vec(soln.states[j].clone().to_vec());
        //         let state_idx = Vector6::from_vec(soln.states[idx].clone().to_vec());
        //         let (xs_array, _, _, _, reached) = self.steering.steer(
        //             &state_j,
        //             &state_idx,
        //             self.U_d,
        //             self.params.steering_acceptance_radius,
        //             self.params.step_size,
        //             10.0 * self.params.max_steering_time,
        //         );
        //         let is_coll = self.is_collision_free(&xs_array);
        //         if (is_coll && reached) || j == idx - 1 {
        //             states.push(soln.states[j].clone());
        //             idx = j;
        //             break;
        //         }
        //     }
        // }
        // assert_eq!(
        //     states.len() > 1,
        //     true,
        //     "Optimized solution has less than 2 states",
        // );
        // states.reverse();
        *soln = self.steer_through_waypoints(&soln.states)?;
        Ok(())
    }

    pub fn distance_to_obstacle(&self, xs: &Vector6<f64>) -> f64 {
        if self.enc.is_empty() {
            return std::f64::INFINITY;
        }
        self.enc.dist2point(&Vector2::new(xs[0], xs[1]))
    }

    pub fn is_collision_free(&self, xs_array: &Vec<Vector6<f64>>) -> bool {
        if self.enc.is_empty() {
            return true;
        }
        if !self.enc.array_inside_bbox(&xs_array) {
            return false;
        }
        let is_collision_free = !self.enc.intersects_with_trajectory(&xs_array);
        is_collision_free
    }

    pub fn is_too_close_to_neighbours(
        &self,
        xs_new: &Vector6<f64>,
        ids_to_exclude: &Option<Vec<NodeId>>,
    ) -> bool {
        let nearest = self
            .rtree
            .nearest_neighbor_iter_with_distance_2(&[xs_new[0], xs_new[1]])
            .next();
        let tup = nearest.unwrap();
        if let Some(ids) = ids_to_exclude {
            if ids.contains(&tup.0.id.clone().unwrap()) {
                return false;
            }
        }
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

    pub fn attempt_direct_goal_growth(&mut self, max_steering_time: f64) -> PyResult<bool> {
        if self.num_iter % self.params.iter_between_direct_goal_growth != 0 {
            return Ok(false);
        }
        let z_goal = RRTNode::new(self.xs_goal.clone(), Vec::new(), Vec::new(), 0.0, 0.0, 0.0);
        let z_nearest = self.nearest(&z_goal)?;
        self.attempt_goal_insertion(&z_nearest, max_steering_time)
    }

    pub fn attempt_goal_insertion(
        &mut self,
        z: &RRTNode,
        max_steering_time: f64,
    ) -> PyResult<bool> {
        if z.id == self.z_best_parent.id {
            println!("Attempted goal insertion with same node as best parent");
            return Ok(false);
        }
        let mut z_goal_ = RRTNode::new(self.xs_goal.clone(), Vec::new(), Vec::new(), 0.0, 0.0, 0.0);
        let (xs_array, u_array, _, t_new, reached) =
            self.steer(&z, &z_goal_, max_steering_time, 5.0)?;
        let x_new: Vector6<f64> = xs_array.last().copied().unwrap();

        if !(self.is_collision_free(&xs_array)
            && t_new > self.params.min_steering_time
            && !self.went_full_loop_backwards(&xs_array)
            && reached)
        {
            return Ok(false);
        }
        let cost = z.cost + utils::compute_path_length(&xs_array);
        if cost >= self.c_best {
            println!(
                "Attempted goal insertion | cost : {} | c_best : {}",
                cost, self.c_best
            );
            return Ok(false);
        }
        z_goal_ = RRTNode::new(x_new, xs_array, u_array, cost, 0.0, z.time + t_new);
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
        let parent_id = match self
            .bookkeeping_tree
            .get(&z.clone().id.unwrap())
            .unwrap()
            .parent()
        {
            Some(id) => id,
            None => {
                return Err(PyErr::new::<pyo3::exceptions::PyException, _>(
                    "z does not have any parents",
                ))
            }
        };
        Ok(parent_id.clone())
    }

    pub fn non_feasible_steer(&self, z_start: &RRTNode, z_end: &RRTNode) -> bool {
        // If its new node is too close and the angle between the two nodes is too large, skip
        let p_end = Vector2::new(z_end.state[0], z_end.state[1]);
        let p_start = Vector2::new(z_start.state[0], z_start.state[1]);
        let los = (p_end[1] - p_start[0]).atan2(p_end[0] - p_start[0]);
        (p_start - p_end).norm() < self.params.min_node_dist
            && los.abs() * 180.0 / std::f64::consts::PI > 90.0
    }

    pub fn rewire(&mut self, z_new: &RRTNode, Z_near: &Vec<RRTNode>) -> PyResult<()> {
        let z_new_parent_id = self.get_parent_id(&z_new)?;
        for z_near in Z_near.iter() {
            let z_near_id = z_near.clone().id.unwrap();
            if z_new_parent_id == z_near_id || self.non_feasible_steer(&z_new, &z_near) {
                continue;
            }
            // let mut waypoints: Vec<Vector6<f64>> = vec![z_new.state.clone()];
            // let x_pad = z_near.state[0] - 30.0 * f64::cos(z_near.state[2]);
            // let y_pad = z_near.state[1] - 30.0 * f64::sin(z_near.state[2]);
            // let xs_middle = Vector6::new(x_pad, y_pad, z_near.state[2], z_near.state[3], 0.0, 0.0);
            // waypoints.push(xs_middle);
            // waypoints.push(z_near.state.clone());
            // let (xs_array, u_array, _, t_array, reached) = self.steering.steer_through_waypoints(
            //     &z_new.state.clone(),
            //     &waypoints,
            //     self.U_d,
            //     self.params.steering_acceptance_radius,
            //     self.params.step_size,
            //     5.0 * self.params.max_steering_time,
            // );
            //let t_new = t_array.last().copied().unwrap();
            let (xs_array, u_array, _, t_new, reached) = self.steer(
                &z_new.clone(),
                &z_near.clone(),
                5.0 * self.params.max_steering_time,
                1.0,
            )?;
            let xs_new_near: Vector6<f64> = xs_array.last().copied().unwrap();
            let path_length = utils::compute_path_length(&xs_array);
            let z_new_near = RRTNode::new(
                xs_new_near,
                xs_array.clone(),
                u_array,
                z_new.cost + path_length,
                0.0,
                z_new.time + t_new,
            );
            if self.is_collision_free(&xs_array)
                && !self.went_full_loop_backwards(&xs_array)
                && !self.is_too_close_to_neighbours(&xs_new_near, &Some(vec![z_near_id.clone()]))
                && t_new > self.params.min_steering_time
                && reached
                && z_new_near.cost < z_near.cost
            {
                self.rtree.remove(z_near);
                // let p_near = Vector2::new(z_near.state[0], z_near.state[1]);
                // let p_new_near = Vector2::new(z_new_near.state[0], z_new_near.state[1]);
                // println!(
                //     "Distance z_near and z_new_near: {}",
                //     (p_near - p_new_near).norm()
                // );
                self.transfer_node_data(&z_near_id, &z_new_near)?;
                self.move_node(&z_near_id, &z_new.clone().id.unwrap())?;
                self.rtree.insert(
                    self.bookkeeping_tree
                        .get(&z_near_id)
                        .unwrap()
                        .data()
                        .clone(),
                );

                // utils::draw_current_situation(
                //     "current_situation.png",
                //     &xs_array.clone(),
                //     &self.bookkeeping_tree,
                //     &self.enc,
                // )
                // .unwrap();
            }
        }
        Ok(())
    }

    pub fn nearest(&mut self, z_rand: &RRTNode) -> PyResult<RRTNode> {
        let nearest = self
            .rtree
            .nearest_neighbor(&z_rand.point())
            .unwrap()
            .clone();
        Ok(nearest)
    }

    fn nearest_neighbors(&self, z_new: &RRTNode) -> PyResult<Vec<RRTNode>> {
        let ball_radius = self.compute_nn_radius();
        if self.rtree.size() == 1 {
            let root_id = self.bookkeeping_tree.root_node_id().unwrap();
            let z = self.bookkeeping_tree.get(root_id).unwrap().data().clone();
            return Ok(vec![z]);
        }
        // println!("NN radius: {}", ball_radius);

        let mut Z_near = self
            .rtree
            .nearest_neighbor_iter(&z_new.point())
            .take_while(|z| z.distance_2(&z_new.point()) < ball_radius.powi(2))
            .map(|z| z.clone())
            .collect::<Vec<_>>();
        Z_near.sort_by(|a, b| a.cost.partial_cmp(&b.cost).unwrap());
        Ok(Z_near)
    }

    /// Find the ancestor of z at level d. If no d-level ancestor, return the highest lvl ancestor. If the node has no ancestor (is root) return error on second line
    pub fn ancestor(&self, z: &RRTNode, d: u64) -> PyResult<RRTNode> {
        if d == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyException, _>(
                "No valid ancestor for d = 0",
            ));
        }
        let root_id = self.bookkeeping_tree.root_node_id().unwrap();
        let mut z_parent_id = self.get_parent_id(&z)?;
        let mut z_parent = self.bookkeeping_tree.get(&z_parent_id).unwrap().data();
        for i in 0..d - 1 {
            if z_parent_id == *root_id && i < d - 1 {
                return Ok(self.bookkeeping_tree.get(&root_id).unwrap().data().clone());
            }
            z_parent_id = self.get_parent_id(&z_parent)?;
            z_parent = self.bookkeeping_tree.get(&z_parent_id).unwrap().data();
        }
        Ok(z_parent.clone())
    }

    pub fn ancestry(&self, Z: &Vec<RRTNode>) -> PyResult<Vec<RRTNode>> {
        let mut Z_parents = Vec::new();
        for z in Z {
            let z_d_level_ancestor = match self.ancestor(&z, self.params.max_ancestry_level) {
                Ok(z) => z,
                Err(_) => continue,
            };
            let ids_already_in_vec = Z_parents
                .iter()
                .any(|z: &RRTNode| z.clone().id.unwrap() == z_d_level_ancestor.clone().id.unwrap());
            if ids_already_in_vec {
                continue;
            }
            Z_parents.push(z_d_level_ancestor);
        }
        Ok(Z_parents)
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

            if self.non_feasible_steer(&z_near, &z_new) {
                continue;
            }

            let (xs_array, u_array, _, t_new, reached) = self.steer(
                &z_near,
                &z_new,
                10.0 * self.params.max_steering_time,
                self.params.steering_acceptance_radius,
            )?;
            let xs_new: Vector6<f64> = xs_array.last().copied().unwrap();
            if self.is_collision_free(&xs_array)
                && t_new > self.params.min_steering_time
                && !self.is_too_close_to_neighbours(&xs_new, &None)
                && !self.went_full_loop_backwards(&xs_array)
                && reached
            {
                let path_length = utils::compute_path_length(&xs_array);
                let cost = z_near.cost + path_length;
                if cost < z_new_.cost {
                    z_new_ =
                        RRTNode::new(xs_new, xs_array, u_array, cost, 0.0, z_near.time + t_new);
                    z_parent = z_near.clone();
                }
            }
        }
        Ok((z_new_, z_parent))
    }

    pub fn went_full_loop_backwards(&self, xs_array: &Vec<Vector6<f64>>) -> bool {
        let n_states = xs_array.len();
        if n_states < 2 {
            return false;
        }
        let x0 = xs_array[0];
        let xend = xs_array[n_states - 1];
        let psi_diffs = xs_array
            .iter()
            .map(|x| utils::wrap_angle_diff_to_pmpi(x[2], x0[2]).abs())
            .collect::<Vec<f64>>();
        let max_psi_diff = *psi_diffs
            .iter()
            .max_by(|x, y| x.partial_cmp(y).unwrap())
            .unwrap();
        let d_0end = ((x0[0] - xend[0]).powi(2) + (x0[1] - xend[1]).powi(2)).sqrt();
        // return false;

        if utils::wrap_angle_to_pmpi(max_psi_diff).abs() * 180.0 / std::f64::consts::PI > 120.0
            && d_0end < 100.0
        {
            // println!(
            //     "Max psi diff: {} | d_0end: {}",
            //     utils::wrap_angle_to_pmpi(max_psi_diff) * 180.0 / std::f64::consts::PI,
            //     d_0end
            // );
            return true;
        }
        false
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
        Ok((
            xs_array,
            u_array,
            refs_array,
            t_array.last().unwrap().clone(),
            reached,
        ))
    }

    pub fn steer_through_waypoints(&mut self, waypoints: &Vec<[f64; 6]>) -> PyResult<RRTResult> {
        let n_wps = waypoints.len();
        if n_wps < 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyException, _>(
                "Must be atleast two waypoints",
            ));
        }
        let (xs_array, u_array, _, t_array, reached_last) = self.steering.steer_through_waypoints(
            &self.xs_start,
            &waypoints
                .clone()
                .into_iter()
                .map(|x| Vector6::from(x))
                .collect(),
            self.U_d,
            self.params.steering_acceptance_radius,
            self.params.step_size,
            10.0 * self.params.max_steering_time,
        );
        assert_eq!(reached_last, true);

        let new_cost = utils::compute_path_length(
            &xs_array
                .iter()
                .map(|x| Vector6::from(*x))
                .collect::<Vec<Vector6<f64>>>(),
        );
        Ok(RRTResult {
            states: xs_array.clone().into_iter().map(|x| x.into()).collect(),
            inputs: u_array.clone().into_iter().map(|u| u.into()).collect(),
            times: t_array,
            cost: new_cost,
        })
    }

    // Uses a potential-field based method to adjust the sample towards the attractive force from the goal region
    pub fn adjust_sample(&mut self, z_rand: &RRTNode) -> PyResult<RRTNode> {
        let mut z_prand = z_rand.clone();
        for _n in 0..self.params.max_sample_adjustments {
            let F_att = self.xs_goal - z_prand.state.clone();
            let d_min = self.distance_to_obstacle(&z_prand.state);
            if d_min <= self.params.safe_distance {
                break;
            }
            z_prand.state =
                z_prand.state + self.params.lambda_sample_adjustment * F_att.normalize();
        }
        Ok(z_prand)
    }

    pub fn sample(&mut self) -> PyResult<RRTNode> {
        let p_start: Vector2<f64> = self.xs_start.fixed_rows::<2>(0).into();
        let p_goal: Vector2<f64> = self.xs_goal.fixed_rows::<2>(0).into();
        let mut map_bbox = self.enc.bbox.clone();
        map_bbox = utils::bbox_from_corner_points(&p_start, &p_goal, 500.0, 500.0);
        // println!("Map bbox: {:?}", map_bbox);
        loop {
            let p_rand = if !self.enc.safe_sea_triangulation.is_empty() {
                // println!("Sampled from triangulation!");
                utils::sample_from_triangulation(&self.enc.safe_sea_triangulation, &mut self.rng)
            } else {
                utils::sample_from_bbox(&map_bbox, &mut self.rng)
            };
            // println!("Sampled: {:?}", p_rand);
            if !self.enc.inside_hazards(&p_rand) && self.enc.inside_bbox(&p_rand) {
                // println!("Sampled outside hazard");
                return Ok(RRTNode {
                    id: None,
                    state: Vector6::new(p_rand[0], p_rand[1], 0.0, 0.0, 0.0, 0.0),
                    cost: 0.0,
                    d2land: 0.0,
                    trajectory: Vec::new(),
                    controls: Vec::new(),
                    time: 0.0,
                });
            } else {
                //println!("Sampled inside hazard");
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

    fn transfer_node_data(&mut self, z_recipient_id: &NodeId, z_new: &RRTNode) -> PyResult<()> {
        let z_recipient = self.bookkeeping_tree.get_mut(&z_recipient_id).unwrap();
        z_recipient.data_mut().state = z_new.state;
        z_recipient.data_mut().trajectory = z_new.trajectory.clone();
        z_recipient.data_mut().controls = z_new.controls.clone();
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
            RRTResult::new((vec![], vec![], vec![], std::f64::INFINITY)),
            |acc, x| {
                if x.cost < acc.cost {
                    x.clone()
                } else {
                    acc
                }
            },
        );
        println!(
            "Extracted best solution: {} | {}",
            opt_soln.cost,
            opt_soln.states.len()
        );
        self.optimize_solution(&mut opt_soln)?;
        Ok(opt_soln)
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
    #[allow(non_snake_case)]
    fn test_ancestry() -> PyResult<()> {
        let mut rrt = PQRRTStar::py_new(PQRRTParams {
            max_nodes: 1000,
            max_iter: 100000,
            iter_between_direct_goal_growth: 100,
            min_node_dist: 50.0,
            goal_radius: 100.0,
            step_size: 0.1,
            min_steering_time: 3.0,
            max_steering_time: 20.0,
            steering_acceptance_radius: 5.0,
            gamma: 200.0,
            max_nn_node_dist: 150.0,
            max_sample_adjustments: 10,
            lambda_sample_adjustment: 0.1,
            safe_distance: 25.0,
            max_ancestry_level: 1,
        });

        let xs_start = [0.0, 0.0, 0.0, 5.0, 0.0, 0.0];
        let xs_goal = [1000.0, 0.0, 0.0, 0.0, 0.0, 0.0];
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
            cost: 50.0,
            d2land: 20.0,
            state: Vector6::new(50.0, 0.0, 0.0, 5.0, 0.0, 0.0),
            trajectory: Vec::new(),
            controls: Vec::new(),
            time: 20.0,
        };
        let mut Z_near = rrt.nearest_neighbors(&z_new)?;
        let Z_an = rrt.ancestry(&Z_near.clone())?;
        println!("z_new: {:?}", z_new);
        println!("Z_near: {:?}", Z_near);
        println!("Z_an: {:?}", Z_an);
        Z_near.append(&mut Z_an.clone());
        let (z_new, z_parent) = rrt.choose_parent(&z_new, &Z_near[0].clone(), &Z_near)?;
        let _z_new_id = rrt.insert(&z_new, &z_parent)?;

        let z_new = RRTNode {
            id: None,
            cost: 150.0,
            d2land: 20.0,
            state: Vector6::new(100.0, 50.0, 0.0, 5.0, 0.0, 0.0),
            trajectory: Vec::new(),
            controls: Vec::new(),
            time: 20.0,
        };
        let mut Z_near = rrt.nearest_neighbors(&z_new)?;
        let Z_an = rrt.ancestry(&Z_near.clone())?;
        println!("z_new: {:?}", z_new);
        println!("Z_near: {:?}", Z_near);
        println!("Z_an: {:?}", Z_an);
        Z_near.append(&mut Z_an.clone());
        let (z_new, z_parent) = rrt.choose_parent(&z_new, &Z_near[0].clone(), &Z_near)?;
        let _z_new_id = rrt.insert(&z_new, &z_parent)?;

        let z_new = RRTNode {
            id: None,
            cost: 200.0,
            d2land: 20.0,
            state: Vector6::new(200.0, 0.0, 0.0, 5.0, 0.0, 0.0),
            trajectory: Vec::new(),
            controls: Vec::new(),
            time: 20.0,
        };
        let mut Z_near = rrt.nearest_neighbors(&z_new)?;
        let Z_an = rrt.ancestry(&Z_near.clone())?;
        println!("z_new: {:?}", z_new);
        println!("Z_near: {:?}", Z_near);
        println!("Z_an: {:?}", Z_an);
        Z_near.append(&mut Z_an.clone());
        let (z_new, z_parent) = rrt.choose_parent(&z_new, &Z_near[0].clone(), &Z_near)?;
        let _z_new_id = rrt.insert(&z_new, &z_parent)?;

        let z_new = RRTNode {
            id: None,
            cost: 270.0,
            d2land: 20.0,
            state: Vector6::new(250.0, 20.0, 0.0, 5.0, 0.0, 0.0),
            trajectory: Vec::new(),
            controls: Vec::new(),
            time: 20.0,
        };
        let mut Z_near = rrt.nearest_neighbors(&z_new)?;
        let Z_an = rrt.ancestry(&Z_near.clone())?;
        println!("z_new: {:?}", z_new);
        println!("Z_near: {:?}", Z_near);
        println!("Z_an: {:?}", Z_an);
        Z_near.append(&mut Z_an.clone());
        let (z_new, z_parent) = rrt.choose_parent(&z_new, &Z_near[0].clone(), &Z_near)?;
        let _z_new_id = rrt.insert(&z_new, &z_parent)?;

        let z_new = RRTNode {
            id: None,
            cost: 300.0,
            d2land: 20.0,
            state: Vector6::new(300.0, 0.0, 0.0, 5.0, 0.0, 0.0),
            trajectory: Vec::new(),
            controls: Vec::new(),
            time: 20.0,
        };
        let mut Z_near = rrt.nearest_neighbors(&z_new)?;
        let Z_an = rrt.ancestry(&Z_near.clone())?;
        println!("z_new: {:?}", z_new);
        println!("Z_near: {:?}", Z_near);
        println!("Z_an: {:?}", Z_an);
        Z_near.append(&mut Z_an.clone());
        let (z_new, z_parent) = rrt.choose_parent(&z_new, &Z_near[0].clone(), &Z_near)?;
        let _z_new_id = rrt.insert(&z_new, &z_parent)?;

        utils::draw_tree(
            "tree.png",
            &rrt.bookkeeping_tree,
            &Vector2::new(xs_start[0], xs_start[1]),
            &Vector2::new(xs_goal[0], xs_goal[1]),
            None,
            &rrt.enc,
        )
        .unwrap();
        println!("done");

        Ok(())
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_optimize_solution() -> PyResult<()> {
        let mut rrt = PQRRTStar::py_new(PQRRTParams {
            max_nodes: 2500,
            max_iter: 4000,
            iter_between_direct_goal_growth: 500,
            min_node_dist: 30.0,
            goal_radius: 300.0,
            step_size: 0.5,
            min_steering_time: 3.0,
            max_steering_time: 25.0,
            steering_acceptance_radius: 5.0,
            gamma: 1200.0,
            max_nn_node_dist: 125.0,
            max_sample_adjustments: 50,
            lambda_sample_adjustment: 0.3,
            safe_distance: 0.0,
            max_ancestry_level: 2,
        });
        let mut soln = RRTResult {
            states: vec![],
            inputs: vec![],
            times: vec![],
            cost: 0.0,
        };
        rrt.enc.load_hazards_from_json()?;
        soln.load_from_json()?;
        println!("soln length: {}", soln.states.len());
        rrt.optimize_solution(&mut soln)?;
        println!("optimized soln length: {}", soln.states.len());
        // soln = rrt.steer_through_solution(&soln)?;
        Python::with_gil(|py| -> PyResult<()> {
            let _soln_py = soln.to_object(py);
            Ok(())
        })?;
        Ok(())
    }

    #[test]
    fn test_grow_towards_goal() -> PyResult<()> {
        let mut rrt = PQRRTStar::py_new(PQRRTParams {
            max_nodes: 500,
            max_iter: 4000,
            iter_between_direct_goal_growth: 500,
            min_node_dist: 30.0,
            goal_radius: 300.0,
            step_size: 0.5,
            min_steering_time: 3.0,
            max_steering_time: 25.0,
            steering_acceptance_radius: 5.0,
            gamma: 1200.0,
            max_nn_node_dist: 125.0,
            max_sample_adjustments: 50,
            lambda_sample_adjustment: 0.3,
            safe_distance: 0.0,
            max_ancestry_level: 2,
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
            rrt.set_speed_reference(4.0)?;

            let do_list = Vec::<[f64; 6]>::new().into_py(py);
            let do_list = do_list.as_ref(py).downcast::<PyList>().unwrap();
            let result = rrt.grow_towards_goal(xs_start_py, 4.0, do_list, py)?;
            let pydict = result.as_ref(py).downcast::<PyDict>().unwrap();
            println!("rrtresult states: {:?}", pydict.get_item("states"));
            Ok(())
        })
    }
}
