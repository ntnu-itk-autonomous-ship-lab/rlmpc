//! # RRT* interface

use id_tree::*;
use nalgebra::{Vector2, Vector3, Vector6};
use pyo3::conversion::ToPyObject;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::FromPyObject;
use rstar::{PointDistance, RTreeObject, AABB};
use serde::{Deserialize, Serialize};
use std::fs::File;

#[derive(Debug, Clone, PartialEq, Default)]
pub struct RRTNode {
    pub id: Option<NodeId>,
    pub cost: f64,
    pub d2land: f64, // Not used as of now, but could be interesting to use in the RRT-cost.
    pub state: Vector6<f64>,
    pub trajectory: Vec<Vector6<f64>>, // Trajectory from parent to this node
    pub controls: Vec<Vector3<f64>>,   // Control inputs from parent to this node
    pub time: f64,
}

impl RRTNode {
    pub fn new(
        state: Vector6<f64>,
        trajectory: Vec<Vector6<f64>>,
        controls: Vec<Vector3<f64>>,
        cost: f64,
        d2land: f64,
        time: f64,
    ) -> Self {
        Self {
            id: None,
            state,
            cost,
            d2land,
            trajectory: trajectory,
            controls: controls,
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

#[derive(Debug, Clone, FromPyObject, Serialize, Deserialize)]
pub struct RRTResult {
    pub states: Vec<[f64; 6]>,
    pub inputs: Vec<[f64; 3]>,
    pub times: Vec<f64>,
    pub cost: f64,
}

impl RRTResult {
    pub fn new(solution: (Vec<[f64; 6]>, Vec<[f64; 3]>, Vec<f64>, f64)) -> Self {
        Self {
            states: solution.0,
            inputs: solution.1,
            times: solution.2,
            cost: solution.3,
        }
    }

    pub fn save_to_json(&self) -> PyResult<()> {
        let rust_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        serde_json::to_writer_pretty(
            &File::create(rust_root.join("data/rrt_result.json"))?,
            &self,
        )
        .unwrap();
        Ok(())
    }

    pub fn load_from_json(&mut self) -> PyResult<()> {
        let rust_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let solution_file = File::open(rust_root.join("data/rrt_result.json")).unwrap();
        let result: RRTResult = serde_json::from_reader(solution_file).unwrap();
        self.states = result.states;
        self.inputs = result.inputs;
        self.times = result.times;
        self.cost = result.cost;
        Ok(())
    }
}

impl ToPyObject for RRTResult {
    fn to_object(&self, py: Python) -> PyObject {
        let states = PyList::empty(py);
        let inputs = PyList::empty(py);
        let times = PyList::empty(py);
        let n_wps = self.states.len();
        for i in 0..n_wps {
            // Only the starting root state should have a time of 0.0
            if i > 0 && self.times[i] < 0.0001 {
                continue;
            }
            states.append(self.states[i].to_object(py)).unwrap();
            times.append(self.times[i].to_object(py)).unwrap();
            if i < n_wps - 1 {
                inputs.append(self.inputs[i].to_object(py)).unwrap();
            }
        }
        let cost = self.cost.to_object(py);
        let result_dict = PyDict::new(py);
        result_dict
            .set_item("states", states)
            .expect("Solution states should be set");
        result_dict
            .set_item("inputs", inputs)
            .expect("Solution inputs should be set");
        result_dict
            .set_item("times", times)
            .expect("Solution times should be set");
        result_dict
            .set_item("cost", cost)
            .expect("Solution cost should be set");
        result_dict.to_object(py)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstar::RTree;

    #[test]
    fn test_rrtnode() {
        let node = RRTNode {
            id: None,
            cost: 0.0,
            d2land: 0.0,
            state: Vector6::zeros(),
            trajectory: Vec::new(),
            controls: Vec::new(),
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
            trajectory: Vec::new(),
            controls: Vec::new(),

            time: 0.0,
        });

        tree.insert(RRTNode {
            id: None,
            cost: 2.0,
            d2land: 40.0,
            state: Vector6::new(50.0, 50.0, 0.0, 0.0, 0.0, 0.0),
            trajectory: Vec::new(),
            controls: Vec::new(),
            time: 20.0,
        });

        let nearest = tree.nearest_neighbor(&[1.0, 0.0]).unwrap();
        assert_eq!(nearest.state, Vector6::zeros());
    }
}
