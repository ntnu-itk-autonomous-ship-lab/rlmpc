//! # RRT* Library
//! Implements two Rapidly-exploring Random Tree (RRT*) algorithms in Rust:
//! - PQ-RRT* (Potential field Quick RRT*)
//! - Informed RRT*
//! - RRT*
//! - RRT
//!
//! ## Usage
//! Intended for use through Python (pyo3) bindings. Relies on getting ENC data from python shapely objects.
use pyo3::prelude::*;
mod common;
mod model;
mod steering;
mod utils;

pub mod enc_data;
pub mod informed_rrt_star;
pub mod pq_rrt_star;
pub mod rrt;
pub mod rrt_star;

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn rrt_star_lib(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<crate::informed_rrt_star::InformedRRTStar>()?;
    m.add_class::<crate::pq_rrt_star::PQRRTStar>()?;
    m.add_class::<crate::rrt_star::RRTStar>()?;
    m.add_class::<crate::rrt::RRT>()?;
    Ok(())
}
