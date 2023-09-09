//! # RRT* Rust
//! Implements a Rapidly-exploring Random Tree (RRT*) algorithm in Rust.
//!
//! ## Usage
//! Intended for use through Python (pyo3) bindings. Relies on getting ENC data from python shapely objects.
use pyo3::prelude::*;
pub mod enc_data;
pub mod informed_rrt_star;
mod model;
mod rrt_error;
mod steering;
mod utils;

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn informed_rrt_star_rust(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<crate::informed_rrt_star::InformedRRTStar>()?;
    Ok(())
}
