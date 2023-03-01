//! Custom error for the RRT library.
//!
//!

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use std::fmt;

#[derive(Debug, Clone)]
struct RRTError;

impl fmt::Display for RRTError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "invalid first item to double")
    }
}

trait ResultExt<T> {
    fn to_py_err(self) -> PyResult<T>;
}

impl<T> ResultExt<T> for Result<T, RRTError> {
    fn to_py_err(self) -> PyResult<T> {
        match self {
            Ok(x) => Ok(x),
            Err(e) => Err(PyErr::new::<PyTypeError, _>(format!("{:?}", e))),
        }
    }
}

type RRTResult<T> = std::result::Result<T, RRTError>;
