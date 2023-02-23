use geo::{LineString, MultiLineString, MultiPolygon, Point, Polygon};
use pyo3::prelude::*;

#[pyclass]
#[derive(Clone, Debug)]
pub struct ENC {
    pub land: MultiPolygon<f64>,
    pub shore: MultiPolygon<f64>,
    pub seabed: MultiPolygon<f64>,
}

impl ENC {
    pub fn new() -> Self {
        let land = MultiPolygon(vec![]);
        let shore = MultiPolygon(vec![]);
        let seabed = MultiPolygon(vec![]);
        Self {
            land,
            shore,
            seabed,
        }
    }

    // pub fn from_pyobject(&self, py: Python, obj: &PyAny) -> Self {
    //     let land = obj
    //         .getattr("land")
    //         .unwrap()
    //         .extract::<MultiPolygon<f64>>()
    //         .unwrap();
    //     let shore = obj
    //         .getattr("shore")
    //         .unwrap()
    //         .extract::<MultiPolygon<f64>>()
    //         .unwrap();
    //     let seabed = obj
    //         .getattr("seabed")
    //         .unwrap()
    //         .extract::<MultiPolygon<f64>>()
    //         .unwrap();
    //     Self {
    //         land,
    //         shore,
    //         seabed,
    //     }
    // }
}
