//! # ENCHazards
//! Implementation of an Electronic Navigational Chart (ENC) data structure in rust.
//! NOTE: Contains only the dangerous seabed, shore and land polygons for the vessel considered.
//! Seabed of sufficient depth is not included.
//!
//! ## Usage
//! Relies on transferring ENC data from python to rust using pyo3
use crate::utils;
use geo::{
    coord, point, BoundingRect, Contains, EuclideanDistance, HasDimensions, Intersects, LineString,
    MultiPolygon, Polygon, Rect,
};
use nalgebra::{Vector2, Vector6};
use pyo3::{exceptions::PyValueError, prelude::*};
use std::fs::File;
#[pyclass]
#[derive(Clone, Debug)]
pub struct ENCHazards {
    pub land: MultiPolygon<f64>,
    pub shore: MultiPolygon<f64>,
    pub seabed: MultiPolygon<f64>,
    pub bbox: Rect<f64>,
}

#[pymethods]
impl ENCHazards {
    #[new]
    pub fn py_new() -> Self {
        let land = MultiPolygon(vec![]);
        let shore = MultiPolygon(vec![]);
        let seabed = MultiPolygon(vec![]);
        let bbox = Rect::new(coord! {x: 0.0, y: 0.0}, coord! {x: 0.0, y: 0.0});
        Self {
            land,
            shore,
            seabed,
            bbox,
        }
    }

    pub fn is_empty(&self) -> bool {
        let empty = self.bbox.min() == self.bbox.max();
        empty
    }

    /// Transfer hazardous ENC data from python to rust. The ENC data is a list on the form:
    /// [land, shore, (dangerous)seabed]
    pub fn transfer_enc_data(&mut self, enc_data: Vec<&PyAny>) -> PyResult<()> {
        assert!(enc_data.len() == 3);
        for (i, hazard) in enc_data.iter().enumerate() {
            let hazard_type = hazard
                .getattr("geom_type")
                .unwrap()
                .extract::<&str>()
                .unwrap();

            assert_eq!(hazard_type, "MultiPolygon");
            let poly_out = self.transfer_multipolygon(hazard)?;
            match i {
                0 => self.land = poly_out,
                1 => self.shore = poly_out,
                2 => self.seabed = poly_out,
                _ => panic!("Unknown hazard type"),
            }
        }
        self.compute_bbox()?;
        self.save_hazards_to_json()?;
        Ok(())
    }

    pub fn save_hazards_to_json(&self) -> PyResult<()> {
        let rust_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        serde_json::to_writer_pretty(&File::create(rust_root.join("data/land.json"))?, &self.land)
            .unwrap();
        serde_json::to_writer_pretty(
            &File::create(rust_root.join("data/shore.json"))?,
            &self.shore,
        )
        .unwrap();
        serde_json::to_writer_pretty(
            &File::create(rust_root.join("data/seabed.json"))?,
            &self.seabed,
        )
        .unwrap();
        Ok(())
    }

    pub fn load_hazards_from_json(&mut self) -> PyResult<()> {
        let rust_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let land_file = File::open(rust_root.join("data/land.json")).unwrap();
        let shore_file = File::open(rust_root.join("data/shore.json")).unwrap();
        let seabed_file = File::open(rust_root.join("data/seabed.json")).unwrap();
        self.land = serde_json::from_reader(land_file).unwrap();
        self.shore = serde_json::from_reader(shore_file).unwrap();
        self.seabed = serde_json::from_reader(seabed_file).unwrap();
        self.compute_bbox()?;
        Ok(())
    }
}

impl ENCHazards {
    pub fn compute_bbox(&mut self) -> PyResult<Rect<f64>> {
        let mut land_bbox = Rect::new(coord! {x: 0.0, y: 0.0}, coord! {x: 0.0, y: 0.0});
        if !self.land.is_empty() {
            land_bbox = self.land.bounding_rect().unwrap();
        }

        let mut shore_bbox = Rect::new(coord! {x: 0.0, y: 0.0}, coord! {x: 0.0, y: 0.0});
        if !self.shore.is_empty() {
            shore_bbox = self.shore.bounding_rect().unwrap();
        }

        let mut seabed_bbox = Rect::new(coord! {x: 0.0, y: 0.0}, coord! {x: 0.0, y: 0.0});
        if !self.seabed.is_empty() {
            seabed_bbox = self.seabed.bounding_rect().unwrap();
        }

        if land_bbox.is_empty() && shore_bbox.is_empty() && seabed_bbox.is_empty() {
            return Err(PyValueError::new_err("ENC Hazard is empty"));
        }
        // println!("Land bbox: {:?}", land_bbox);
        // println!("Shore bbox: {:?}", shore_bbox);
        // println!("Seabed bbox: {:?}", seabed_bbox);
        let max_x = land_bbox
            .max()
            .x
            .max(shore_bbox.max().x)
            .max(seabed_bbox.max().x);
        let max_y = land_bbox
            .max()
            .y
            .max(shore_bbox.max().y)
            .max(seabed_bbox.max().y);
        let min_x = land_bbox
            .min()
            .x
            .min(shore_bbox.min().x)
            .min(seabed_bbox.min().x);
        let min_y = land_bbox
            .min()
            .y
            .min(shore_bbox.min().y)
            .min(seabed_bbox.min().y);
        self.bbox = Rect::new(coord! {x: min_x, y: min_y}, coord! {x: max_x, y: max_y});
        Ok(self.bbox)
    }

    /// Check if a point is inside the ENC Hazards
    pub fn inside_hazards(&self, p: &Vector2<f64>) -> bool {
        if self.is_empty() {
            return false;
        }
        let point = point![x: p[0], y: p[1]];
        self.land.contains(&point) || self.shore.contains(&point) || self.seabed.contains(&point)
    }

    pub fn inside_bbox(&self, p: &Vector2<f64>) -> bool {
        if self.is_empty() {
            return true;
        }
        let point = point![x: p[0], y: p[1]];
        self.bbox.contains(&point)
    }

    pub fn intersects_with_linestring(&self, linestring: &LineString<f64>) -> bool {
        let intersect_w_land = linestring.intersects(&self.land);
        let intersect_w_shore = linestring.intersects(&self.shore);
        let intersect_w_seabed = linestring.intersects(&self.seabed);
        // println!(
        //     "Linestring intersect with land: {}, shore: {}, seabed: {}",
        //     intersect_w_land, intersect_w_shore, intersect_w_seabed
        // );
        intersect_w_land || intersect_w_shore || intersect_w_seabed
    }

    pub fn intersects_with_segment(&self, p1: &Vector2<f64>, p2: &Vector2<f64>) -> bool {
        let line = LineString(vec![coord![x: p1[0], y: p1[1]], coord![x: p2[0], y: p2[1]]]);
        self.intersects_with_linestring(&line)
    }

    pub fn intersects_with_trajectory(&self, xs_array: &Vec<Vector6<f64>>) -> bool {
        let traj_linestring = LineString(
            xs_array
                .iter()
                .step_by(2)
                .map(|x| coord! {x: x[0], y: x[1]})
                .collect(),
        );
        let intersect = self.intersects_with_linestring(&traj_linestring);
        // utils::draw_enc_hazards_vs_linestring(
        //     "enc_vs_linestring.png",
        //     &self,
        //     &traj_linestring,
        //     &xs_array,
        // )
        // .unwrap();
        intersect
    }

    /// Calculate the distance from a point to the closest point on the ENC
    pub fn dist2point(&self, p: &Vector2<f64>) -> f64 {
        if self.is_empty() {
            println!("ENCHazards is empty");
            return -1.0;
        }
        let point = point![x: p[0], y: p[1]];
        let dist2land = point.euclidean_distance(&self.land);
        let dist2shore = point.euclidean_distance(&self.shore);
        let dist2seabed = point.euclidean_distance(&self.seabed);
        println!(
            "dist2land: {:?}, dist2shore: {:?}, dist2seabed: {:?}",
            dist2land, dist2shore, dist2seabed
        );
        dist2land.min(dist2shore).min(dist2seabed)
    }

    /// Care only about the polygon exterior ring, as this is the only relevant part
    /// for vessel trajectory planning. The polygons are assumed to have coordinates
    /// in the form (east, north)
    pub fn transfer_polygon(&self, py_poly: &PyAny) -> PyResult<Polygon<f64>> {
        let exterior = py_poly.getattr("exterior").unwrap().extract::<&PyAny>()?;
        let exterior_coords = exterior
            .getattr("coords")
            .unwrap()
            .extract::<Vec<&PyAny>>()?;

        let mut exterior_vec = vec![];
        for coord in exterior_coords {
            let coord_tuple = coord.extract::<(f64, f64)>().unwrap();
            exterior_vec.push(coord![x:coord_tuple.1, y:coord_tuple.0]);
        }
        Ok(Polygon::new(
            LineString(exterior_vec),
            vec![LineString(vec![])],
        ))
    }

    pub fn transfer_multipolygon(&self, py_multipoly: &PyAny) -> PyResult<MultiPolygon<f64>> {
        let py_geoms = py_multipoly
            .getattr("geoms")
            .unwrap()
            .extract::<Vec<&PyAny>>()?;
        let mut poly_vec = vec![];
        for py_poly in py_geoms {
            let polygon = self.transfer_polygon(&py_poly)?;
            poly_vec.push(polygon);
        }
        Ok(MultiPolygon(poly_vec))
    }

    pub fn set_land(&mut self, py_multipoly: &PyAny) -> PyResult<()> {
        self.land = self.transfer_multipolygon(&py_multipoly)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::PyList;

    #[test]
    fn test_transfer_polygon() {
        Python::with_gil(|py| {
            let enc = ENCHazards::py_new();

            let geometry = PyModule::import(py, "shapely.geometry").unwrap();
            let poly_class = geometry.getattr("Polygon").unwrap();
            let elements = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)];
            let pytuple_coords = PyList::new(py, elements);
            let args = (pytuple_coords,);

            let polygon = poly_class.call1(args).unwrap();
            let exterior = polygon.getattr("exterior").unwrap();
            let exterior_coords = exterior
                .getattr("coords")
                .unwrap()
                .extract::<Vec<&PyAny>>()
                .unwrap();
            println!("Polygon: {:?}", exterior_coords);

            let poly_out = enc.transfer_polygon(polygon).unwrap();
            println!("Polygon: {:?}", poly_out);
            assert_eq!(poly_out.exterior().0[0], coord!(x:0.0, y:0.0));
            assert_eq!(poly_out.exterior().0[1], coord!(x:0.0, y:1.0));
            assert_eq!(poly_out.exterior().0[2], coord!(x:1.0, y:1.0));
            assert_eq!(poly_out.exterior().0[3], coord!(x:1.0, y:0.0));
            assert_eq!(poly_out.exterior().0[4], coord!(x:0.0, y:0.0));
        })
    }

    #[test]
    fn test_transfer_multipolygon() {
        Python::with_gil(|py| {
            let enc = ENCHazards::py_new();

            let geometry = PyModule::import(py, "shapely.geometry").unwrap();
            let poly_class = geometry.getattr("Polygon").unwrap();
            let elements = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)];
            let pytuple_coords = PyList::new(py, elements);
            let args = (pytuple_coords,);

            let mut poly_vec = vec![];

            let polygon1 = poly_class.call1(args).unwrap();
            let exterior = polygon1.getattr("exterior").unwrap();
            let exterior_coords = exterior
                .getattr("coords")
                .unwrap()
                .extract::<Vec<&PyAny>>()
                .unwrap();
            println!("Polygon1: {:?}", exterior_coords);
            poly_vec.push(polygon1);

            let elements = vec![(2.0, 2.0), (4.0, 2.0), (4.0, 4.0), (2.0, 4.0), (2.0, 2.0)];
            let pytuple_coords = PyList::new(py, elements);

            let polygon2 = poly_class.call1((pytuple_coords,)).unwrap();
            let exterior = polygon2.getattr("exterior").unwrap();
            let exterior_coords = exterior
                .getattr("coords")
                .unwrap()
                .extract::<Vec<&PyAny>>()
                .unwrap();
            poly_vec.push(polygon2);
            println!("Polygon2: {:?}", exterior_coords);

            let multipoly_class = geometry.getattr("MultiPolygon").unwrap();
            let py_poly_list = PyList::new(py, poly_vec);
            let py_multipoly = multipoly_class.call1((py_poly_list,)).unwrap();

            let multipoly = enc.transfer_multipolygon(py_multipoly).unwrap();

            assert_eq!(multipoly.0[0].exterior().0[0], coord!(x:0.0, y:0.0));
            assert_eq!(multipoly.0[0].exterior().0[1], coord!(x:0.0, y:1.0));
            assert_eq!(multipoly.0[0].exterior().0[2], coord!(x:1.0, y:1.0));
            assert_eq!(multipoly.0[0].exterior().0[3], coord!(x:1.0, y:0.0));
            assert_eq!(multipoly.0[0].exterior().0[4], coord!(x:0.0, y:0.0));

            assert_eq!(multipoly.0[1].exterior().0[0], coord!(x:2.0, y:2.0));
            assert_eq!(multipoly.0[1].exterior().0[1], coord!(x:2.0, y:4.0));
            assert_eq!(multipoly.0[1].exterior().0[2], coord!(x:4.0, y:4.0));
            assert_eq!(multipoly.0[1].exterior().0[3], coord!(x:4.0, y:2.0));
            assert_eq!(multipoly.0[1].exterior().0[4], coord!(x:2.0, y:2.0));

            println!("Polygon: {:?}", multipoly);
        })
    }

    #[test]
    fn test_dist2point() {
        Python::with_gil(|py| {
            let mut enc = ENCHazards::py_new();

            let geometry = PyModule::import(py, "shapely.geometry").unwrap();
            let poly_class = geometry.getattr("Polygon").unwrap();
            let elements = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)];
            let pytuple_coords = PyList::new(py, elements);
            let args = (pytuple_coords,);
            let polygon = poly_class.call1(args).unwrap();

            let multipoly_class = geometry.getattr("MultiPolygon").unwrap();
            let py_poly_list = PyList::new(py, vec![polygon.clone()]);
            let py_multipoly = multipoly_class.call1((py_poly_list,)).unwrap();

            enc.set_land(py_multipoly).unwrap();

            let point = Vector2::new(0.5, 0.5);
            println!("Point: {:?}", point);
            println!("Polygon: {:?}", enc.land.0[0]);
            assert_eq!(enc.dist2point(&point), 0.0);
        })
    }

    #[test]
    fn test_inside_hazards() {
        Python::with_gil(|py| {
            let mut enc = ENCHazards::py_new();

            let geometry = PyModule::import(py, "shapely.geometry").unwrap();
            let poly_class = geometry.getattr("Polygon").unwrap();
            let elements = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)];
            let pytuple_coords = PyList::new(py, elements);
            let args = (pytuple_coords,);
            let polygon = poly_class.call1(args).unwrap();

            let multipoly_class = geometry.getattr("MultiPolygon").unwrap();
            let py_poly_list = PyList::new(py, vec![polygon.clone()]);
            let py_multipoly = multipoly_class.call1((py_poly_list,)).unwrap();

            enc.set_land(py_multipoly).unwrap();

            let point = Vector2::new(0.5, 0.5);

            println!("Point: {:?}", point);
            println!("Polygon: {:?}", enc.land.0[0]);
            assert_eq!(enc.inside_hazards(&point), true);
        })
    }

    #[test]
    fn test_intersections() {
        Python::with_gil(|py| {
            let mut enc = ENCHazards::py_new();

            let geometry = PyModule::import(py, "shapely.geometry").unwrap();
            let poly_class = geometry.getattr("Polygon").unwrap();
            let elements = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)];
            let pytuple_coords = PyList::new(py, elements);
            let args = (pytuple_coords,);
            let polygon = poly_class.call1(args).unwrap();

            let multipoly_class = geometry.getattr("MultiPolygon").unwrap();
            let py_poly_list = PyList::new(py, vec![polygon.clone()]);
            let py_multipoly = multipoly_class.call1((py_poly_list,)).unwrap();

            enc.set_land(py_multipoly).unwrap();

            let point1 = Vector2::new(-2.0, 2.0);
            let point2 = Vector2::new(2.0, -2.0);

            println!("Point1: {:?}", point1);
            println!("Point2: {:?}", point2);
            println!("Polygon: {:?}", enc.land.0[0]);
            assert_eq!(enc.intersects_with_segment(&point1, &point2), true);

            let linestring = LineString(vec![coord! {x: 0.0, y: 2.0}, coord! {x: 0.0, y: -2.0}]);
            println!("Linestring: {:?}", linestring);
            assert_eq!(enc.intersects_with_linestring(&linestring), true);
        })
    }

    #[test]
    fn test_compute_bbox() {
        Python::with_gil(|py| {
            let mut enc = ENCHazards::py_new();

            let geometry = PyModule::import(py, "shapely.geometry").unwrap();
            let poly_class = geometry.getattr("Polygon").unwrap();
            let elements = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)];
            let pytuple_coords = PyList::new(py, elements);
            let args = (pytuple_coords,);
            let polygon = poly_class.call1(args).unwrap();

            let multipoly_class = geometry.getattr("MultiPolygon").unwrap();
            let py_poly_list = PyList::new(py, vec![polygon.clone()]);
            let py_multipoly = multipoly_class.call1((py_poly_list,)).unwrap();

            enc.set_land(py_multipoly).unwrap();

            let bbox = enc.compute_bbox().unwrap();
            assert_eq!(bbox.min(), coord! {x: 0.0, y: 0.0});
            assert_eq!(bbox.max(), coord! {x: 1.0, y: 1.0});
        })
    }

    #[test]
    fn test_load_and_save_hazards_from_json() {
        let mut enc = ENCHazards::py_new();
        enc.load_hazards_from_json().unwrap();
        println!("Land: {:?}", enc.land);
        println!("Shore: {:?}", enc.shore);
        println!("Seabed: {:?}", enc.seabed);
        enc.save_hazards_to_json().unwrap();
    }
}
