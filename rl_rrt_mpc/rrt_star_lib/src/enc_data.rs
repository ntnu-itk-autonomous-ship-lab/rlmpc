//! # ENCData
//! Implementation of an Electronic Navigational Chart (ENC) data structure in rust.
//! NOTE: Contains only the dangerous seabed, shore and land polygons for the vessel considered.
//! Seabed of sufficient depth is not included.
//!
//! ## Usage
//! Relies on transferring ENC data from python to rust using pyo3
use geo::{
    coord, point, BoundingRect, Contains, EuclideanDistance, HasDimensions, Intersects, LineString,
    MultiPolygon, Polygon, Rect,
};
use nalgebra::{Vector2, Vector6};
use pyo3::prelude::*;
use pyo3::types::PyList;
use std::fs::File;
#[pyclass]
#[derive(Clone, Debug)]
pub struct ENCData {
    pub hazards: MultiPolygon<f64>,
    pub safe_sea_triangulation: Vec<Polygon<f64>>,
    pub bbox: Rect<f64>,
}

#[pymethods]
impl ENCData {
    #[new]
    pub fn py_new() -> Self {
        let hazards = MultiPolygon(vec![]);
        let safe_sea_triangulation = vec![];
        let bbox = Rect::new(coord! {x: 0.0, y: 0.0}, coord! {x: 0.0, y: 0.0});
        Self {
            hazards,
            safe_sea_triangulation,
            bbox,
        }
    }

    pub fn is_empty(&self) -> bool {
        let empty = self.bbox.min() == self.bbox.max();
        empty
    }

    /// Transfer hazardous ENC data from python to rust. The ENC data is a list on the form:
    /// [land, shore, (dangerous)seabed]
    pub fn transfer_enc_hazards(&mut self, hazards: &PyAny) -> PyResult<()> {
        let hazard_type = hazards
            .getattr("geom_type")
            .unwrap()
            .extract::<&str>()
            .unwrap();

        assert_eq!(hazard_type, "MultiPolygon");
        self.hazards = self.transfer_multipolygon(hazards)?;
        self.compute_bbox()?;
        self.save_hazards_to_json()?;
        Ok(())
    }

    pub fn transfer_safe_sea_triangulation(
        &mut self,
        py_safe_sea_triangulation: &PyList,
    ) -> PyResult<()> {
        let mut poly_vec = vec![];
        for py_poly in py_safe_sea_triangulation {
            let polygon = self.transfer_polygon(py_poly)?;
            poly_vec.push(polygon);
        }
        self.safe_sea_triangulation = poly_vec;
        self.save_triangulation_to_json()?;
        Ok(())
    }

    pub fn save_hazards_to_json(&self) -> PyResult<()> {
        let rust_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        serde_json::to_writer_pretty(
            &File::create(rust_root.join("data/hazards.json"))?,
            &self.hazards,
        )
        .unwrap();
        Ok(())
    }

    pub fn save_triangulation_to_json(&self) -> PyResult<()> {
        let rust_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        serde_json::to_writer_pretty(
            &File::create(rust_root.join("data/safe_sea_triangulation.json"))?,
            &self.safe_sea_triangulation,
        )
        .unwrap();
        Ok(())
    }

    pub fn load_hazards_from_json(&mut self) -> PyResult<()> {
        let rust_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let hazards_file = File::open(rust_root.join("data/hazards.json")).unwrap();
        self.hazards = serde_json::from_reader(hazards_file).unwrap();
        self.compute_bbox()?;
        Ok(())
    }

    pub fn load_safe_sea_triangulation_from_json(&mut self) -> PyResult<()> {
        let rust_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let safe_sea_triangulation_file =
            File::open(rust_root.join("data/safe_sea_triangulation.json")).unwrap();
        self.safe_sea_triangulation = serde_json::from_reader(safe_sea_triangulation_file).unwrap();
        Ok(())
    }
}

impl ENCData {
    pub fn compute_bbox(&mut self) -> PyResult<Rect<f64>> {
        let mut hazards_bbox = Rect::new(coord! {x: 0.0, y: 0.0}, coord! {x: 0.0, y: 0.0});
        if !self.hazards.is_empty() {
            hazards_bbox = self.hazards.bounding_rect().unwrap();
        }
        let min_x = hazards_bbox.min().x;
        let min_y = hazards_bbox.min().y;
        let max_x = hazards_bbox.max().x;
        let max_y = hazards_bbox.max().y;
        self.bbox = Rect::new(coord! {x: min_x, y: min_y}, coord! {x: max_x, y: max_y});
        Ok(self.bbox)
    }

    /// Check if a point is inside the ENC Hazards
    pub fn inside_hazards(&self, p: &Vector2<f64>) -> bool {
        if self.is_empty() {
            return false;
        }
        let point = point![x: p[0], y: p[1]];
        self.hazards.contains(&point)
    }

    pub fn inside_bbox(&self, p: &Vector2<f64>) -> bool {
        if self.is_empty() {
            return true;
        }
        let point = point![x: p[0], y: p[1]];
        self.bbox.contains(&point)
    }

    pub fn intersects_with_linestring(&self, linestring: &LineString<f64>) -> bool {
        linestring.intersects(&self.hazards)
    }

    pub fn intersects_with_segment(&self, p1: &Vector2<f64>, p2: &Vector2<f64>) -> bool {
        let line = LineString(vec![coord![x: p1[0], y: p1[1]], coord![x: p2[0], y: p2[1]]]);
        self.intersects_with_linestring(&line)
    }

    pub fn intersects_with_trajectory(&self, xs_array: &Vec<Vector6<f64>>) -> bool {
        let traj_linestring =
            LineString(xs_array.iter().map(|x| coord! {x: x[0], y: x[1]}).collect());
        let intersect = self.intersects_with_linestring(&traj_linestring);
        intersect
    }

    pub fn array_inside_bbox(&self, xs_array: &Vec<Vector6<f64>>) -> bool {
        let x_min = self.bbox.min().x;
        let x_max = self.bbox.max().x;
        let y_min = self.bbox.min().y;
        let y_max = self.bbox.max().y;
        for xs in xs_array.iter() {
            let x = xs[0];
            let y = xs[1];
            if x < x_min || x > x_max || y < y_min || y > y_max {
                return false;
            }
        }
        true
    }

    /// Calculate the distance from a point to the closest point on the ENC
    pub fn dist2point(&self, p: &Vector2<f64>) -> f64 {
        if self.is_empty() {
            println!("ENCData is empty");
            return -1.0;
        }
        let point = point![x: p[0], y: p[1]];
        let dist2hazards = point.euclidean_distance(&self.hazards);
        // println!("dist2hazards: {:?}", dist2hazards);
        dist2hazards
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

    pub fn set_hazards(&mut self, py_multipoly: &PyAny) -> PyResult<()> {
        self.hazards = self.transfer_multipolygon(&py_multipoly)?;
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
            let enc = ENCData::py_new();

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
            let enc = ENCData::py_new();

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
            let mut enc = ENCData::py_new();

            let geometry = PyModule::import(py, "shapely.geometry").unwrap();
            let poly_class = geometry.getattr("Polygon").unwrap();
            let elements = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)];
            let pytuple_coords = PyList::new(py, elements);
            let args = (pytuple_coords,);
            let polygon = poly_class.call1(args).unwrap();

            let multipoly_class = geometry.getattr("MultiPolygon").unwrap();
            let py_poly_list = PyList::new(py, vec![polygon.clone()]);
            let py_multipoly = multipoly_class.call1((py_poly_list,)).unwrap();

            enc.set_hazards(py_multipoly).unwrap();

            let point = Vector2::new(0.5, 0.5);
            println!("Point: {:?}", point);
            println!("Polygon: {:?}", enc.hazards.0[0]);
            assert_eq!(enc.dist2point(&point), 0.0);
        })
    }

    #[test]
    fn test_inside_hazards() {
        Python::with_gil(|py| {
            let mut enc = ENCData::py_new();

            let geometry = PyModule::import(py, "shapely.geometry").unwrap();
            let poly_class = geometry.getattr("Polygon").unwrap();
            let elements = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)];
            let pytuple_coords = PyList::new(py, elements);
            let args = (pytuple_coords,);
            let polygon = poly_class.call1(args).unwrap();

            let multipoly_class = geometry.getattr("MultiPolygon").unwrap();
            let py_poly_list = PyList::new(py, vec![polygon.clone()]);
            let py_multipoly = multipoly_class.call1((py_poly_list,)).unwrap();

            enc.set_hazards(py_multipoly).unwrap();

            let point = Vector2::new(0.5, 0.5);

            println!("Point: {:?}", point);
            println!("Polygon: {:?}", enc.hazards.0[0]);
            assert_eq!(enc.inside_hazards(&point), true);
        })
    }

    #[test]
    fn test_intersections() {
        Python::with_gil(|py| {
            let mut enc = ENCData::py_new();

            let geometry = PyModule::import(py, "shapely.geometry").unwrap();
            let poly_class = geometry.getattr("Polygon").unwrap();
            let elements = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)];
            let pytuple_coords = PyList::new(py, elements);
            let args = (pytuple_coords,);
            let polygon = poly_class.call1(args).unwrap();

            let multipoly_class = geometry.getattr("MultiPolygon").unwrap();
            let py_poly_list = PyList::new(py, vec![polygon.clone()]);
            let py_multipoly = multipoly_class.call1((py_poly_list,)).unwrap();

            enc.set_hazards(py_multipoly).unwrap();

            let point1 = Vector2::new(-2.0, 2.0);
            let point2 = Vector2::new(2.0, -2.0);

            println!("Point1: {:?}", point1);
            println!("Point2: {:?}", point2);
            println!("Polygon: {:?}", enc.hazards.0[0]);
            assert_eq!(enc.intersects_with_segment(&point1, &point2), true);

            let linestring = LineString(vec![coord! {x: 0.0, y: 2.0}, coord! {x: 0.0, y: -2.0}]);
            println!("Linestring: {:?}", linestring);
            assert_eq!(enc.intersects_with_linestring(&linestring), true);
        })
    }

    #[test]
    fn test_compute_bbox() {
        Python::with_gil(|py| {
            let mut enc = ENCData::py_new();

            let geometry = PyModule::import(py, "shapely.geometry").unwrap();
            let poly_class = geometry.getattr("Polygon").unwrap();
            let elements = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)];
            let pytuple_coords = PyList::new(py, elements);
            let args = (pytuple_coords,);
            let polygon = poly_class.call1(args).unwrap();

            let multipoly_class = geometry.getattr("MultiPolygon").unwrap();
            let py_poly_list = PyList::new(py, vec![polygon.clone()]);
            let py_multipoly = multipoly_class.call1((py_poly_list,)).unwrap();

            enc.set_hazards(py_multipoly).unwrap();

            let bbox = enc.compute_bbox().unwrap();
            assert_eq!(bbox.min(), coord! {x: 0.0, y: 0.0});
            assert_eq!(bbox.max(), coord! {x: 1.0, y: 1.0});
        })
    }

    #[test]
    fn test_load_and_save_hazards_from_json() {
        let mut enc = ENCData::py_new();
        enc.load_hazards_from_json().unwrap();
        println!("Hazards: {:?}", enc.hazards);
        enc.save_hazards_to_json().unwrap();
    }
}
