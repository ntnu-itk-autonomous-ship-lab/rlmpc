//! # utils
//! Contains utility functions for the RRT* algorithm
//!
use crate::enc_hazards::ENCHazards;
use crate::informed_rrt_star::RRTNode;
use geo::{coord, LineString, MultiPolygon, Rect};
use id_tree::*;
use nalgebra::{
    ClosedAdd, ClosedMul, Matrix2, Matrix3, SMatrix, Scalar, Vector2, Vector3, Vector6,
};
use num::traits::{One, Zero};
use plotters::coord::types::RangedCoordf32;
use plotters::coord::Shift;
use plotters::{drawing, prelude::*};
use pyo3::prelude::*;
use rand::Rng;
use rand_chacha::ChaChaRng;
use std::f64::consts;
use std::iter;

pub fn bbox_from_corner_points(p1: &Vector2<f64>, p2: &Vector2<f64>, buffer: f64) -> Rect {
    let p_min = Vector2::new(p1[0].min(p2[0]) - buffer, p1[1].min(p2[1]) - buffer);
    let p_max = Vector2::new(p1[0].max(p2[0]) + buffer, p1[1].max(p2[1]) + buffer);
    Rect::new(
        coord! { x: p_min[0], y: p_min[1] },
        coord! { x: p_max[0], y: p_max[1]},
    )
}

#[allow(non_snake_case)]
pub fn informed_sample(
    p_start: &Vector2<f64>,
    p_goal: &Vector2<f64>,
    c_max: f64,
    rng: &mut ChaChaRng,
) -> Vector2<f64> {
    assert!(c_max < f64::INFINITY && c_max > 0.0);
    let c_min = (p_start - p_goal).norm();
    let p_centre = (p_start + p_goal) / 2.0;
    let r_1 = c_max / 2.0;
    let r_2 = (c_max.powi(2) - c_min.powi(2)).abs().sqrt() / 2.0;
    let L = Matrix2::from_partial_diagonal(&[r_1, r_2]);
    let x_ball: Vector2<f64> = sample_from_unit_ball(rng);
    let p_rand: Vector2<f64> = transform_standard_sample(x_ball, L, p_centre);
    p_rand
}

pub fn sample_from_unit_ball(rng: &mut ChaChaRng) -> Vector2<f64> {
    let mut p = Vector2::zeros();
    loop {
        p[0] = rng.gen_range(-1.0..1.0);
        p[1] = rng.gen_range(-1.0..1.0);
        if p.norm() <= 1.0 {
            return p;
        }
    }
}

pub fn sample_from_bbox(bbox: &Rect, rng: &mut ChaChaRng) -> Vector2<f64> {
    let x = rng.gen_range(bbox.min().x..bbox.max().x);
    let y = rng.gen_range(bbox.min().y..bbox.max().y);
    Vector2::new(x, y)
}

pub fn transform_standard_sample<T, const S: usize>(
    x_rand: SMatrix<T, S, 1>,
    mtrx: SMatrix<T, S, S>,
    offset: SMatrix<T, S, 1>,
) -> SMatrix<T, S, 1>
where
    T: Scalar + Zero + One + ClosedAdd + ClosedMul,
{
    mtrx * x_rand + offset
}

#[allow(non_snake_case)]
pub fn Cmtrx(Mmtrx: Matrix3<f64>, nu: Vector3<f64>) -> Matrix3<f64> {
    let mut Cmtrx = Matrix3::zeros();

    let c13 = -(Mmtrx[(1, 1)] * nu[1] + Mmtrx[(1, 2)] * nu[2]);
    let c23 = Mmtrx[(0, 0)] * nu[0];
    Cmtrx[(0, 2)] = c13;
    Cmtrx[(1, 2)] = c23;
    Cmtrx[(2, 0)] = -c13;
    Cmtrx[(2, 1)] = -c23;
    Cmtrx
}

#[allow(non_snake_case)]
pub fn Dmtrx(
    D_l: Matrix3<f64>,
    D_q: Matrix3<f64>,
    D_c: Matrix3<f64>,
    nu: Vector3<f64>,
) -> Matrix3<f64> {
    let D_q_res = D_q * Matrix3::from_partial_diagonal(&[nu[0].abs(), nu[1].abs(), nu[2].abs()]);
    let nu_squared = nu.component_mul(&nu);
    let D_c_res =
        D_c * Matrix3::from_partial_diagonal(&[nu_squared[0], nu_squared[1], nu_squared[2]]);
    // println!("D_l: {:?}", D_l);
    // println!("D_q: {:?}", D_q);
    // println!("D_q_res: {:?}", D_q_res);
    // println!("D_c: {:?}", D_c);
    // println!("D_c_res: {:?}", D_c_res);
    // println!("nu_squared: {:?}", nu_squared);
    D_l + D_q_res + D_c_res
}

#[allow(non_snake_case)]
pub fn Rmtrx(psi: f64) -> Matrix3<f64> {
    let mut Rmtrx = Matrix3::zeros();
    Rmtrx[(0, 0)] = psi.cos();
    Rmtrx[(0, 1)] = -psi.sin();
    Rmtrx[(1, 0)] = psi.sin();
    Rmtrx[(1, 1)] = psi.cos();
    Rmtrx[(2, 2)] = 1.0;
    Rmtrx
}

pub fn wrap_min_max(x: f64, x_min: f64, x_max: f64) -> f64 {
    x_min + (x - x_min) % (x_max - x_min)
}

pub fn wrap_angle_to_pmpi(x: f64) -> f64 {
    wrap_min_max(x, -consts::PI, consts::PI)
}

// pub fn wrap_angle_to_02pi(x: f64) -> f64 {
//     wrap_min_max(x, 0.0, 2.0 * consts::PI)
// }

pub fn wrap_angle_diff_to_pmpi(x: f64, y: f64) -> f64 {
    let diff = x - y;
    wrap_angle_to_pmpi(diff)
}

pub fn unwrap_angle(x_prev: f64, x: f64) -> f64 {
    x_prev + wrap_angle_diff_to_pmpi(x, x_prev)
}

pub fn rad2deg(x: f64) -> f64 {
    x * 180.0 / consts::PI
}

pub fn deg2rad(x: f64) -> f64 {
    x * consts::PI / 180.0
}

pub fn saturate(x: f64, x_min: f64, x_max: f64) -> f64 {
    x.min(x_max).max(x_min)
}

pub fn compute_path_length(xs_array: &Vec<Vector6<f64>>) -> f64 {
    xs_array
        .iter()
        .zip(xs_array.iter().skip(1))
        .map(|(x1, x2)| {
            //println!("x1: {:?} | xs: {:?}", x1, x2);
            (Vector2::new(x1[0], x1[1]) - Vector2::new(x2[0], x2[1])).norm()
        })
        .sum()
}

pub fn map_err_to_pyerr<E>(e: E) -> PyErr
where
    E: std::fmt::Display,
{
    PyErr::new::<pyo3::exceptions::PyException, _>(e.to_string())
}

pub fn draw_multipolygon(
    drawing_area: &DrawingArea<SVGBackend, Shift>,
    chart: &mut ChartContext<SVGBackend, Cartesian2d<RangedCoordf32, RangedCoordf32>>,
    multipolygon: &MultiPolygon<f64>,
    color: &RGBColor,
) -> Result<(), Box<dyn std::error::Error>> {
    for polygon in multipolygon.0.iter() {
        let poly_points: Vec<(f32, f32)> = polygon
            .exterior()
            .0
            .iter()
            .map(|p| (p.y as f32, p.x as f32))
            .collect();
        println!("poly_points: {:?}", poly_points);
        chart.draw_series(LineSeries::new(poly_points, color))?;
    }
    drawing_area.present()?;
    Ok(())
}

pub fn draw_linestring(
    drawing_area: &DrawingArea<SVGBackend, Shift>,
    chart: &mut ChartContext<SVGBackend, Cartesian2d<RangedCoordf32, RangedCoordf32>>,
    linestring: &LineString<f64>,
    color: &RGBColor,
) -> Result<(), Box<dyn std::error::Error>> {
    let line_points: Vec<(f32, f32)> = linestring
        .0
        .iter()
        .map(|p| (p.y as f32, p.x as f32))
        .collect();
    println!("line_points: {:?}", line_points);
    chart.draw_series(LineSeries::new(line_points, color))?;
    drawing_area.present()?;
    Ok(())
}

pub fn draw_enc_hazards_vs_linestring(
    filename: &str,
    enc_hazards: &ENCHazards,
    linestring: &LineString<f64>,
) -> Result<(), Box<dyn std::error::Error>> {
    let drawing_area = SVGBackend::new(filename, (2048, 1440)).into_drawing_area();
    drawing_area.fill(&WHITE)?;
    let bbox = enc_hazards.bbox;
    let buffer = 500.0;
    let min_x_ = linestring
        .0
        .iter()
        .fold(f32::INFINITY, |acc, p| acc.min(p.x as f32))
        - buffer;
    let min_y_ = linestring
        .0
        .iter()
        .fold(f32::INFINITY, |acc, p| acc.min(p.y as f32))
        - buffer;
    let max_x_ = linestring
        .0
        .iter()
        .fold(f32::NEG_INFINITY, |acc, p| acc.max(p.x as f32))
        + buffer;
    let max_y_ = linestring
        .0
        .iter()
        .fold(f32::NEG_INFINITY, |acc, p| acc.max(p.y as f32))
        + buffer;
    let mut chart = ChartBuilder::on(&drawing_area)
        .caption("ENC Hazards vs linestring", ("sans-serif", 40).into_font())
        .x_label_area_size(75)
        .y_label_area_size(75)
        .build_cartesian_2d(
            min_y_..max_y_,
            min_x_..max_x_,
            // bbox.min().y as f32..bbox.max().y as f32,
            // bbox.min().x as f32..bbox.max().x as f32,
        )?;

    chart
        .configure_mesh()
        .x_labels(10)
        .y_labels(10)
        .x_label_formatter(&|x| format!("{:.1}", x))
        .y_label_formatter(&|x| format!("{:.1}", x))
        .draw()?;

    draw_multipolygon(&drawing_area, &mut chart, &enc_hazards.land, &RED)?;
    draw_multipolygon(&drawing_area, &mut chart, &enc_hazards.shore, &YELLOW)?;
    draw_multipolygon(&drawing_area, &mut chart, &enc_hazards.seabed, &BLUE)?;
    draw_linestring(&drawing_area, &mut chart, &linestring, &MAGENTA)?;
    Ok(())
}

pub fn draw_steering_results(
    xs_start: &Vector6<f64>,
    xs_goal: &Vector6<f64>,
    refs_array: &Vec<(f64, f64)>,
    xs_array: &Vec<Vector6<f64>>,
    acceptance_radius: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    draw_north_east_chart(
        "steer.png",
        &xs_array,
        &vec![xs_start.clone(), xs_goal.clone()],
    )?;

    // Draw psi vs psi_d
    let mut psi_array: Vec<f64> = xs_array.iter().map(|xs| rad2deg(xs[2])).collect();
    psi_array.remove(0);
    let psi_d_array: Vec<f64> = refs_array.iter().map(|refs| rad2deg(refs.1)).collect();

    let psi_error_array: Vec<f64> = psi_array
        .iter()
        .zip(psi_d_array.iter())
        .map(|(psi, psi_d)| wrap_angle_diff_to_pmpi(*psi_d, *psi))
        .collect();
    let ref_error_array: Vec<f64> = psi_error_array.iter().map(|_| 0.0).collect();

    draw_variable_vs_reference("psi_comp.png", "psi error", &psi_array, &psi_d_array)?;

    // Draw u vs u_d
    let mut u_array: Vec<f64> = xs_array
        .iter()
        .map(|xs| (xs[3] * xs[3] + xs[4] * xs[4]).sqrt())
        .collect();
    u_array.remove(0);
    let u_d_array: Vec<f64> = refs_array.iter().map(|refs| refs.0).collect();
    let u_error_array: Vec<f64> = u_array
        .iter()
        .zip(u_d_array.iter())
        .map(|(u, u_d)| u_d - u)
        .collect();
    draw_variable_vs_reference("u_comp.png", "u error", &u_array, &u_d_array)?;
    Ok(())
}

pub fn draw_tree(
    filename: &str,
    tree: &Tree<RRTNode>,
    p_start: &Vector2<f64>,
    p_goal: &Vector2<f64>,
    xs_soln_array: Option<&Vec<[f64; 6]>>,
    enc_hazards: &ENCHazards,
) -> Result<(), Box<dyn std::error::Error>> {
    let drawing_area = BitMapBackend::new(filename, (640, 480)).into_drawing_area();
    drawing_area.fill(&WHITE)?;
    let mut bbox = enc_hazards.bbox;
    if enc_hazards.is_empty() {
        bbox = bbox_from_corner_points(p_start, p_goal, 100.0);
    }
    println!("Map bbox: {:?}", bbox);
    let mut chart = ChartBuilder::on(&drawing_area)
        .caption("Tree", ("sans-serif", 40).into_font())
        .x_label_area_size(50)
        .y_label_area_size(50)
        .build_cartesian_2d(
            bbox.min().y as f32..bbox.max().y as f32,
            bbox.min().x as f32..bbox.max().x as f32,
        )?;

    chart
        .configure_mesh()
        .x_labels(10)
        .y_labels(10)
        .x_label_formatter(&|x| format!("{:.1}", x))
        .y_label_formatter(&|x| format!("{:.1}", x))
        .draw()?;

    let root_node_id = tree.root_node_id().unwrap();
    draw_tree_lines(&drawing_area, &mut chart, tree, &root_node_id)?;

    match xs_soln_array {
        Some(xs_soln_array) => {
            let p_soln_array = xs_soln_array
                .iter()
                .map(|xs| (xs[1] as f32, xs[0] as f32))
                .collect::<Vec<(f32, f32)>>();
            chart.draw_series(LineSeries::new(p_soln_array, &BLUE))?;
        }
        None => {}
    }
    drawing_area.present()?;
    Ok(())
}

pub fn draw_tree_lines(
    drawing_area: &DrawingArea<BitMapBackend, Shift>,
    chart: &mut ChartContext<BitMapBackend, Cartesian2d<RangedCoordf32, RangedCoordf32>>,
    tree: &Tree<RRTNode>,
    node_id: &NodeId,
) -> Result<(), Box<dyn std::error::Error>> {
    let node = tree.get(node_id).unwrap();
    let mut children_ids = tree.children_ids(node_id).unwrap();
    loop {
        let child_id = match children_ids.next() {
            Some(id) => id,
            None => break,
        };

        let child_node = tree.get(child_id).unwrap();

        let points = vec![
            (node.data().state[1] as f32, node.data().state[0] as f32),
            (
                child_node.data().state[1] as f32,
                child_node.data().state[0] as f32,
            ),
        ];

        chart.draw_series(LineSeries::new(points.clone(), &BLACK))?;
        chart.draw_series(PointSeries::of_element(points, 2, &RED, &|c, s, st| {
            EmptyElement::at(c) + Circle::new((0, 0), s, st.filled())
            // + Text::new(
            //     format!("({:.1}, {:.1})", c.0, c.1),
            //     (0, 15),
            //     ("sans-serif", 12),
            // )
        }))?;

        drawing_area.present()?;
        draw_tree_lines(drawing_area, chart, tree, child_id)?;
    }
    Ok(())
}

pub fn draw_north_east_chart(
    filename: &str,
    xs_array: &Vec<Vector6<f64>>,
    waypoints: &Vec<Vector6<f64>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let buffer = 100.0;
    let min_wp_y = waypoints
        .iter()
        .map(|x| x[1])
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap()
        - buffer;
    let min_wp_x = waypoints
        .iter()
        .map(|x| x[0])
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap()
        - buffer;
    let max_wp_y = waypoints
        .iter()
        .map(|x| x[1])
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap()
        + buffer;
    let max_wp_x = waypoints
        .iter()
        .map(|x| x[0])
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap()
        + buffer;

    let root = BitMapBackend::new(filename, (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;
    let root = root.margin(12, 12, 12, 12);
    let mut chart = ChartBuilder::on(&root)
        .caption("NE Plot", ("sans-serif", 40).into_font())
        .x_label_area_size(25)
        .y_label_area_size(25)
        .build_cartesian_2d(
            min_wp_y as f32..max_wp_y as f32,
            min_wp_x as f32..max_wp_x as f32,
        )?;

    chart
        .configure_mesh()
        .x_labels(5)
        .y_labels(5)
        .y_label_formatter(&|x| format!("{:.1}", x))
        .draw()?;

    let ne_array_points = xs_array
        .iter()
        .map(|x| (x[1] as f32, x[0] as f32))
        .collect::<Vec<(f32, f32)>>();

    chart.draw_series(LineSeries::new(ne_array_points, &BLACK))?;

    let ne_waypoints = waypoints
        .iter()
        .map(|x| (x[1] as f32, x[0] as f32))
        .collect::<Vec<(f32, f32)>>();

    chart.draw_series(LineSeries::new(ne_waypoints.clone(), &BLUE))?;
    chart.draw_series(PointSeries::of_element(
        ne_waypoints,
        5,
        &RED,
        &|c, s, st| {
            return EmptyElement::at(c)    // We want to construct a composed element on-the-fly
            + Circle::new((0,0),s,st.filled()) // At this point, the new pixel coordinate is established
            + Text::new(format!("{:?}", c), (10, 0), ("sans-serif", 10).into_font());
        },
    ))?;

    root.present()?;
    Ok(())
}

pub fn draw_variable_vs_reference(
    filename: &str,
    chart_name: &str,
    variable: &Vec<f64>,
    reference: &Vec<f64>,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(filename, (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;
    let root = root.margin(12, 12, 12, 12);

    let samples: Vec<f32> = (0..variable.len())
        .collect::<Vec<usize>>()
        .iter()
        .map(|x| *x as f32)
        .collect();
    let var_points = variable.iter().map(|x| *x as f32).collect::<Vec<f32>>();
    let min_var: f32 = var_points.iter().fold(f32::INFINITY, |a, b| a.min(*b));
    let max_var: f32 = var_points.iter().fold(-f32::INFINITY, |a, b| a.max(*b));
    let ref_points = reference.iter().map(|x| *x as f32).collect::<Vec<f32>>();
    let min_ref: f32 = ref_points.iter().fold(f32::INFINITY, |a, b| a.min(*b));
    let max_ref: f32 = ref_points.iter().fold(-f32::INFINITY, |a, b| a.max(*b));

    let min_y = min_var.min(min_ref);
    let max_y = max_var.max(max_ref);
    let mut chart = ChartBuilder::on(&root)
        .caption(chart_name, ("sans-serif", 40).into_font())
        .x_label_area_size(25)
        .y_label_area_size(25)
        .build_cartesian_2d(0f32..*samples.last().unwrap(), min_y..max_y)?;

    chart
        .configure_mesh()
        .x_labels(5)
        .y_labels(5)
        .y_label_formatter(&|x| format!("{:.1}", x))
        .draw()?;

    let ref_lineseries_data = iter::zip(samples.clone(), ref_points).collect::<Vec<(f32, f32)>>();
    chart.draw_series(LineSeries::new(ref_lineseries_data, &RED))?;

    let var_lineseries_data = iter::zip(samples, var_points).collect::<Vec<(f32, f32)>>();
    chart.draw_series(LineSeries::new(var_lineseries_data, &BLUE))?;

    root.present()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wrap_min_max() {
        assert_eq!(wrap_min_max(0.0, 0.0, 1.0), 0.0);
        assert_eq!(wrap_min_max(1.0, 0.0, 1.0), 1.0);
        assert_eq!(wrap_min_max(2.0, 0.0, 1.0), 1.0);
        assert_eq!(wrap_min_max(3.0, 0.0, 1.0), 1.0);
        assert_eq!(wrap_min_max(4.0, 0.0, 1.0), 1.0);
        assert_eq!(wrap_min_max(5.0, 0.0, 1.0), 1.0);
        assert_eq!(wrap_min_max(6.0, 0.0, 1.0), 1.0);
    }

    #[test]
    fn test_wrap_angle_pmpi() {
        assert_eq!(wrap_angle_to_pmpi(0.0), 0.0);
        assert_eq!(wrap_angle_to_pmpi(consts::PI), consts::PI);
        assert_eq!(wrap_angle_to_pmpi(-consts::PI), -consts::PI);
        assert_eq!(wrap_angle_to_pmpi(2.0 * consts::PI), 0.0);
        assert_eq!(wrap_angle_to_pmpi(3.0 * consts::PI), -consts::PI);
        assert_eq!(wrap_angle_to_pmpi(-2.0 * consts::PI), 0.0);
        assert_eq!(wrap_angle_to_pmpi(-3.0 * consts::PI), consts::PI);
    }

    #[allow(non_snake_case)]
    #[test]
    fn test_damping_matrix() {
        let D_c = Matrix3::from_partial_diagonal(&[0.0, 0.0, 3224.0]);
        let D_q = Matrix3::from_partial_diagonal(&[135.0, 2000.0, 0.0]);
        let D_l = Matrix3::from_partial_diagonal(&[50.0, 200.0, 1281.0]);
        let nu: Vector3<f64> = Vector3::new(1.0, 1.0, 1.0);

        let Dmtrx_res = Dmtrx(D_l, D_q, D_c, nu);
        println!("Dmtrx_res: {:?}", Dmtrx_res);
    }

    #[allow(non_snake_case)]
    #[test]
    fn test_rotation_matrix() {
        let angle = consts::PI / 2.0;
        println!("angle: {:?}", angle);
        let Rmtrx_res = Rmtrx(angle);
        println!("Rmtrx_res: {:?}", Rmtrx_res);
        println!("Rmtrx[0, 1]: {:?}", Rmtrx_res[(0, 1)]);
        assert_eq!(Rmtrx_res[(0, 1)], -1.0);
        assert_eq!(Rmtrx_res * Rmtrx_res.transpose(), Matrix3::identity());
    }

    #[allow(non_snake_case)]
    #[test]
    fn test_coriolis_matrix() {
        let nu: Vector3<f64> = Vector3::new(1.0, 1.0, 1.0);
        let Mmtrx = Matrix3::from_partial_diagonal(&[3000.0, 3000.0, 19000.0]);
        let Cmtrx_res = Cmtrx(Mmtrx, nu);
        println!("Cmtrx_res: {:?}", Cmtrx_res);
    }
}
