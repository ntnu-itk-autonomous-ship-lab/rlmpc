//! # utils
//! Contains utility functions for the RRT* algorithm
//!
use crate::enc_hazards::ENCHazards;
use nalgebra::{Matrix3, Matrix2, Vector2, Vector3};
use rand::Rng;
use rand_chacha::ChaChaRng;

pub fn sample_free_position(enc: &ENCHazards, rng: &ChaChaRng) -> Vector2<f64> {
    loop {
        let mut x = rng.gen_range(enc.bbox.min().x..enc.bbox.max().x);
        let mut y = rng.gen_range(enc.bbox.min().y..enc.bbox.max().y);
        let p = Vector2::new(x, y);
        if !enc.inside_hazards(&p) {
            return p;
        }
    }
}

pub fn Cmtrx(Mmtrx: Matrix3<f64>, nu: Vector3<f64>) -> Matrix3<f64> {
    let mut Cmtrx = Matrix3::zeros();

    let c13 = -(Mmtrx[(1, 1)] * nu[1] + Mmtrx[(1, 2)] * nu[2]);
    let c23 = -Mmtrx[(0, 0)] * nu[0];
    Cmtrx[(0, 2)] = c13;
    Cmtrx[(1, 2)] = c23;
    Cmtrx[(2, 0)] = -c13;
    Cmtrx[(2, 1)] = -c23;
    Cmtrx
}

pub fn Dmtrx(
    D_l: Matrix3<f64>,
    D_q: Matrix3<f64>,
    D_c: Matrix3<f64>,
    nu: Vector3<f64>,
) -> Matrix3<f64> {
    D_l + D_q. * nu.abs() + D_c * (nu * nu)
}

pub fn Rmtrx(psi: f64) -> Matrix3<f64>{
    let mut Rmtrx = Matrix3::zeros();
    Rmtrx[(0, 0)] = psi.cos();
    Rmtrx[(0, 1)] = -psi.sin();
    Rmtrx[(1, 0)] = psi.sin();
    Rmtrx[(1, 1)] = psi.cos();
    Rmtrx[(2, 2)] = 1.0;
    Rmtrx
}

pub fn Rmtrx2D(psi: f64) -> Matrix2<f64>{
    let mut Rmtrx = Matrix2::zeros();
    Rmtrx[(0, 0)] = psi.cos();
    Rmtrx[(0, 1)] = -psi.sin();
    Rmtrx[(1, 0)] = psi.sin();
    Rmtrx[(1, 1)] = psi.cos();
    Rmtrx
}

pub fn wrap_min_max(x: f64, x_min: f64, x_max: f64) -> f64 {
    x_min + (x - x_min) % (x_max - x_min)
}

pub fn wrap_angle_to_pmpi(x: f64) -> f64 {
    wrap_min_max(x, -std::f64::consts::PI, std::f64::consts::PI)
}

pub fn wrap_angle_to_02pi(x: f64) -> f64 {
    wrap_min_max(x, 0.0, 2.0 * std::f64::consts::PI)
}

pub fn wrap_angle_diff_to_pmpi(x: f64, y: f64) -> f64 {
    let diff = wrap_angle_to_pmpi(x) - wrap_angle_to_pmpi(y);
    wrap_angle_to_pmpi(diff)
}
