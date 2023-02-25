//! # utils
//! Contains utility functions for the RRT* algorithm
//!
use crate::enc_hazards::ENCHazards;
use nalgebra::{Matrix2, Matrix3, Vector2, Vector3};
use rand::Rng;
use rand_chacha::ChaChaRng;
use std::f64::consts;

pub fn sample_free_position(enc: &ENCHazards, rng: &mut ChaChaRng) -> Vector2<f64> {
    loop {
        let x = rng.gen_range(enc.bbox.min().x..enc.bbox.max().x);
        let y = rng.gen_range(enc.bbox.min().y..enc.bbox.max().y);
        let p = Vector2::new(x, y);
        if !enc.inside_hazards(&p) {
            return p;
        }
    }
}

#[allow(non_snake_case)]
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
    println!("D_l: {:?}", D_l);
    println!("D_q: {:?}", D_q);
    println!("D_q_res: {:?}", D_q_res);
    println!("D_c: {:?}", D_c);
    println!("D_c_res: {:?}", D_c_res);
    println!("nu_squared: {:?}", nu_squared);
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

#[allow(non_snake_case)]
pub fn Rmtrx2D(psi: f64) -> Matrix2<f64> {
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
    wrap_min_max(x, -consts::PI, consts::PI)
}

pub fn wrap_angle_to_02pi(x: f64) -> f64 {
    wrap_min_max(x, 0.0, 2.0 * consts::PI)
}

pub fn wrap_angle_diff_to_pmpi(x: f64, y: f64) -> f64 {
    let diff = wrap_angle_to_pmpi(x) - wrap_angle_to_pmpi(y);
    wrap_angle_to_pmpi(diff)
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
