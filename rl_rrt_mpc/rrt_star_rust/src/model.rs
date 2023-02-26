//! # Model
//! Implements a 3DOF surface ship model as in Tengesdal et. al. 2021:
//!
//! eta_dot = Rpsi(eta) * nu
//! M * nu_dot + C(nu) * nu + (D_l(nu) + D_nl) * nu = tau
//!
//! with eta = [x, y, psi]^T, nu = [u, v, r]^T and xs = [eta, nu]^T.
//!
//! Parameters:
//!    M: Rigid body mass matrix
//!    C: Coriolis matrix, computed from M = M_rb + M_a
//!    D_l: Linear damping matrix
//!    D_q: Nonlinear damping matrix
//!    D_c: Nonlinear damping matrix
//!
//! NOTE: When using Euler`s method, keep the time step small enough (e.g. around 0.1 or less) to ensure numerical stability.
//!
use crate::utils;
use nalgebra::Vector6;
use nalgebra::{Matrix3, Vector2, Vector3};
use std::f64::consts::PI;

#[allow(non_snake_case)]
#[derive(Debug, Clone, Copy)]
pub struct ShipModelParams {
    pub draft: f64,
    pub length: f64,
    pub width: f64,
    pub l_r: f64,
    pub M_inv: Matrix3<f64>,
    pub M: Matrix3<f64>,
    pub D_c: Matrix3<f64>,
    pub D_q: Matrix3<f64>,
    pub D_l: Matrix3<f64>,
    pub Fx_limits: Vector2<f64>,
    pub Fy_limits: Vector2<f64>,
    pub r_max: f64,
    pub U_max: f64,
    pub U_min: f64,
}

#[allow(non_snake_case)]
impl ShipModelParams {
    pub fn new() -> Self {
        let r_max = 15.0 * PI / 180.0;
        let M_inv = Matrix3::from_partial_diagonal(&[1.0 / 3980.0, 1.0 / 3980.0, 1.0 / 19703.0]);
        Self {
            draft: 1.0,
            length: 10.0,
            width: 3.0,
            l_r: 4.0,
            M_inv: M_inv,
            M: M_inv.try_inverse().unwrap(),
            D_c: Matrix3::from_partial_diagonal(&[0.0, 0.0, 3224.0]),
            D_q: Matrix3::from_partial_diagonal(&[135.0, 2000.0, 0.0]),
            D_l: Matrix3::from_partial_diagonal(&[50.0, 200.0, 1281.0]),
            Fx_limits: Vector2::new(-6550.0, 13100.0),
            Fy_limits: Vector2::new(-645.0, 645.0),
            r_max: r_max,
            U_max: 15.0,
            U_min: 0.0,
        }
    }
}

pub struct ShipModel {
    pub params: ShipModelParams,
    pub n_x: usize,
    pub n_u: usize,
}

impl ShipModel {
    pub fn new() -> Self {
        Self {
            params: ShipModelParams::new(),
            n_x: 6,
            n_u: 3,
        }
    }

    #[allow(non_snake_case)]
    pub fn dynamics(&self, xs: &Vector6<f64>, tau: &Vector3<f64>) -> Vector6<f64> {
        let eta: Vector3<f64> = xs.fixed_rows::<3>(0).into();
        let nu: Vector3<f64> = xs.fixed_rows::<3>(3).into();

        let Dmtrx = utils::Dmtrx(self.params.D_l, self.params.D_q, self.params.D_c, nu);

        let eta_dot: Vector3<f64> = (utils::Rmtrx(eta[2]) * nu).into();
        let nu_dot: Vector3<f64> =
            (self.params.M_inv * (tau - utils::Cmtrx(self.params.M, nu) * nu - Dmtrx * nu)).into();
        let mut xs_dot: Vector6<f64> = Vector6::zeros();
        xs_dot.fixed_rows_mut::<3>(0).copy_from(&eta_dot);
        xs_dot.fixed_rows_mut::<3>(3).copy_from(&nu_dot);
        xs_dot
    }

    pub fn erk4_step(&self, dt: f64, xs: &Vector6<f64>, tau: &Vector3<f64>) -> Vector6<f64> {
        let k1 = self.dynamics(xs, tau);
        let k2 = self.dynamics(&(xs + dt * k1 / 2.0), tau);
        let k3 = self.dynamics(&(xs + dt * k2 / 2.0), tau);
        let k4 = self.dynamics(&(xs + dt * k3), tau);
        xs + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    }

    pub fn euler_step(&self, dt: f64, xs: &Vector6<f64>, tau: &Vector3<f64>) -> Vector6<f64> {
        xs + dt * self.dynamics(xs, tau)
    }
}
