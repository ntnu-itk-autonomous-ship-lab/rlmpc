//! # Steering
//! Implements a simple way of steering a Ship from a startpoint to an endpoint, using a simple surge and heading controller for a 3DOF surface ship model as in Tengesdal et. al. 2021, with LOS guidance.
//!
use crate::model::ShipModelParams;
use crate::utils;
use nalgebra::Vector6;
use nalgebra::{Matrix3, Vector2, Vector3};
use std::f64::consts;

pub trait Steering {
    pub fn steer(
        &self,
        xs_start: &Vector6<f64>,
        xs_goal: &Vector6<f64>,
        model_params: &ShipModelParams,
    ) -> Vector3<f64>;
}

pub trait IController {
    fn compute_inputs(
        &self,
        refs: &Vector9<f64>,
        xs: &Vector6<f64>,
        model_params: &ShipModelParams,
    ) -> Vector3<f64>;
}

pub struct LOSGuidanceParams {
    pub K_p_u: f64,
    pub K_p_psi: f64,
    pub K_d_psi: f64,
}

pub struct LOSGuidance {
    params: LOSGuidanceParams,
}

struct FLSHParams {
    K_p_u: f64,
    K_p_psi: f64,
    K_d_psi: f64,
}

impl FLSHParams {
    pub fn new() -> Self {
        Self {
            K_p_u: 5.0,
            K_p_psi: 6.0,
            K_d_psi: 12.0,
        }
    }
}

struct FLSHController {
    params: FLSHParams,
}

impl FLSHController {
    pub fn new() -> Self {
        Self {
            params: FLSHParams::new(),
        }
    }
}

impl IController for FLSHController {
    fn compute_inputs(
        &self,
        refs: &Vector9<f64>,
        xs: &Vector6<f64>,
        model_params: &ShipModelParams,
    ) -> Vector3<f64> {
        let psi = xs[2];
        let psi_d = refs[2];
        let psi_diff = utils::wrap_angle_diff_to_pmpi(psi_d, psi);

        let u = xs[3];
        let u_d = refs[3];
        let r_d = refs[5];

        let nu = xs.fixed_rows::<3>(3);
        let Cvv = utils::Cmtrx(model_params.M, nu) * nu;
        let Dvv =
            Dmtrx = utils::Dmtrx(model_params.D_l, model_params.D_q, model_params.D_c, nu) * nu;
        let Fx = Cvv[0] + Dvv[0] + model_params.M[(0, 0)] * self.params.K_p_u * (u_d - u);
        let Fy = (model_params.M[(2, 2)] / model_params.l_r)
            * (self.params.K_p_psi * psi_diff - self.params.K_d_psi * r_d);
        tau
    }
}
