//! # Steering
//! Implements a simple way of steering a Ship from a startpoint to an endpoint, using a simple surge and heading controller for a 3DOF surface ship model as in Tengesdal et. al. 2021, with LOS guidance.
//!
use crate::model::{ShipModel, ShipModelParams};
use crate::utils;
use nalgebra::Vector3;
use nalgebra::Vector6;
use std::f64;
use std::f64::consts;

pub trait Steering {
    fn steer(
        &self,
        xs_start: &Vector6<f64>,
        xs_goal: &Vector6<f64>,
        time_step: f64,
        max_steering_time: f64,
    ) -> (Vec<Vector6<f64>>, Vec<Vector3<f64>>, f64);
}

/// Simple LOS guidance specialized for following 1 waypoint segment
#[allow(non_snake_case)]
pub struct LOSGuidance {
    K_p: f64,
    K_i: f64,
    T_u: f64,
    R_a: f64,
}

#[allow(non_snake_case)]
impl LOSGuidance {
    pub fn new() -> Self {
        Self {
            K_p: 0.015,
            K_i: 0.0,
            T_u: 20.0,
            R_a: 20.0,
        }
    }

    pub fn compute_refs(&self, xs_start: &Vector6<f64>, xs_goal: &Vector6<f64>) -> (f64, f64) {
        let dist_to_goal =
            f64::sqrt((xs_goal[0] - xs_start[0]).powi(2) + (xs_goal[1] - xs_start[1]).powi(2));
        let U_start = f64::sqrt(xs_start[3].powi(2) + xs_start[4].powi(2));
        let U_goal = f64::sqrt(xs_goal[3].powi(2) + xs_goal[4].powi(2));
        let U_d = U_start * f64::exp(-dist_to_goal / self.T_u) + U_goal;
        let alpha = f64::atan2(xs_goal[1] - xs_start[1], xs_goal[0] - xs_start[0]);
        let cross_track_error = -(xs_start[0] - xs_goal[0]) * f64::sin(alpha)
            + (xs_start[1] - xs_goal[1]) * f64::cos(alpha);

        let chi_r = f64::atan(self.K_p * cross_track_error);
        let psi_d = utils::wrap_angle_to_pmpi(alpha + chi_r);

        if dist_to_goal < self.R_a {
            return (0.0, psi_d);
        }

        (U_d, psi_d)
    }
}

#[allow(non_snake_case)]
struct FLSHController {
    K_p_u: f64,
    K_p_psi: f64,
    K_d_psi: f64,
}

#[allow(non_snake_case)]
impl FLSHController {
    pub fn new() -> Self {
        Self {
            K_p_u: 5.0,
            K_p_psi: 6.0,
            K_d_psi: 12.0,
        }
    }

    fn compute_inputs(
        &self,
        refs: &(f64, f64),
        xs: &Vector6<f64>,
        model_params: &ShipModelParams,
    ) -> Vector3<f64> {
        let psi = xs[2];
        let psi_d = refs.1;
        let psi_diff = utils::wrap_angle_diff_to_pmpi(psi_d, psi);

        let u = xs[3];
        let u_d = refs.0;
        let r = xs[5];

        let nu: Vector3<f64> = xs.fixed_rows::<3>(3).into();
        let Cvv = utils::Cmtrx(model_params.M, nu) * nu;
        let Dvv = utils::Dmtrx(model_params.D_l, model_params.D_q, model_params.D_c, nu) * nu;
        let Fx = Cvv[0] + Dvv[0] + model_params.M[(0, 0)] * self.K_p_u * (u_d - u);
        let Fy = (model_params.M[(2, 2)] / model_params.l_r)
            * (self.K_p_psi * psi_diff - self.K_d_psi * r);
        let tau = Vector3::new(Fx, Fy, Fy * model_params.l_r);

        println!(
            "tau: {:?} | psi_diff: {:?} | u_diff: {:?}",
            tau,
            psi_diff,
            u_d - u
        );
        tau
    }
}

pub struct SimpleSteering {
    los_guidance: LOSGuidance,
    flsh_controller: FLSHController,
    ship_model: ShipModel,
}

impl SimpleSteering {
    pub fn new() -> Self {
        Self {
            los_guidance: LOSGuidance::new(),
            flsh_controller: FLSHController::new(),
            ship_model: ShipModel::new(),
        }
    }
}

impl Steering for SimpleSteering {
    fn steer(
        &self,
        xs_start: &Vector6<f64>,
        xs_goal: &Vector6<f64>,
        time_step: f64,
        max_steering_time: f64,
    ) -> (Vec<Vector6<f64>>, Vec<Vector3<f64>>, f64) {
        let mut time = 0.0;
        let mut xs_array: Vec<Vector6<f64>> = vec![xs_start.clone()];
        let mut u_array: Vec<Vector3<f64>> = vec![];
        while time < max_steering_time {
            let refs: (f64, f64) = self.los_guidance.compute_refs(xs_start, xs_goal);
            // Break if the desired speed is 0 => we are at the goal
            if refs.0 <= 0.001 {
                break;
            }
            let tau: Vector3<f64> =
                self.flsh_controller
                    .compute_inputs(&refs, xs_start, &self.ship_model.params);
            let xs_next: Vector6<f64> = self.ship_model.erk4_step(time_step, xs_start, &tau);

            xs_array.push(xs_next);
            u_array.push(tau);
            time += time_step;
        }
        (xs_array, u_array, time)
    }
}
