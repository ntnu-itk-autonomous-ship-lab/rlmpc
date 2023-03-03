//! # Steering
//! Implements a simple way of steering a Ship from a startpoint to an endpoint, using a simple surge and heading controller for a 3DOF surface ship model as in Tengesdal et. al. 2021, with LOS guidance.
//!
use crate::model::{ShipModel, ShipModelParams};
use crate::utils;
use nalgebra::Vector3;
use nalgebra::Vector6;
use std::f64;

pub trait Steering {
    fn steer(
        &mut self,
        xs_start: &Vector6<f64>,
        xs_goal: &Vector6<f64>,
        U_d: f64,
        acceptance_radius: f64,
        time_step: f64,
        max_steering_time: f64,
    ) -> (
        Vec<Vector6<f64>>,
        Vec<Vector3<f64>>,
        Vec<(f64, f64)>,
        f64,
        bool,
    );
}

/// Simple LOS guidance specialized for following 1 waypoint segment
#[allow(non_snake_case)]
pub struct LOSGuidance {
    K_p: f64,
    K_i: f64,
    max_cross_track_error_int: f64,
    cross_track_error_int: f64,
    cross_track_error_int_threshold: f64,
}

#[allow(non_snake_case)]
impl LOSGuidance {
    pub fn new() -> Self {
        Self {
            K_p: 0.035,
            K_i: 0.0,
            max_cross_track_error_int: 30.0,
            cross_track_error_int: 0.0,
            cross_track_error_int_threshold: 5.0,
        }
    }

    pub fn compute_refs(
        &mut self,
        xs_now: &Vector6<f64>,
        xs_start: &Vector6<f64>,
        xs_goal: &Vector6<f64>,
        U_d: f64,
        acceptance_radius: f64,
        dt: f64,
    ) -> (f64, f64) {
        let U_start = f64::sqrt(xs_start[3].powi(2) + xs_start[4].powi(2));
        let alpha = f64::atan2(xs_goal[1] - xs_start[1], xs_goal[0] - xs_start[0]);
        let cross_track_error = -(xs_now[0] - xs_goal[0]) * f64::sin(alpha)
            + (xs_now[1] - xs_goal[1]) * f64::cos(alpha);

        if cross_track_error.abs() > self.max_cross_track_error_int {
            self.cross_track_error_int = 0.0;
        }
        if cross_track_error.abs() <= self.cross_track_error_int_threshold {
            self.cross_track_error_int += cross_track_error * dt;
        }

        let chi_r =
            f64::atan(-self.K_p * cross_track_error - self.K_i * self.cross_track_error_int);
        let psi_d = utils::wrap_angle_to_pmpi(alpha + chi_r);
        (U_d, psi_d)
    }
}

#[allow(non_snake_case)]
struct FLSHController {
    K_p_u: f64,
    K_i_u: f64,
    K_p_psi: f64,
    K_d_psi: f64,
    K_i_psi: f64,
    max_U_error_int: f64,
    U_error_int: f64,
    U_error_int_threshold: f64,
    max_psi_error_int: f64,
    psi_error_int_threshold: f64,
    psi_error_int: f64,
    psi_d_prev: f64,
    psi_prev: f64,
}

#[allow(non_snake_case)]
impl FLSHController {
    pub fn new() -> Self {
        Self {
            K_p_u: 1.0,
            K_i_u: 0.05,
            K_p_psi: 3.0,
            K_d_psi: 3.0,
            K_i_psi: 0.005,
            max_U_error_int: 0.75,
            U_error_int: 0.0,
            U_error_int_threshold: 0.1,
            max_psi_error_int: 20.0 * f64::consts::PI / 180.0,
            psi_error_int_threshold: 10.0 * f64::consts::PI / 180.0,
            psi_error_int: 0.0,
            psi_d_prev: 0.0,
            psi_prev: 0.0,
        }
    }

    fn compute_inputs(
        &mut self,
        refs: &(f64, f64),
        xs: &Vector6<f64>,
        dt: f64,
        model_params: &ShipModelParams,
    ) -> Vector3<f64> {
        let psi: f64 = xs[2];
        let psi_unwrapped = utils::unwrap_angle(self.psi_prev, psi);
        let psi_d: f64 = refs.1;
        let psi_d_unwrapped = utils::unwrap_angle(self.psi_d_prev, psi_d);
        let psi_error: f64 = utils::wrap_angle_diff_to_pmpi(psi_d, psi);
        // println!("psi: {}, psi_d: {}, psi_error: {}", psi, psi_d, psi_error);
        let psi_error: f64 = utils::wrap_angle_diff_to_pmpi(psi_d_unwrapped, psi_unwrapped);
        // println!(
        //     "psi_unwrapped: {}, psi_d_unwrapped: {}, psi_error_unwrapped: {}",
        //     psi_unwrapped, psi_d_unwrapped, psi_error
        // );
        self.psi_prev = psi;
        self.psi_d_prev = psi_d;
        if psi_error.abs() > self.max_psi_error_int {
            self.psi_error_int -= psi_error * dt;
        }
        if psi_error.abs() <= self.psi_error_int_threshold {
            self.psi_error_int += psi_error * dt;
        }
        if psi_error.abs() <= 0.02 * f64::consts::PI / 180.0 {
            self.psi_error_int = 0.0;
        }
        self.psi_error_int = utils::wrap_angle_to_pmpi(self.psi_error_int);

        let U: f64 = f64::sqrt(xs[3].powi(2) + xs[4].powi(2));
        let U_d: f64 = refs.0;
        let U_error: f64 = U_d - U;
        if U_error.abs() > self.max_U_error_int {
            self.U_error_int -= U_error * dt;
        }
        if U_error.abs() <= self.U_error_int_threshold {
            self.U_error_int += U_error * dt;
        }
        if U_error.abs() <= 0.01 {
            self.U_error_int = 0.0;
        }
        let r: f64 = xs[5];

        let nu: Vector3<f64> = xs.fixed_rows::<3>(3).into();
        let Cvv: Vector3<f64> = utils::Cmtrx(model_params.M, nu) * nu;
        let Dvv: Vector3<f64> =
            utils::Dmtrx(model_params.D_l, model_params.D_q, model_params.D_c, nu) * nu;
        let Fx: f64 = Cvv[0]
            + Dvv[0]
            + model_params.M[(0, 0)] * (self.K_p_u * U_error + self.K_i_u * self.U_error_int);
        let Fx = utils::saturate(Fx, model_params.Fx_limits[0], model_params.Fx_limits[1]);
        let Fy: f64 = (model_params.M[(2, 2)] / model_params.l_r)
            * (self.K_p_psi * psi_error - self.K_d_psi * r + self.K_i_psi * self.psi_error_int);
        let Fy = utils::saturate(Fy, model_params.Fy_limits[0], model_params.Fy_limits[1]);
        let tau: Vector3<f64> = Vector3::new(Fx, Fy, Fy * model_params.l_r);

        // println!(
        //     "tau: {:?} | psi_error: {:.2} | u_diff: {:.2}",
        //     tau,
        //     psi_error,
        //     u_d - u
        // );
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
        &mut self,
        xs_start: &Vector6<f64>,
        xs_goal: &Vector6<f64>,
        U_d: f64,
        acceptance_radius: f64,
        time_step: f64,
        max_steering_time: f64,
    ) -> (
        Vec<Vector6<f64>>,
        Vec<Vector3<f64>>,
        Vec<(f64, f64)>,
        f64,
        bool,
    ) {
        let mut time = 0.0;
        let mut xs_array: Vec<Vector6<f64>> = vec![xs_start.clone()];
        let mut u_array: Vec<Vector3<f64>> = vec![];
        let mut refs_array: Vec<(f64, f64)> = vec![];
        let mut xs_next = xs_start.clone();
        let mut reached_goal = false;
        while time <= max_steering_time {
            let refs: (f64, f64) = self.los_guidance.compute_refs(
                &xs_next,
                xs_start,
                xs_goal,
                U_d,
                acceptance_radius,
                time_step,
            );

            let tau: Vector3<f64> = self.flsh_controller.compute_inputs(
                &refs,
                &xs_next,
                time_step,
                &self.ship_model.params,
            );
            xs_next = self.ship_model.erk4_step(time_step, &xs_next, &tau);

            refs_array.push(refs);
            xs_array.push(xs_next);
            u_array.push(tau);
            time += time_step;

            // Break if inside final waypoint acceptance radius
            let dist2goal =
                ((xs_goal[0] - xs_next[0]).powi(2) + (xs_goal[1] - xs_next[1]).powi(2)).sqrt();
            if dist2goal < acceptance_radius {
                reached_goal = true;
                break;
            }
        }
        (xs_array, u_array, refs_array, time, reached_goal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts;

    #[test]
    pub fn test_steer() -> Result<(), Box<dyn std::error::Error>> {
        let mut steering = SimpleSteering::new();
        let xs_start = Vector6::new(0.0, 0.0, consts::PI / 2.0, 5.0, 0.0, 0.0);
        let acceptance_radius = 10.0;
        let xs_goal = Vector6::new(100.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let (xs_array, u_array, refs_array, time, reached_goal) =
            steering.steer(&xs_start, &xs_goal, 5.0, acceptance_radius, 0.2, 70.0);
        println!("time: {:?}", time);
        assert!(xs_array.len() > 0);
        assert!(u_array.len() > 0);
        assert!(time > 0.0);

        let res = utils::draw_steering_results(
            &xs_start,
            &xs_goal,
            &refs_array,
            &xs_array,
            acceptance_radius,
        );
        Ok(())
    }
}
