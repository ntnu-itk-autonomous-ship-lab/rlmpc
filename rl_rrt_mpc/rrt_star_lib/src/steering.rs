//! # Steering
//! Implements a simple way of steering a Ship from a startpoint to an endpoint, using a simple surge and heading controller for a 3DOF surface ship model as in Tengesdal et. al. 2021, with LOS guidance.
//!
use crate::model::{KinematicCSOG, ShipModel, Telemetron, TelemetronParams};
use crate::utils;
use nalgebra::Vector3;
use nalgebra::Vector6;
use std::f64;

#[allow(non_snake_case)]
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
        Vec<f64>,
        bool,
    );

    fn steer_through_waypoints(
        &mut self,
        xs_start: &Vector6<f64>,
        waypoints: &Vec<Vector6<f64>>,
        U_d: f64,
        acceptance_radius: f64,
        time_step: f64,
        max_steering_time: f64,
    ) -> (
        Vec<Vector6<f64>>,
        Vec<Vector3<f64>>,
        Vec<(f64, f64)>,
        Vec<f64>,
        bool,
    ) {
        let radius = acceptance_radius;
        let mut t_array: Vec<f64> = Vec::new();
        let mut xs_array: Vec<Vector6<f64>> = Vec::new();
        let mut u_array: Vec<Vector3<f64>> = Vec::new();
        let mut refs_array: Vec<(f64, f64)> = Vec::new();
        let mut reached_last = false;
        let n_wps = waypoints.len();
        assert_eq!(n_wps > 1, true);
        let mut xs_current = xs_start.clone();
        let mut wp_idx = 0;
        while wp_idx < n_wps - 1 {
            let (mut xs_array_, u_array_, refs_array_, t_array_, reached) = self.steer(
                &xs_current,
                &waypoints[wp_idx + 1].into(),
                U_d,
                radius,
                time_step,
                max_steering_time,
            );
            let last_psi_diff = xs_array_.last().unwrap()[2] - waypoints[wp_idx + 1][2];
            reached_last = reached && last_psi_diff.abs() < 3.0 * 180.0 / f64::consts::PI;
            xs_current = xs_array_.last().unwrap().clone().into();
            wp_idx += 1;
            xs_array_.pop();
            xs_array.extend(xs_array_);
            u_array.extend(u_array_);
            refs_array.extend(refs_array_);
            t_array.extend(
                t_array_
                    .iter()
                    .map(|t| {
                        if t_array.len() > 0 {
                            t + t_array.last().unwrap().clone()
                        } else {
                            *t
                        }
                    })
                    .collect::<Vec<f64>>(),
            );
        }
        (xs_array, u_array, refs_array, t_array, reached_last)
    }
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

    pub fn reset(&mut self) {
        self.cross_track_error_int = 0.0;
    }

    pub fn compute_refs(
        &mut self,
        xs_now: &Vector6<f64>,
        xs_start: &Vector6<f64>,
        xs_goal: &Vector6<f64>,
        U_d: f64,
        dt: f64,
    ) -> (f64, f64) {
        let alpha = (xs_goal[1] - xs_start[1]).atan2(xs_goal[0] - xs_start[0]);
        let cross_track_error = -(xs_now[0] - xs_goal[0]) * f64::sin(alpha)
            + (xs_now[1] - xs_goal[1]) * f64::cos(alpha);

        if cross_track_error.abs() <= self.cross_track_error_int_threshold {
            self.cross_track_error_int += cross_track_error * dt;
        }
        if self.cross_track_error_int.abs() > self.max_cross_track_error_int {
            self.cross_track_error_int -= cross_track_error * dt;
        }

        let chi_r =
            (-self.K_p * cross_track_error - self.K_i * self.cross_track_error_int).atan2(1.0);
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
            U_error_int_threshold: 0.2,
            max_psi_error_int: 20.0 * f64::consts::PI / 180.0,
            psi_error_int_threshold: 10.0 * f64::consts::PI / 180.0,
            psi_error_int: 0.0,
            psi_d_prev: 0.0,
            psi_prev: 0.0,
        }
    }

    fn reset(&mut self) {
        self.U_error_int = 0.0;
        self.psi_error_int = 0.0;
        self.psi_d_prev = 0.0;
        self.psi_prev = 0.0;
    }

    fn compute_inputs(
        &mut self,
        refs: &(f64, f64),
        xs: &Vector6<f64>,
        dt: f64,
        model_params: &TelemetronParams,
    ) -> Vector3<f64> {
        let psi: f64 = utils::wrap_angle_to_pmpi(xs[2]);
        let psi_unwrapped = utils::unwrap_angle(self.psi_prev, psi);
        let psi_d: f64 = refs.1;
        let psi_d_unwrapped = utils::unwrap_angle(self.psi_d_prev, psi_d);
        let psi_error: f64 = utils::wrap_angle_diff_to_pmpi(psi_d_unwrapped, psi_unwrapped);
        // if (psi_d < 0.0 && psi > 0.0) || (psi_d > 0.0 && psi < 0.0) {
        //     println!("psi_d={psi_d} | psi_d_unwrapped={psi_d_unwrapped} | psi={psi} | psi_unwrapped={psi_unwrapped} | psi_error={psi_error}");
        // }
        self.psi_prev = psi;
        self.psi_d_prev = psi_d;
        if self.psi_error_int.abs() > self.max_psi_error_int {
            self.psi_error_int -= psi_error * dt;
        }
        if psi_error.abs() <= self.psi_error_int_threshold {
            self.psi_error_int += psi_error * dt;
        }

        self.psi_error_int = utils::wrap_angle_to_pmpi(self.psi_error_int);

        let U: f64 = f64::sqrt(xs[3].powi(2) + xs[4].powi(2));
        let U_d: f64 = refs.0;
        let U_error: f64 = U_d - U;
        if self.U_error_int.abs() > self.max_U_error_int {
            self.U_error_int -= U_error * dt;
        }
        if U_error.abs() <= self.U_error_int_threshold {
            self.U_error_int += U_error * dt;
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

pub struct SimpleSteering<M: ShipModel> {
    los_guidance: LOSGuidance,
    flsh_controller: FLSHController,
    ship_model: M,
}

impl<M: ShipModel> SimpleSteering<M> {
    pub fn new() -> SimpleSteering<M> {
        Self {
            los_guidance: LOSGuidance::new(),
            flsh_controller: FLSHController::new(),
            ship_model: M::new(),
        }
    }
}

#[allow(non_snake_case)]
impl Steering for SimpleSteering<Telemetron> {
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
        Vec<f64>,
        bool,
    ) {
        let mut time = 0.0;
        let mut t_array = vec![];
        let mut xs_array: Vec<Vector6<f64>> = vec![xs_start.clone()];
        let mut u_array: Vec<Vector3<f64>> = vec![];
        let mut refs_array: Vec<(f64, f64)> = vec![];
        let mut xs_next = xs_start.clone();
        let mut reached_goal = false;
        self.los_guidance.reset();
        self.flsh_controller.reset();
        //println!("xs_start: {:?} | xs_goal: {:?}", xs_start, xs_goal);
        while time <= max_steering_time {
            let refs: (f64, f64) = self
                .los_guidance
                .compute_refs(&xs_next, xs_start, xs_goal, U_d, time_step);

            let tau: Vector3<f64> = self.flsh_controller.compute_inputs(
                &refs,
                &xs_next,
                time_step,
                &self.ship_model.params(),
            );
            xs_next = self.ship_model.erk4_step(time_step, &xs_next, &tau);

            refs_array.push(refs);
            u_array.push(tau);
            t_array.push(time.clone());
            time += time_step;

            xs_array.push(xs_next);
            // Break if inside final waypoint acceptance radius
            let dist2goal =
                ((xs_goal[0] - xs_next[0]).powi(2) + (xs_goal[1] - xs_next[1]).powi(2)).sqrt();
            if dist2goal < acceptance_radius {
                reached_goal = true;
                // refs_array.push(refs);
                // u_array.push(tau);
                // t_array.push(time.clone());
                break;
            }
        }
        //println!("xs_next: {:?} | time: {:.2}", xs_next, time);
        (xs_array, u_array, refs_array, t_array, reached_goal)
    }
}

#[allow(non_snake_case)]
impl Steering for SimpleSteering<KinematicCSOG> {
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
        Vec<f64>,
        bool,
    ) {
        let mut time = 0.0;
        let mut t_array = vec![];
        let mut xs_array: Vec<Vector6<f64>> = vec![xs_start.clone()];
        let mut u_array: Vec<Vector3<f64>> = vec![];
        let mut refs_array: Vec<(f64, f64)> = vec![];
        let mut xs_next = xs_start.clone();
        let mut reached_goal = false;
        //println!("xs_start: {:?} | xs_goal: {:?}", xs_start, xs_goal);
        while time <= max_steering_time {
            let refs: (f64, f64) = self
                .los_guidance
                .compute_refs(&xs_next, xs_start, xs_goal, U_d, time_step);

            let tau: Vector3<f64> = Vector3::new(refs.0, refs.1, 0.0);
            xs_next = self.ship_model.erk4_step(time_step, &xs_next, &tau);

            refs_array.push(refs);
            u_array.push(tau);
            t_array.push(time.clone());
            time += time_step;

            xs_array.push(xs_next);
            // Break if inside final waypoint acceptance radius
            let dist2goal =
                ((xs_goal[0] - xs_next[0]).powi(2) + (xs_goal[1] - xs_next[1]).powi(2)).sqrt();
            if dist2goal < acceptance_radius {
                reached_goal = true;
                // refs_array.push(refs);
                // u_array.push(tau);
                // t_array.push(time.clone());
                break;
            }
        }
        //println!("xs_next: {:?} | time: {:.2}", xs_next, time);
        (xs_array, u_array, refs_array, t_array, reached_goal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts;

    #[test]
    pub fn test_steer() -> Result<(), Box<dyn std::error::Error>> {
        let mut steering = SimpleSteering::<Telemetron>::new();
        let xs_start = Vector6::new(0.0, 0.0, consts::PI / 2.0, 5.0, 0.0, 0.0);
        let acceptance_radius = 10.0;
        let xs_goal = Vector6::new(100.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let (xs_array, u_array, refs_array, t_array, _) =
            steering.steer(&xs_start, &xs_goal, 5.0, acceptance_radius, 0.2, 70.0);
        println!("time: {:?}", t_array.last().unwrap().clone());
        assert!(xs_array.len() > 0);
        assert!(u_array.len() > 0);
        assert!(t_array.last().unwrap().clone() > 0.0);

        let _ = utils::draw_steering_results(
            &xs_start,
            &xs_goal,
            &refs_array,
            &xs_array,
            acceptance_radius,
        );
        Ok(())
    }
}
