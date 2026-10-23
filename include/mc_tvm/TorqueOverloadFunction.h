/*
 * Copyright 2015-2026 CNRS-UM LIRMM, CNRS-AIST JRL
 */

#pragma once

#include <mc_tvm/api.h>

#include <mc_rbdyn/fwd.h>

#include <tvm/function/abstract/LinearFunction.h>

#include <vector>

namespace mc_tvm
{

/** Time-limited torque overload constraint (I²t / thermal budget) expressed on joint torques only.
 *
 * The robot model torque limits [tl, tu] are treated as the *continuous* rating of each actuator.
 * The actuator may exceed them during a limited time, up to \p peakRatio times the limit, as long as
 * a per-joint thermal budget is not exhausted. No current measurement is needed: since tau = k_t * i,
 * the I²t integral can be written directly on torque.
 *
 * For each limited joint i, with tau_c,i = max(|tl_i|, |tu_i|), the normalized heat budget (in seconds) is
 *
 *   e_i(k+1) = clamp( e_i(k) + ((tau_i(k) / tau_c,i)^2 - 1) * dt , 0, e_max )
 *   e_max    = (peakRatio^2 - 1) * peakDuration
 *
 * where tau_i(k) is the torque obtained at the previous solver iteration (commanded torque).
 * The budget charges when |tau| > tau_c and recovers when |tau| < tau_c.
 *
 * At each iteration, the allowed scale on the continuous limits is the largest value that keeps the budget
 * below e_max at the end of the step:
 *
 *   s_i = min( peakRatio , sqrt( 1 + (e_max - e_i) / dt ) ),  s_i >= 1
 *
 * and the function encodes s_i * tl_i <= tau_i <= s_i * tu_i as a single linear function that must be
 * non-positive:
 *
 *   f(tau) = [  S tau - s .* tu_S ]  <= 0
 *            [ -S tau + s .* tl_S ]
 *
 * where S selects the joints with finite, non-zero limits. The Jacobian is constant, only b is time-varying,
 * so the constraint stays linear in the QP (and remains compatible with the tau substitution of the
 * equation of motion).
 *
 * With peakRatio = 2 and peakDuration = 1 s, a joint starting cold can deliver twice its limit for 1 s,
 * or any (tau, t) pair with the same I²t energy, e.g. 1.5x for ~2.3 s.
 *
 * This function does NOT enforce the hard peak cap on the joints it ignores (floating base, unlimited joints);
 * the hard bound |tau| <= peakRatio * limit should also be added as a regular box constraint
 * (mc_solver::DynamicsConstraint does this).
 */
struct MC_TVM_DLLAPI TorqueOverloadFunction : public tvm::function::abstract::LinearFunction
{
public:
  using Output = tvm::function::abstract::LinearFunction::Output;

  DISABLE_OUTPUTS(Output::JDot)
  SET_UPDATES(TorqueOverloadFunction, B)

  /** Constructor
   *
   * \param robot Robot for which the function is built (its tvmRobot() limits are used as continuous limits)
   *
   * \param dt Solver time step
   *
   * \param peakRatio Maximum allowed torque as a multiple of the continuous limit (>= 1)
   *
   * \param peakDuration Time during which peakRatio * limit can be held starting from a full budget [s] (> 0)
   */
  TorqueOverloadFunction(const mc_rbdyn::Robot & robot, double dt, double peakRatio = 2.0, double peakDuration = 1.0);

  /** Reset the thermal budget of every joint (as if the actuators were cold) */
  void reset();

  /** Normalized consumed budget of each limited joint [s], in [0, maxBudget()] */
  inline const Eigen::VectorXd & budget() const noexcept { return budget_; }

  /** Remaining fraction of the budget of each limited joint, 1 = full, 0 = exhausted */
  inline Eigen::VectorXd remainingBudget() const
  {
    return (Eigen::VectorXd::Ones(budget_.size()) - budget_ / maxBudget_).cwiseMax(0.0);
  }

  /** Current scale applied to the continuous limits of each limited joint, in [1, peakRatio] */
  inline const Eigen::VectorXd & scale() const noexcept { return scale_; }

  /** Current upper torque bound of each limited joint */
  inline Eigen::VectorXd upperBound() const { return scale_.cwiseProduct(tu_); }

  /** Current lower torque bound of each limited joint */
  inline Eigen::VectorXd lowerBound() const { return scale_.cwiseProduct(tl_); }

  /** Indices (in the robot dof vector) of the joints handled by this function */
  inline const std::vector<Eigen::Index> & indices() const noexcept { return indices_; }

  inline double peakRatio() const noexcept { return peakRatio_; }

  inline double peakDuration() const noexcept { return peakDuration_; }

  inline double maxBudget() const noexcept { return maxBudget_; }

protected:
  /** Integrate the budget with the last torque, compute the new scale and update b */
  void updateB();

  /** Compute the scale from the current budget and write b */
  void computeBoundsFromBudget();

  const mc_rbdyn::Robot & robot_;
  tvm::VariablePtr tau_;
  const double dt_;
  const double peakRatio_;
  const double peakDuration_;
  const double maxBudget_;

  /** Indices of the limited joints in the dof vector */
  std::vector<Eigen::Index> indices_;
  /** Continuous limits of the limited joints */
  Eigen::VectorXd tl_;
  Eigen::VectorXd tu_;
  /** Inverse of the squared nominal (continuous) torque of the limited joints */
  Eigen::VectorXd invTauC2_;

  Eigen::VectorXd budget_;
  Eigen::VectorXd scale_;

  /** False until the first solver iteration (no previous torque to integrate) */
  bool initialized_ = false;
};

using TorqueOverloadFunctionPtr = std::shared_ptr<TorqueOverloadFunction>;

} // namespace mc_tvm
