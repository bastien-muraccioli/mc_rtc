/*
 * Copyright 2015-2026 CNRS-UM LIRMM, CNRS-AIST JRL
 */

#pragma once

#include <mc_tvm/api.h>

#include <mc_rbdyn/fwd.h>

#include <tvm/function/abstract/LinearFunction.h>

#include <deque>

namespace mc_tvm
{

/** Implement a derivative function for a given variable using a backward finite difference.
 *
 * This function can be used to constrain derivatives of variables that are not directly
 * available or updated by tvm_robot, such as joint jerk or torque derivatives.
 *
 * The order of the derivative can be specified through the \p order parameter. For example,
 * when \p var is the joint acceleration qdd:
 * - order = 1 computes the joint jerk qddd
 * - order = 2 computes the joint snap qdddd
 *
 * For variables whose derivatives are already available and updated by tvm_robot, it is
 * recommended to use tvm::dot() instead. For example, joint velocities are directly
 * available from tvm_robot, so tvm::dot(tvm_robot.qJoints(), 1) can be used to obtain
 * the joint velocities.
 *
 * In contrast, joint jerk is not a variable directly available or updated by tvm_robot.
 * In this case, DerivativeFunction can be used to compute and constrain the jerk, e.g.
 *
 *   qddd_min <= DerivativeFunction(robot, dt, qdd, 1) <= qddd_max
 *
 * Similarly, a second-order derivative can be constrained with:
 *
 *   qdddd_min <= DerivativeFunction(robot, dt, qdd, 2) <= qdddd_max
 */

struct MC_TVM_DLLAPI DerivativeFunction : public tvm::function::abstract::LinearFunction
{
public:
  using Output = tvm::function::abstract::LinearFunction::Output;

  DISABLE_OUTPUTS(Output::JDot)
  SET_UPDATES(DerivativeFunction, B)

  /** Construct the derivative function for a given variable
   *
   * The derivative is computed using a backward finite difference:
   *
   *   d^n x / dt^n =
   *       1 / dt^n * sum_{j=0}^{n} (-1)^j C(n,j) x_{k-j}
   *
   * The current value x_k is the optimization variable, while the
   * previous values are stored internally and contribute to b.
   *
   * For order = 1:
   *
   *   dx/dt = (x_k - x_{k-1}) / dt
   *
   * For order = 2:
   *
   *   d2x/dt2 = (x_k - 2*x_{k-1} + x_{k-2}) / dt^2
   *
   * \param robot Robot for which the function is built
   *
   * \param dt Time step used for the finite difference
   *
   * \param var Variable for which the derivative is computed
   *
   * \param order Order of the derivative of var. Defaults to 1.
   */
  DerivativeFunction(const mc_rbdyn::Robot & robot, double dt, tvm::VariablePtr var, int order = 1);

protected:
  /** Update the constant term b of the linear function */
  void updateb();

  /** Compute the binomial coefficient C(n, k) */
  static double binomialCoefficient(int n, int k);

  const mc_rbdyn::Robot & robot_;
  const double dt_;
  tvm::VariablePtr var_;

  /** Order of the derivative */
  const int order_;

  /** Previous values of the variable:
   *
   *   prev_var_[0] = x_{k-1}
   *   prev_var_[1] = x_{k-2}
   *   ...
   *   prev_var_[order_-1] = x_{k-order}
   */
  std::deque<Eigen::VectorXd> prev_var_;
};

using DerivativeFunctionPtr = std::shared_ptr<DerivativeFunction>;

} // namespace mc_tvm
