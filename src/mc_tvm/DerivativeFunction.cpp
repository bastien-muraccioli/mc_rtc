/*
 * Copyright 2015-2026 CNRS-UM LIRMM, CNRS-AIST JRL
 */

#include <mc_tvm/DerivativeFunction.h>

#include <mc_tvm/Robot.h>
#include <mc_tvm/RobotFrame.h>

#include <mc_rtc/logging.h>

#include <cmath>
#include <stdexcept>

namespace mc_tvm
{

DerivativeFunction::DerivativeFunction(const mc_rbdyn::Robot & robot, double dt, tvm::VariablePtr var, int order)
: tvm::function::abstract::LinearFunction(int(var->value().size())), robot_(robot), dt_(dt), var_(var), order_(order)
{
  if(dt_ <= 0.0) { throw std::invalid_argument("DerivativeFunction: dt must be strictly positive"); }

  if(order_ < 1) { throw std::invalid_argument("DerivativeFunction: order must be >= 1"); }

  registerUpdates(Update::B, &DerivativeFunction::updateb);
  addOutputDependency<DerivativeFunction>(Output::B, Update::B);
  addVariable(var_, true);

  const auto variableSize = var_->value().size();

  /*
   * For the backward finite difference:
   *
   * d^n x / dt^n =
   *   1 / dt^n *
   *   [C(n,0)x_k + C(n,1)(-x_{k-1}) + ...]
   *
   * The current variable x_k is the optimization variable, therefore
   * only the j=0 term contributes to the Jacobian.
   *
   * C(n,0) = 1
   *
   * Hence:
   *
   *   A = I / dt^n
   */
  jacobian_[var_.get()] = Eigen::MatrixXd::Identity(variableSize, variableSize) / std::pow(dt_, order_);

  /*
   * We need order previous values:
   *
   *   order = 1 -> x_{k-1}
   *   order = 2 -> x_{k-1}, x_{k-2}
   *   order = 3 -> x_{k-1}, x_{k-2}, x_{k-3}
   *   ...
   *
   * Initialize the history with the current value. This avoids using
   * uninitialized data during the first update.
   */
  prev_var_.resize(static_cast<size_t>(order_), var_->value());

  b_.setZero(variableSize);

  /*
   * Initialize b consistently with the current history.
   */
  updateb();
}

double DerivativeFunction::binomialCoefficient(int n, int k)
{
  if(k < 0 || k > n) { return 0.0; }

  /*
   * C(n, 0) = 1
   */
  if(k == 0) { return 1.0; }

  /*
   * Use the symmetry of the binomial coefficients to reduce
   * the number of iterations.
   */
  k = std::min(k, n - k);

  double result = 1.0;

  for(int i = 1; i <= k; ++i)
  {
    result *= static_cast<double>(n - i + 1);
    result /= static_cast<double>(i);
  }

  return result;
}

void DerivativeFunction::updateb()
{
  const auto variableSize = var_->value().size();

  b_.setZero(variableSize);

  const double dt_power = std::pow(dt_, order_);

  /*
   * The backward finite difference is:
   *
   * d^n x / dt^n =
   *
   *   1 / dt^n *
   *   sum_{j=0}^{n} (-1)^j C(n,j) x_{k-j}
   *
   * The j=0 term is x_k and is represented by the Jacobian.
   *
   * Therefore b contains:
   *
   *   j=1 ... n
   */
  for(int j = 1; j <= order_; ++j)
  {
    const double sign = (j % 2 == 0) ? 1.0 : -1.0;

    const double coefficient = sign * binomialCoefficient(order_, j);

    b_ += coefficient * prev_var_[static_cast<size_t>(j - 1)];
  }

  b_ = b_ / dt_power;

  /*
   * Update the history:
   *
   *   x_{k-1} <- x_k
   *   x_{k-2} <- x_{k-1}
   *   ...
   *
   * We update from the back to avoid overwriting values that are
   * still needed.
   */
  for(int j = order_ - 1; j > 0; --j) { prev_var_[static_cast<size_t>(j)] = prev_var_[static_cast<size_t>(j - 1)]; }

  prev_var_[0] = var_->value();
}

} // namespace mc_tvm
