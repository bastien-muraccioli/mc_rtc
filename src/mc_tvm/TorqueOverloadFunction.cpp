/*
 * Copyright 2015-2026 CNRS-UM LIRMM, CNRS-AIST JRL
 */

#include <mc_tvm/TorqueOverloadFunction.h>

#include <mc_tvm/Robot.h>

#include <mc_rbdyn/Robot.h>

#include <mc_rtc/logging.h>

#include <cmath>

namespace mc_tvm
{

TorqueOverloadFunction::TorqueOverloadFunction(const mc_rbdyn::Robot & robot,
                                               double dt,
                                               double peakRatio,
                                               double peakDuration)
: tvm::function::abstract::LinearFunction(0), robot_(robot), tau_(robot.tvmRobot().tau()), dt_(dt),
  peakRatio_(peakRatio), peakDuration_(peakDuration), maxBudget_((peakRatio * peakRatio - 1.0) * peakDuration)
{
  if(dt_ <= 0.0) { mc_rtc::log::error_and_throw("[TorqueOverloadFunction] dt must be strictly positive"); }
  if(peakRatio_ < 1.0) { mc_rtc::log::error_and_throw("[TorqueOverloadFunction] peakRatio must be >= 1"); }
  if(peakDuration_ <= 0.0)
  {
    mc_rtc::log::error_and_throw("[TorqueOverloadFunction] peakDuration must be strictly positive");
  }

  const auto & limits = robot.tvmRobot().limits();
  const auto & tl = limits.tl;
  const auto & tu = limits.tu;
  const Eigen::Index nDof = tau_->size();
  if(tl.size() != nDof || tu.size() != nDof)
  {
    mc_rtc::log::error_and_throw("[TorqueOverloadFunction] Torque limits size ({}, {}) does not match tau size ({})",
                                 tl.size(), tu.size(), nDof);
  }

  /* Only handle joints whose continuous rating is finite and non-zero. The floating base (tau = 0) and
   * unlimited joints are left to the regular box constraint. */
  const Eigen::Index fbSize = robot.tvmRobot().qFloatingBase()->size();
  for(Eigen::Index i = fbSize; i < nDof; ++i)
  {
    const double tauC = std::max(std::abs(tl(i)), std::abs(tu(i)));
    if(std::isfinite(tl(i)) && std::isfinite(tu(i)) && tauC > 1e-9) { indices_.push_back(i); }
  }

  const auto m = static_cast<Eigen::Index>(indices_.size());
  tl_.resize(m);
  tu_.resize(m);
  invTauC2_.resize(m);
  for(Eigen::Index k = 0; k < m; ++k)
  {
    const auto i = indices_[static_cast<size_t>(k)];
    tl_(k) = tl(i);
    tu_(k) = tu(i);
    const double tauC = std::max(std::abs(tl(i)), std::abs(tu(i)));
    invTauC2_(k) = 1.0 / (tauC * tauC);
  }

  resize(static_cast<int>(2 * m));

  registerUpdates(Update::B, &TorqueOverloadFunction::updateB);
  addOutputDependency<TorqueOverloadFunction>(Output::B, Update::B);
  addVariable(tau_, true);

  /* Constant Jacobian: [S; -S] */
  Eigen::MatrixXd J = Eigen::MatrixXd::Zero(2 * m, nDof);
  for(Eigen::Index k = 0; k < m; ++k)
  {
    const auto i = indices_[static_cast<size_t>(k)];
    J(k, i) = 1.0;
    J(m + k, i) = -1.0;
  }
  jacobian_[tau_.get()] = J;

  reset();
}

void TorqueOverloadFunction::reset()
{
  budget_.setZero(tl_.size());
  scale_.setConstant(tl_.size(), peakRatio_);
  initialized_ = false;
  computeBoundsFromBudget();
}

void TorqueOverloadFunction::updateB()
{
  const auto m = tl_.size();
  if(initialized_)
  {
    /* tau_ holds the torque obtained at the previous solver iteration */
    const auto & tau = tau_->value();
    for(Eigen::Index k = 0; k < m; ++k)
    {
      const double t = tau(indices_[static_cast<size_t>(k)]);
      const double e = budget_(k) + (t * t * invTauC2_(k) - 1.0) * dt_;
      budget_(k) = std::min(std::max(e, 0.0), maxBudget_);
    }
  }
  initialized_ = true;
  computeBoundsFromBudget();
}

void TorqueOverloadFunction::computeBoundsFromBudget()
{
  const auto m = tl_.size();
  scale_.resize(m);
  for(Eigen::Index k = 0; k < m; ++k)
  {
    /* Largest scale such that budget + (s^2 - 1) dt <= maxBudget */
    const double s2 = 1.0 + (maxBudget_ - budget_(k)) / dt_;
    scale_(k) = std::min(peakRatio_, std::sqrt(std::max(s2, 1.0)));
  }
  /* f = [S tau - s.*tu ; -S tau + s.*tl] <= 0 */
  b_.resize(2 * m);
  b_.head(m) = -scale_.cwiseProduct(tu_);
  b_.tail(m) = scale_.cwiseProduct(tl_);
}

} // namespace mc_tvm
