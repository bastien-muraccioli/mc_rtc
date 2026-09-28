/*
 * Copyright 2015-2019 CNRS-UM LIRMM, CNRS-AIST JRL
 *
 * This file is  inspired by Stephane's Caron implementation as part of
 * lipm_walking_controller <https://github.com/stephane-caron/lipm_walking_controller>
 */

#pragma once

#include <mc_rtc/logging.h>
#include <Eigen/Core>
#include <algorithm>
#include <type_traits>

namespace mc_filter
{

/** Low-pass filter from series of velocity measurements.
 *
 * Expects T to have:
 * - T::Zero() static method (e.g Eigen::Vector3d, etc), or be an arithmetic or dynamic Eigen type
 */
template<typename T>
struct LowPass
{
  /** Constructor with cutoff period.
   *
   * \param dt Sampling period.
   *
   * \param period Cutoff period.
   *
   */
  LowPass(double dt, double period = 0) : cutoffPeriod_(period), dt_(dt)
  {
    if constexpr(std::is_base_of_v<Eigen::DenseBase<T>, T>)
    {
      if constexpr(T::SizeAtCompileTime != Eigen::Dynamic)
      {
        reset(T::Zero());
      }
    }
    else if constexpr(std::is_arithmetic_v<T>)
    {
      reset(static_cast<T>(0));
    }
    else
    {
      reset(T::Zero());
    }
  }

  /** Constructor with cutoff period and initial value.
   *
   * \param dt Sampling period.
   *
   * \param period Cutoff period.
   *
   * \param initialValue Initial value for the filter output.
   *
   */
  LowPass(double dt, double period, const T & initialValue) : cutoffPeriod_(period), dt_(dt)
  {
    reset(initialValue);
  }

  /** Get cutoff period. */
  double cutoffPeriod() const { return cutoffPeriod_; }

  /** Set cutoff period.
   *
   * \param period New cutoff period.
   *
   * \note period is explicitely enforced to respect the Nyquist–Shannon sampling theorem, that is T is at least
   * 2*timestep.
   */
  void cutoffPeriod(double period)
  {
    if(period < 2 * dt_)
    {
      mc_rtc::log::warning("Time constant must be at least twice the timestep (Nyquist–Shannon sampling theorem)");
      period = 2 * dt_;
    }
    cutoffPeriod_ = period;
  }

  /** Reset position to an initial rest value.
   *
   * \param pos New position.
   *
   */
  void reset(const T & value) { eval_ = value; }

  /** Update velocity estimate from new position value.
   *
   * \param newPos New observed position.
   *
   */
  void update(const T & newValue)
  {
    if constexpr(std::is_base_of_v<Eigen::DenseBase<T>, T>)
    {
      if constexpr(T::SizeAtCompileTime == Eigen::Dynamic)
      {
        if(eval_.size() == 0)
        {
          eval_ = newValue;
          return;
        }
      }
    }
    double x = (cutoffPeriod_ <= dt_) ? 1. : dt_ / cutoffPeriod_;
    eval_ = x * newValue + (1. - x) * eval_;
  }

  /** Get filtered velocity.
   *
   */
  const T & eval() const { return eval_; }

  /** Get sampling period.
   *
   */
  double dt() const { return dt_; }

  /** Set sampling period.
   *
   * \param dt Sampling period.
   *
   * \note the cutoff period is updated to satisfy the Nyquist–Shannon sampling theorem according the new sampling
   * period.
   */
  void dt(double dt)
  {
    dt_ = dt;
    cutoffPeriod(cutoffPeriod_);
  }

private:
  T eval_;
  double cutoffPeriod_ = 0.;

protected:
  double dt_ = 0.005; // [s]
};

} // namespace mc_filter
