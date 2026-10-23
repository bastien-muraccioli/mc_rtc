/*
 * Copyright 2015-2022 CNRS-UM LIRMM, CNRS-AIST JRL
 */

#pragma once

#include <mc_solver/KinematicsConstraint.h>

#include <mc_tvm/DynamicFunction.h>
#include <mc_tvm/TorqueOverloadFunction.h>

#include <Tasks/QPMotionConstr.h>

#include <optional>

namespace mc_solver
{

/** Parameters of the time-limited torque overload (I²t thermal budget) used by DynamicsConstraint
 *
 * The robot model torque limits are interpreted as the continuous rating of each actuator. Each joint may go up
 * to \ref peakRatio times its limit, for at most \ref peakDuration seconds at that peak starting from a full budget
 * (any torque/time pair with the same I²t energy is allowed). See mc_tvm::TorqueOverloadFunction.
 */
struct TorqueOverloadLimits
{
  /** Maximum torque as a multiple of the model torque limit (hard cap) */
  double peakRatio = 2.0;
  /** Time the peak can be held from a full budget [s] */
  double peakDuration = 1.0;
};

/** \class DynamicsConstraint
 * Holds dynamics constraints (equation of motion) for a robot
 */

struct MC_SOLVER_DLLAPI DynamicsConstraint : public KinematicsConstraint
{
public:
  /** Constructor
   * Builds a regular joint limits constraint and a motion constr depending on
   * the nature of the robot
   * See tasks::qp::MotionConstr for details on the latter one
   * \param robots The robots including the robot affected by this constraint
   * \param robotIndex The index of the robot affected by this constraint
   * \param timeStep Solver timestep
   * \param infTorque If true, ignore the torque limits set in the robot model
   * \param compensateExtTorques If true, compensates external disturbances using a feedforward torque signal. The
   * constraint will search for the compensation value in robot by calling `compensationTorques()` method, if not an
   * estimation of external torques acting on the robot will be used by calling `externalTorques()` method.
   * \param real If true (TVM backend only), the equation of motion (H and C) is evaluated on the robot's real
   * counterpart instead of the control robot. Has no effect on the Tasks backend.
   */
  DynamicsConstraint(const mc_rbdyn::Robots & robots,
                     unsigned int robotIndex,
                     double timeStep,
                     bool infTorque = false,
                     bool compensateExtTorques = true,
                     bool real = false);

  /** Constructor
   * Builds a damped joint limits constraint and a motion constr depending on
   * the nature of the robot
   * See tasks::qp::MotionConstr for details on the latter one
   * \param robots The robots including the robot affected by this constraint
   * \param robotIndex The index of the robot affected by this constraint
   * \param timeStep Solver timestep
   * \param damper Value of the damper {interaction distance, safety distance,
   * offset}
   * \param velocityPercent Maximum joint velocity percentage, 0.5 is advised
   * \param infTorque If true, ignore the torque limits set in the robot model
   * \param compensateExtTorques If true, compensates external disturbances using a feedforward torque signal. The
   * constraint will search for the compensation value in robot by calling `compensationTorques()` method, if not an
   * estimation of external torques acting on the robot will be used by calling `externalTorques()` method.
   * \param real If true (TVM backend only), the equation of motion (H and C) is evaluated on the robot's real
   * counterpart instead of the control robot. Has no effect on the Tasks backend.
   */
  DynamicsConstraint(const mc_rbdyn::Robots & robots,
                     unsigned int robotIndex,
                     double timeStep,
                     const std::array<double, 3> & damper,
                     double velocityPercent = 1.0,
                     bool infTorque = false,
                     bool compensateExtTorques = true,
                     bool real = false);

  /** Constructor
   * Builds a CBF joint limits constraint and a motion constr depending on
   * the nature of the robot
   * See details of the CBF constraint in Safe Execution of RL Policies via Acceleration-based CBF-QP Constraint
   * Enforcement for Real-World Robotic Deployments, B.Muraccioli et al. (2026) See tasks::qp::MotionConstr for details
   * on the latter one
   * \param robots The robots including the robot affected by this constraint
   * \param robotIndex The index of the robot affected by this constraint
   * \param damperSecond Value of the damper {interaction distance, safety distance,
   * offset, amortization margin, lambda}
   * \param velocityPercent Maximum joint velocity percentage, 0.5 is advised
   * \param compensateExtTorques If true, external torques are added to the dynamic model constraint
   * \param activateConstraints If true, the constraint is activated
   * \param real If true (TVM backend only), the equation of motion (H and C) is evaluated on the robot's real
   * counterpart instead of the control robot. Has no effect on the Tasks backend.
   */
  DynamicsConstraint(const mc_rbdyn::Robots & robots,
                     unsigned int robotIndex,
                     const std::array<double, 5> & damperSecond,
                     double velocityPercent = 1.0,
                     bool compensateExtTorques = true,
                     bool activateConstraints = true,
                     bool real = false);

  /** Constructor
   * Builds a regular joint limits constraint and a motion constr where the torque limits are replaced by a
   * time-limited overload constraint (TVM backend only):
   * - hard bound: peakRatio * tl <= tau <= peakRatio * tu
   * - thermal bound: s(t) * tl <= tau <= s(t) * tu, with s(t) in [1, peakRatio] computed from an I²t budget written
   *   on the commanded torque (no current measurement needed), see mc_tvm::TorqueOverloadFunction
   * \param robots The robots including the robot affected by this constraint
   * \param robotIndex The index of the robot affected by this constraint
   * \param timeStep Solver timestep
   * \param overload Overload parameters (peak ratio and duration)
   * \param compensateExtTorques If true, external torques are added to the dynamic model constraint
   * \param real If true, the equation of motion (H and C) is evaluated on the robot's real counterpart
   */
  DynamicsConstraint(const mc_rbdyn::Robots & robots,
                     unsigned int robotIndex,
                     double timeStep,
                     const TorqueOverloadLimits & overload,
                     bool compensateExtTorques = true,
                     bool real = false);

  /** Constructor
   * Builds a CBF joint limits constraint (same as the damperSecond constructor) and a motion constr where the torque
   * limits are replaced by a time-limited overload constraint (TVM backend only), see the TorqueOverloadLimits
   * constructor for details on the torque part
   * \param robots The robots including the robot affected by this constraint
   * \param robotIndex The index of the robot affected by this constraint
   * \param damperSecond Value of the damper {interaction distance, safety distance,
   * offset, amortization margin, lambda}
   * \param overload Overload parameters (peak ratio and duration)
   * \param velocityPercent Maximum joint velocity percentage, 0.5 is advised
   * \param compensateExtTorques If true, external torques are added to the dynamic model constraint
   * \param activateConstraints If true, the constraint is activated (if false, the torque overload is disabled too)
   * \param real If true, the equation of motion (H and C) is evaluated on the robot's real counterpart
   */
  DynamicsConstraint(const mc_rbdyn::Robots & robots,
                     unsigned int robotIndex,
                     const std::array<double, 5> & damperSecond,
                     const TorqueOverloadLimits & overload,
                     double velocityPercent = 1.0,
                     bool compensateExtTorques = true,
                     bool activateConstraints = true,
                     bool real = false);

  /** \brief Update the constraint
   *
   * This is called at every iteration of the controller once the constraint has been added to a solver
   *
   * \param solver Solver in which the constraint has been inserted
   */
  void update(QPSolver & solver) override;

  /** Returns the tasks::qp::MotionConstr
   *
   * This assumes the backend was Tasks
   */
  inline tasks::qp::MotionConstr & motionConstr() noexcept
  {
    assert(backend_ == QPSolver::Backend::Tasks);
    return *static_cast<tasks::qp::MotionConstr *>(motion_constr_.get());
  }

  /** Returns the mc_tvm::DynamicFunction
   *
   * Assumes the backend was TVM
   */
  inline mc_tvm::DynamicFunction & dynamicFunction()
  {
    assert(backend_ == QPSolver::Backend::TVM);
    return *(static_cast<mc_tvm::DynamicFunctionPtr *>(motion_constr_.get())->get());
  }

  /** Returns the torque overload function, nullptr if this constraint does not use the overload mode or has not
   * been added to a solver yet (TVM backend only) */
  inline const mc_tvm::TorqueOverloadFunctionPtr & torqueOverloadFunction() const noexcept { return overload_fn_; }

  /** True if the constraint was built with the time-limited torque overload */
  inline bool hasTorqueOverload() const noexcept { return overload_.has_value(); }

  void addToSolverImpl(QPSolver & solver) override;

  void removeFromSolverImpl(QPSolver & solver) override;

  inline unsigned int robotIndex() const noexcept { return robotIndex_; }

  /** Whether this constraint evaluates the equation of motion on the real robot (TVM backend only) */
  inline bool real() const noexcept { return real_; }

protected:
  /** Holds the motion constraint implementation
   *
   * In Tasks backend:
   * - tasks::qp::MotionConstr or a derived constraint if the robot has flexibilities
   *
   * In TVM backend:
   * - mc_tvm::DynamicFunctionPtr
   */
  mc_rtc::void_ptr motion_constr_;
  /** Robot index for the constraint */
  unsigned int robotIndex_;
  bool activateConstraints_ = true;
  /** If true (TVM backend only), the equation of motion is evaluated on the robot's real counterpart */
  bool real_ = false;
  /** Overload parameters, set only by the overload constructor */
  std::optional<TorqueOverloadLimits> overload_;
  /** Overload function (TVM backend), created on the first addToSolver and kept so the budget survives re-adds */
  mc_tvm::TorqueOverloadFunctionPtr overload_fn_;

private:
  void addLogging(QPSolver & solver);
};

} // namespace mc_solver
