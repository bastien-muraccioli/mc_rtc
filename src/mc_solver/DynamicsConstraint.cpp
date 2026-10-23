/*
 * Copyright 2015-2022 CNRS-UM LIRMM, CNRS-AIST JRL
 */

#include <mc_solver/DynamicsConstraint.h>

#include <mc_solver/ConstraintSetLoader.h>
#include <mc_solver/TasksQPSolver.h>

#include <mc_tvm/DerivativeFunction.h>
#include <mc_tvm/DynamicFunction.h>
#include <mc_tvm/TorqueOverloadFunction.h>

#include <Tasks/Bounds.h>

#include "TVMKinematicsConstraint.h"

#include <mc_rtc/log/Logger.h>

namespace mc_solver
{

static mc_rtc::void_ptr initialize_tasks(const mc_rbdyn::Robots & robots,
                                         unsigned int robotIndex,
                                         double timeStep,
                                         bool infTorque,
                                         bool compensateExtTorques)
{
  const auto & robot = robots.robot(robotIndex);
  std::vector<std::vector<double>> tl = robot.tl();
  std::vector<std::vector<double>> tu = robot.tu();
  std::vector<std::vector<double>> tdl = robot.tdl();
  std::vector<std::vector<double>> tdu = robot.tdu();
  if(infTorque)
  {
    for(auto & ti : tl)
    {
      for(auto & t : ti) { t = -INFINITY; }
    }
    for(auto & ti : tu)
    {
      for(auto & t : ti) { t = INFINITY; }
    }
    for(auto & tdi : tdl)
    {
      for(auto & td : tdi) { td = -INFINITY; }
    }
    for(auto & tdi : tdu)
    {
      for(auto & td : tdi) { td = INFINITY; }
    }
  }
  tasks::TorqueBound tBound(tl, tu);
  tasks::TorqueDBound tDBound(tdl, tdu);
  if(robot.flexibility().size() != 0)
  {
    std::vector<tasks::qp::SpringJoint> sjList;
    for(const auto & flex : robot.flexibility())
    {
      sjList.push_back(tasks::qp::SpringJoint(flex.jointName, flex.K, flex.C, flex.O));
    }
    if(compensateExtTorques)
    {
      if(robot.compensationTorques())
      {
        return mc_rtc::make_void_ptr<tasks::qp::MotionSpringConstr>(robots.mbs(), static_cast<int>(robotIndex), tBound,
                                                                    tDBound, timeStep, sjList,
                                                                    robot.compensationTorques().value());
      }
      else
      {

        return mc_rtc::make_void_ptr<tasks::qp::MotionSpringConstr>(robots.mbs(), static_cast<int>(robotIndex), tBound,
                                                                    tDBound, timeStep, sjList, robot.externalTorques());
      }
    }
    else
    {
      return mc_rtc::make_void_ptr<tasks::qp::MotionSpringConstr>(robots.mbs(), static_cast<int>(robotIndex), tBound,
                                                                  tDBound, timeStep, sjList);
    }
  }
  else
  {
    if(compensateExtTorques)
    {
      if(robot.compensationTorques())
      {
        return mc_rtc::make_void_ptr<tasks::qp::MotionConstr>(robots.mbs(), static_cast<int>(robotIndex), tBound,
                                                              tDBound, timeStep, robot.compensationTorques().value());
      }
      else
      {
        return mc_rtc::make_void_ptr<tasks::qp::MotionConstr>(robots.mbs(), static_cast<int>(robotIndex), tBound,
                                                              tDBound, timeStep, robot.externalTorques());
      }
    }
    else
    {
      return mc_rtc::make_void_ptr<tasks::qp::MotionConstr>(robots.mbs(), static_cast<int>(robotIndex), tBound, tDBound,
                                                            timeStep);
    }
  }
}

mc_rtc::void_ptr initialize_tvm(const mc_rbdyn::Robot & robot, bool compensateExtTorques, bool real)
{
  return mc_rtc::make_void_ptr<mc_tvm::DynamicFunctionPtr>(
      std::make_shared<mc_tvm::DynamicFunction>(robot, compensateExtTorques, real));
}

static mc_rtc::void_ptr initialize(QPSolver::Backend backend,
                                   const mc_rbdyn::Robots & robots,
                                   unsigned int robotIndex,
                                   double timeStep,
                                   bool infTorque,
                                   bool compensateExtTorques,
                                   bool real)
{
  switch(backend)
  {
    case QPSolver::Backend::Tasks:
      return initialize_tasks(robots, robotIndex, timeStep, infTorque, compensateExtTorques);
    case QPSolver::Backend::TVM:
      return initialize_tvm(robots.robot(robotIndex), compensateExtTorques, real);
    default:
      mc_rtc::log::error_and_throw("[DynamicsConstraint] Not implemented for solver backend: {}", backend);
  }
}

static mc_rtc::void_ptr initialize(QPSolver::Backend backend,
                                   const mc_rbdyn::Robots & robots,
                                   unsigned int robotIndex,
                                   bool compensateExtTorques,
                                   bool real)
{
  switch(backend)
  {
    case QPSolver::Backend::TVM:
      return initialize_tvm(robots.robot(robotIndex), compensateExtTorques, real);
    default:
      mc_rtc::log::error_and_throw("[DynamicsConstraint] Not implemented for solver backend: {}", backend);
  }
}

DynamicsConstraint::DynamicsConstraint(const mc_rbdyn::Robots & robots,
                                       unsigned int robotIndex,
                                       double timeStep,
                                       bool infTorque,
                                       bool compensateExtTorques,
                                       bool real)
: KinematicsConstraint(robots, robotIndex, timeStep),
  motion_constr_(initialize(backend_, robots, robotIndex, timeStep, infTorque, compensateExtTorques, real)),
  robotIndex_(robotIndex), real_(real)
{
}

DynamicsConstraint::DynamicsConstraint(const mc_rbdyn::Robots & robots,
                                       unsigned int robotIndex,
                                       double timeStep,
                                       const std::array<double, 3> & damper,
                                       double velocityPercent,
                                       bool infTorque,
                                       bool compensateExtTorques,
                                       bool real)
: KinematicsConstraint(robots, robotIndex, timeStep, damper, velocityPercent),
  motion_constr_(initialize(backend_, robots, robotIndex, timeStep, infTorque, compensateExtTorques, real)),
  robotIndex_(robotIndex), real_(real)
{
}

DynamicsConstraint::DynamicsConstraint(const mc_rbdyn::Robots & robots,
                                       unsigned int robotIndex,
                                       const std::array<double, 5> & damperSecond,
                                       double velocityPercent,
                                       bool compensateExtTorques,
                                       bool activateConstraints,
                                       bool real)
: KinematicsConstraint(robots, robotIndex, damperSecond, velocityPercent, activateConstraints),
  motion_constr_(initialize(backend_, robots, robotIndex, compensateExtTorques, real)), robotIndex_(robotIndex),
  activateConstraints_(activateConstraints), real_(real)
{
}

static void checkOverload(const TorqueOverloadLimits & overload)
{
  if(overload.peakRatio < 1.0)
  {
    mc_rtc::log::error_and_throw("[DynamicsConstraint] TorqueOverloadLimits::peakRatio must be >= 1 (got {})",
                                 overload.peakRatio);
  }
  if(overload.peakDuration <= 0.0)
  {
    mc_rtc::log::error_and_throw("[DynamicsConstraint] TorqueOverloadLimits::peakDuration must be > 0 (got {})",
                                 overload.peakDuration);
  }
}

DynamicsConstraint::DynamicsConstraint(const mc_rbdyn::Robots & robots,
                                       unsigned int robotIndex,
                                       double timeStep,
                                       const TorqueOverloadLimits & overload,
                                       bool compensateExtTorques,
                                       bool real)
: KinematicsConstraint(robots, robotIndex, timeStep),
  motion_constr_(initialize(backend_, robots, robotIndex, compensateExtTorques, real)), robotIndex_(robotIndex),
  real_(real), overload_(overload)
{
  checkOverload(overload);
}

DynamicsConstraint::DynamicsConstraint(const mc_rbdyn::Robots & robots,
                                       unsigned int robotIndex,
                                       const std::array<double, 5> & damperSecond,
                                       const TorqueOverloadLimits & overload,
                                       double velocityPercent,
                                       bool compensateExtTorques,
                                       bool activateConstraints,
                                       bool real)
: KinematicsConstraint(robots, robotIndex, damperSecond, velocityPercent, activateConstraints),
  motion_constr_(initialize(backend_, robots, robotIndex, compensateExtTorques, real)), robotIndex_(robotIndex),
  activateConstraints_(activateConstraints), real_(real), overload_(overload)
{
  checkOverload(overload);
}

void DynamicsConstraint::update(QPSolver & solver)
{
  if(backend_ == QPSolver::Backend::Tasks)
  {
    auto & robot = solver.robot(robotIndex_);
    if(robot.compensationTorques())
    {
      static_cast<tasks::qp::MotionConstr *>(motion_constr_.get())
          ->setExternalTorques(robot.compensationTorques().value());
    }
    else
    {
      static_cast<tasks::qp::MotionConstr *>(motion_constr_.get())->setExternalTorques(robot.externalTorques());
    }
  }
}

void DynamicsConstraint::addToSolverImpl(QPSolver & solver)
{
  KinematicsConstraint::addToSolverImpl(solver);
  switch(backend_)
  {
    case QPSolver::Backend::Tasks:
      static_cast<tasks::qp::MotionConstr *>(motion_constr_.get())
          ->addToSolver(solver.robots().mbs(), static_cast<TasksQPSolver &>(solver).solver());
      break;
    case QPSolver::Backend::TVM:
    {
      auto & constraints_ = static_cast<TVMKinematicsConstraint *>(constraint_.get())->constraints_;
      auto & problem = tvm_solver(solver).problem();
      auto & tvm_robot = solver.robot(robotIndex_).tvmRobot();

      // Add the torque bounds
      auto tl_lim = tvm_robot.limits().tl;
      auto tu_lim = tvm_robot.limits().tu;
      // Add the torque derivative bounds
      auto tdl_lim = tvm_robot.limits().tdl;
      auto tdu_lim = tvm_robot.limits().tdu;

      if(!activateConstraints_)
      {
        // Keep 0 at the floating base and set the joint limits to infinity to effectively disable the constraints
        tl_lim.head(tvm_robot.qFloatingBase()->size()).setZero();
        tu_lim.head(tvm_robot.qFloatingBase()->size()).setZero();
        tl_lim.tail(tvm_robot.qJoints()->size()).setConstant(-INFINITY);
        tu_lim.tail(tvm_robot.qJoints()->size()).setConstant(INFINITY);

        tdl_lim.head(tvm_robot.qFloatingBase()->size()).setZero();
        tdu_lim.head(tvm_robot.qFloatingBase()->size()).setZero();
        tdl_lim.tail(tvm_robot.qJoints()->size()).setConstant(-INFINITY);
        tdu_lim.tail(tvm_robot.qJoints()->size()).setConstant(INFINITY);
      }

      if(overload_ && activateConstraints_)
      {
        /* Hard cap: the actuator can never exceed peakRatio * limit (protects magnets, gearbox and driver).
         * The floating base stays at 0 and infinite limits stay infinite. */
        const auto nJ = tvm_robot.qJoints()->size();
        tl_lim.tail(nJ) *= overload_->peakRatio;
        tu_lim.tail(nJ) *= overload_->peakRatio;
      }

      auto tL = problem.add(tl_lim <= tvm_robot.tau() <= tu_lim, tvm::task_dynamics::None(),
                            {tvm::requirements::PriorityLevel(0)});
      constraints_.push_back(tL);

      if(overload_ && activateConstraints_)
      {
        /* Time-limited bound: s(t) * tl <= tau <= s(t) * tu with s(t) from the I²t budget */
        if(!overload_fn_)
        {
          overload_fn_ = std::make_shared<mc_tvm::TorqueOverloadFunction>(
              solver.robot(robotIndex_), solver.dt(), overload_->peakRatio, overload_->peakDuration);
        }
        if(overload_fn_->size() > 0)
        {
          auto tO = problem.add(overload_fn_ <= 0., tvm::task_dynamics::None(), {tvm::requirements::PriorityLevel(0)});
          constraints_.push_back(tO);
        }
      }

      /** Torque derivative limits */
      mc_tvm::DerivativeFunctionPtr taud_fn =
          std::make_shared<mc_tvm::DerivativeFunction>(solver.robot(robotIndex_), solver.dt(), tvm_robot.tau());
      auto tDL =
          problem.add(tdl_lim <= taud_fn <= tdu_lim, tvm::task_dynamics::None(), {tvm::requirements::PriorityLevel(0)});
      constraints_.push_back(tDL);

      mc_tvm::DynamicFunctionPtr dyn_fn = *static_cast<mc_tvm::DynamicFunctionPtr *>(motion_constr_.get());
      auto dyn = problem.add(dyn_fn == 0., tvm::task_dynamics::None(), {tvm::requirements::PriorityLevel(0)});
      constraints_.push_back(dyn);
      auto cstr = problem.constraint(*dyn);
      problem.add(tvm::hint::Substitution(cstr, tvm_robot.tau()));
      break;
    }
    default:
      break;
  }
  addLogging(solver);
}

void DynamicsConstraint::removeFromSolverImpl(QPSolver & solver)
{
  switch(backend_)
  {
    case QPSolver::Backend::Tasks:
    {
      KinematicsConstraint::removeFromSolverImpl(solver);
      static_cast<tasks::qp::MotionConstr *>(motion_constr_.get())
          ->removeFromSolver(static_cast<TasksQPSolver &>(solver).solver());
      break;
    }
    case QPSolver::Backend::TVM:
    {
      // Remove the log entry before destroying the constraint
      if(solver.logger())
      {
        solver.logger()->removeLogEntry("DynamicsConstraint_contactTorque");
        if(overload_fn_)
        {
          const auto prefix = "DynamicsConstraint_" + solver.robot(robotIndex_).name() + "_torqueOverload_";
          solver.logger()->removeLogEntry(prefix + "remainingBudget");
          solver.logger()->removeLogEntry(prefix + "scale");
          solver.logger()->removeLogEntry(prefix + "upperBound");
          solver.logger()->removeLogEntry(prefix + "lowerBound");
        }
      }
      auto & constr = *static_cast<TVMKinematicsConstraint *>(constraint_.get());
      auto & problem = tvm_solver(solver).problem();
      problem.removeSubstitutionFor(*problem.constraint(*constr.constraints_.back()));
      KinematicsConstraint::removeFromSolverImpl(solver);
      break;
    }
    default:
      break;
  }
}

void DynamicsConstraint::addLogging(QPSolver & solver)
{
  auto & logger = *solver.logger();
  switch(backend_)
  {
    case QPSolver::Backend::TVM:
    {
      mc_tvm::DynamicFunctionPtr fn_ptr = *static_cast<mc_tvm::DynamicFunctionPtr *>(motion_constr_.get());
      logger.addLogEntry("DynamicsConstraint_contactTorque", [fn_ptr]() { return fn_ptr->contactTorque(); });
      if(overload_fn_)
      {
        /* Values are given for the limited joints only, in the order of overload_fn_->indices() */
        const auto prefix = "DynamicsConstraint_" + solver.robot(robotIndex_).name() + "_torqueOverload_";
        auto ov = overload_fn_;
        logger.addLogEntry(prefix + "remainingBudget", [ov]() { return ov->remainingBudget(); });
        logger.addLogEntry(prefix + "scale", [ov]() -> const Eigen::VectorXd & { return ov->scale(); });
        logger.addLogEntry(prefix + "upperBound", [ov]() { return ov->upperBound(); });
        logger.addLogEntry(prefix + "lowerBound", [ov]() { return ov->lowerBound(); });
      }
      break;
    }
    default:
      break;
  }
}

} // namespace mc_solver

namespace
{

mc_solver::ConstraintSetPtr load_kin_constr(mc_solver::QPSolver & solver, const mc_rtc::Configuration & config)
{
  const auto robotIndex = robotIndexFromConfig(config, solver.robots(), "kinematics");
  if(config.has("damper"))
  {
    return std::make_shared<mc_solver::KinematicsConstraint>(solver.robots(), robotIndex, solver.dt(), config("damper"),
                                                             config("velocityPercent", 0.5));
  }
  else
  {
    return std::make_shared<mc_solver::KinematicsConstraint>(solver.robots(), robotIndex, solver.dt());
  }
}

mc_solver::ConstraintSetPtr load_dyn_constr(mc_solver::QPSolver & solver, const mc_rtc::Configuration & config)
{
  const auto robotIndex = robotIndexFromConfig(config, solver.robots(), "dynamics");
  if(config.has("torqueOverload"))
  {
    /* torqueOverload: { peakRatio: 2.0, peakDuration: 1.0 } */
    const auto & c = config("torqueOverload");
    mc_solver::TorqueOverloadLimits overload;
    c("peakRatio", overload.peakRatio);
    c("peakDuration", overload.peakDuration);
    if(config.has("damperSecond"))
    {
      const std::array<double, 5> damperSecond = config("damperSecond");
      return std::make_shared<mc_solver::DynamicsConstraint>(
          solver.robots(), robotIndex, damperSecond, overload, config("velocityPercent", 1.0),
          config("compensateExtTorques", true), config("activateConstraints", true), config("real", false));
    }
    return std::make_shared<mc_solver::DynamicsConstraint>(solver.robots(), robotIndex, solver.dt(), overload,
                                                           config("compensateExtTorques", true), config("real", false));
  }
  if(config.has("damper"))
  {
    return std::make_shared<mc_solver::DynamicsConstraint>(solver.robots(), robotIndex, solver.dt(), config("damper"),
                                                           config("velocityPercent", 0.5), config("infTorque", false));
  }
  else
  {
    return std::make_shared<mc_solver::DynamicsConstraint>(solver.robots(), robotIndex, solver.dt(),
                                                           config("infTorque", false));
  }
}

static auto kin_registered = mc_solver::ConstraintSetLoader::register_load_function("kinematics", &load_kin_constr);
static auto dyn_registered = mc_solver::ConstraintSetLoader::register_load_function("dynamics", &load_dyn_constr);

} // namespace
