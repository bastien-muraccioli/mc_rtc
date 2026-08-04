#include <mc_tasks/MetaTask.h>
#include <mc_tasks/OperationalSpaceTask.h>
#include <mc_tvm/WrenchFunction.h>

#include <mc_rtc/gui/ArrayInput.h>
#include <mc_rtc/gui/ArrayLabel.h>
#include <mc_rtc/gui/Checkbox.h>
#include <mc_rtc/gui/NumberInput.h>
#include <mc_rtc/gui/NumberSlider.h>
#include <mc_rtc/gui/Transform.h>
#include <mc_tvm/Robot.h>
#include <mc_tvm/RobotFrame.h>
#include <SpaceVecAlg/EigenTypedef.h>
#include <SpaceVecAlg/SpaceVecAlg>
#include "mc_rtc/logging.h"
#include "mc_tasks/WrenchTask.h"
#include <Eigen/src/Core/Matrix.h>
#include <vector>

namespace mc_tasks
{

OperationalSpaceTask::OperationalSpaceTask(const mc_solver::QPSolver & solver,
                                           const mc_rbdyn::RobotFrame & frame,
                                           double weight)
: WrenchTask(frame, weight), posTarget_(sva::PTransformd::Identity()), velTarget_(sva::MotionVecd::Zero()),
  robots_(solver.robots()), rIndex_(frame.robot().robotIndex()),
  nbActuatedJoints(
      (robots_.robot(rIndex_).mb().nrJoints() > 0 && robots_.robot(rIndex_).mb().joint(0).type() == rbd::Joint::Free)
          ? robots_.robot(rIndex_).mb().nrDof() - 6
          : robots_.robot(rIndex_).mb().nrDof()),
  stiffness_(sva::MotionVecd::Zero()), damping_(sva::MotionVecd::Zero()), integralGain_(sva::MotionVecd::Zero()),
  maxIntegralWrench_(sva::ForceVecd::Zero()), integralWrench_(sva::ForceVecd::Zero()),
  accelerationFeedforward_(sva::MotionVecd::Zero()), externalWrench_(sva::ForceVecd::Zero()),
  gravityWrench_(sva::ForceVecd::Zero()), posError_(sva::MotionVecd::Zero()), velError_(sva::MotionVecd::Zero()),
  integralError_(sva::MotionVecd::Zero()), wrenchTarget_(sva::ForceVecd::Zero()),
  prevPosTarget_(sva::PTransformd::Identity()), frameJac_(robots_.robot(rIndex_).mb(), frame.body()),
  jacMat_(6, robots_.robot(rIndex_).mb().nrDof()), dynamicJacTransposeLocal_(6, robots_.robot(rIndex_).mb().nrDof()),
  cartesianInertiaLocal_(Eigen::Matrix6d::Zero()), shortJacMat_(6, frameJac_.dof())
{
  if(backend_ == Backend::Tasks)
    mc_rtc::log::error_and_throw<std::runtime_error>(
        "[mc_tasks] Can't use OperationalSpaceTask with {} backend, please use TVM backend", backend_);

  name_ = "operational_space_" + frame.name() + "_" + solver.robots().robot(frame.robot().robotIndex()).name();
  type_ = "operational_space";
  setStiffnessTranslational(10.0);
  setStiffnessRotational(10.0);
  setDampingTranslational(5.0);
  setDampingRotational(5.0);
  setIntegralGainTranslational(0.0);
  setIntegralGainRotational(0.0);
  setMaxIntegralWrench(5, 10); // Anti-windup limits for the integral term
  enableIntegralTerm(false); // Integral term disabled by default
  reset();
}

OperationalSpaceTask::OperationalSpaceTask(const mc_solver::QPSolver & solver,
                                           const std::string & bodyName,
                                           unsigned int rIndex,
                                           double weight)
: OperationalSpaceTask(solver, solver.robots().robot(rIndex).frame(bodyName), weight)
{
}

void OperationalSpaceTask::reset()
{
  posTarget_ = frame().position();
  prevPosTarget_ = posTarget_;
  velTarget_.Zero();
  accelerationFeedforward_.Zero();
  integralError_.Zero();
  integralWrench_.Zero();
  WrenchTask::reset();
}

void OperationalSpaceTask::update(mc_solver::QPSolver & solver)
{
  wrenchTarget_ = sva::ForceVecd::Zero();

  computeDynamicJacobian();

  // Error in world frame — frame().position()/velocity() are plain mc_rbdyn
  posError_ = sva::transformVelocity(frame().position() * posTarget_.inv());
  velError_ = velTarget_ - frame_->velocity();

  Eigen::Vector6d pd = stiffness_.vector().cwiseProduct(posError_.vector())
                       + damping_.vector().cwiseProduct(velError_.vector()) + accelerationFeedforward_.vector();
  sva::ForceVecd pdWrench(cartesianInertiaLocal_ * pd);
  wrenchTarget_ += pdWrench;

  computeForceGravityCompensation();
  if(compensateGravity_) { wrenchTarget_ += gravityWrench_; }

  mc_rtc::log::info("[OSC] posError={} velError={} pd={} pdWrench={} wrenchTarget={}", posError_.vector().transpose(),
                    velError_.vector().transpose(), pd.transpose(), pdWrench.vector().transpose(),
                    wrenchTarget_.vector().transpose());

  WrenchTask::targetWrench(wrenchTarget_);
}

void OperationalSpaceTask::computeDynamicJacobian()
{
  auto & robot = robots_.robot(rIndex_);

  // Task Jacobian at the frame's actual point, fresh from the current mbc
  shortJacMat_ = frameJac_.jacobian(robot.mb(), robot.mbc(), frame().position());
  frameJac_.fullJacobian(robot.mb(), shortJacMat_, jacMat_);

  rbd::ForwardDynamics fd(robot.mb());
  fd.computeH(robot.mb(), robot.mbc());
  const Eigen::MatrixXd & H = fd.H();

  Eigen::LDLT<Eigen::MatrixXd> H_ldlt(H);
  Eigen::MatrixXd MinvJt = H_ldlt.solve(jacMat_.transpose()); // M^{-1} J^T

  Eigen::Matrix6d JMinvJT = jacMat_ * MinvJt; // J M^{-1} J^T
  cartesianInertiaLocal_ = JMinvJT.ldlt().solve(Eigen::Matrix6d::Identity()); // Λ

  Eigen::MatrixXd JMInv = MinvJt.transpose(); // J M^{-1}
  dynamicJacTransposeLocal_ = cartesianInertiaLocal_ * JMInv; // (J^#)^T = Λ J M^{-1}

  mc_rtc::log::info("[OSC] frame={} cartesianInertiaLocal_ hasNaN={} \n{}", frame().name(),
                    !cartesianInertiaLocal_.allFinite(), cartesianInertiaLocal_);
}

void OperationalSpaceTask::computeForceGravityCompensation()
{
  auto & robot = robots_.robot(rIndex_);

  rbd::ForwardDynamics fd(robot.mb());
  fd.computeH(robot.mb(), robot.mbc());
  fd.computeC(robot.mb(), robot.mbc());
  const Eigen::VectorXd & Cg = fd.C();

  sva::MotionVecd jdotAlpha =
      frameJac_.normalAcceleration(robot.mb(), robot.mbc(), frame().X_b_f(), frame().velocity());

  Eigen::Vector6d forceGravityVec = dynamicJacTransposeLocal_ * Cg + cartesianInertiaLocal_ * jdotAlpha.vector();
  gravityWrench_ = sva::ForceVecd(forceGravityVec);
}

void OperationalSpaceTask::setDeriveVelocityTargetFromPosition(bool compute)
{
  deriveVelocityTargetFromPosition_ = compute;
  if(compute)
    prevPosTarget_ = posTarget_; // Reset previous position target to avoid large velocity target on the first update
}

void OperationalSpaceTask::enableIntegralTerm(bool enable)
{
  integralTermEnabled_ = enable;
  if(!enable)
  {
    integralError_.Zero();
    integralWrench_.Zero();
  }
}

void OperationalSpaceTask::addToGUI(mc_rtc::gui::StateBuilder & gui)
{
  gui.addElement({"Tasks", name_, "Gains"}, mc_rtc::gui::ArrayInput("Stiffness", stiffness_),
                 mc_rtc::gui::ArrayInput("Damping", damping_),
                 mc_rtc::gui::NumberInput(
                     "Constant Stiffness Translational", [this]() { return stiffness_.linear()[0]; },
                     [this](const double & s) { setStiffnessTranslational(s); }),
                 mc_rtc::gui::NumberInput(
                     "Constant Damping Translational", [this]() { return damping_.linear()[0]; },
                     [this](const double & d) { setDampingTranslational(d); }),
                 mc_rtc::gui::NumberInput(
                     "Constant Stiffness Rotational", [this]() { return stiffness_.angular()[0]; },
                     [this](const double & s) { setStiffnessRotational(s); }),
                 mc_rtc::gui::NumberInput(
                     "Constant Damping Rotational", [this]() { return damping_.angular()[0]; },
                     [this](const double & d) { setDampingRotational(d); }),
                 mc_rtc::gui::Checkbox(
                     "Enable Integral Term", [this]() { return integralTermEnabled(); },
                     [this]() { enableIntegralTerm(!integralTermEnabled()); }),
                 mc_rtc::gui::Checkbox(
                     "Derive Velocity Target from Position Target",
                     [this]() { return deriveVelocityTargetFromPosition(); },
                     [this]() { setDeriveVelocityTargetFromPosition(!deriveVelocityTargetFromPosition()); }));

  gui.addElement({"Tasks", name_, "Gains", "Integral Term"}, mc_rtc::gui::ArrayInput("Integral Gain", integralGain_),
                 mc_rtc::gui::ArrayInput("Max Integral Wrench", maxIntegralWrench_),
                 mc_rtc::gui::NumberInput(
                     "Constant Integral Gain Translational", [this]() { return integralGain_.linear()[0]; },
                     [this](const double & g) { setIntegralGainTranslational(g); }),
                 mc_rtc::gui::NumberInput(
                     "Constant Integral Gain Rotational", [this]() { return integralGain_.angular()[0]; },
                     [this](const double & g) { setIntegralGainRotational(g); }),
                 mc_rtc::gui::NumberInput(
                     "Constant Max Integral Force", [this]() { return maxIntegralWrench_.force()[0]; },
                     [this](const double & w) { setMaxIntegralForce(w); }),
                 mc_rtc::gui::NumberInput(
                     "Constant Max Integral Torque", [this]() { return maxIntegralWrench_.couple()[0]; },
                     [this](const double & w) { setMaxIntegralTorque(w); }));

  gui.addElement({"Tasks", name_, "Additional Forces"},
                 mc_rtc::gui::Checkbox(
                     "Compensate External Forces", [this]() { return isCompensatingExternalWrench(); },
                     [this]() { setCompensateExternalWrench(!isCompensatingExternalWrench()); }),
                 mc_rtc::gui::Checkbox(
                     "Compensate Gravity (+ Coriolis)", [this]() { return isCompensatingGravity(); },
                     [this]() { setCompensateGravity(!isCompensatingGravity()); }),
                 mc_rtc::gui::ArrayLabel("External Wrench", externalWrench_),
                 mc_rtc::gui::ArrayLabel("Gravity Wrench", gravityWrench_));

  gui.addElement({"Tasks", name_, "Details"}, mc_rtc::gui::ArrayLabel("Position Error", posError_),
                 mc_rtc::gui::ArrayLabel("Velocity Error", velError_));

  gui.addElement({"Tasks", name_, "Position Target"},
                 mc_rtc::gui::Transform(
                     "pos_target", [this]() { return this->posTarget(); },
                     [this](const sva::PTransformd & X_0_target) { this->setPosTarget(X_0_target); }),
                 mc_rtc::gui::Transform("pos", [this]() { return this->frame().position(); }));

  gui.addElement({"Tasks", name_, "Velocity Target"}, mc_rtc::gui::ArrayInput("vel_target", velTarget_),
                 mc_rtc::gui::ArrayLabel("vel", [this]() { return this->frame().velocity(); }));

  WrenchTask::addToGUI(gui);
}

void OperationalSpaceTask::addToLogger(mc_rtc::Logger & logger)
{
  WrenchTask::addToLogger(logger);
  logger.addLogEntry(name_ + "_stiffness", [this]() { return stiffness_; });
  logger.addLogEntry(name_ + "_damping", [this]() { return damping_; });
  logger.addLogEntry(name_ + "_posTarget", [this]() { return posTarget_; });
  logger.addLogEntry(name_ + "_velTarget", [this]() { return velTarget_; });
  logger.addLogEntry(name_ + "_torqueFeedforward", [this]() { return accelerationFeedforward_; });
  logger.addLogEntry(name_ + "_posError", [this]() { return posError_; });
  logger.addLogEntry(name_ + "_velError", [this]() { return velError_; });
  logger.addLogEntry(name_ + "_deriveVelocityTargetFromPosition",
                     [this]() { return deriveVelocityTargetFromPosition_; });
  logger.addLogEntry(name_ + "_compensateExternalForces", [this]() { return compensateExternalWrench_; });
  logger.addLogEntry(name_ + "_gravityWrenchCompensation", [this]() { return gravityWrench_; });
  logger.addLogEntry(name_ + "_compensateGravity", [this]() { return compensateGravity_; });
  logger.addLogEntry(name_ + "_integralTermEnabled", [this]() { return integralTermEnabled_; });
  logger.addLogEntry(name_ + "_integralGain", [this]() { return integralGain_; });
  logger.addLogEntry(name_ + "_maxIntegralWrench", [this]() { return maxIntegralWrench_; });
  logger.addLogEntry(name_ + "_integralError", [this]() { return integralError_; });
  logger.addLogEntry(name_ + "_integralWrench", [this]() { return integralWrench_; });
}

} // namespace mc_tasks
