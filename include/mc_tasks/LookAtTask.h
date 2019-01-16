#pragma once
#include <mc_tasks/VectorOrientationTask.h>

namespace mc_tasks
{

/*! \brief Orient a "gaze" vector defined on a body to look towards a world
 * position. This task is a convenience wrapper around VectorOrientationTask
 */
struct MC_TASKS_DLLAPI LookAtTask : public VectorOrientationTask
{
  /*! \brief Constructor
   * \param bodyName Name of the body to control
   * \param bodyVector Gaze vector for the body.
        For instance [1., 0, 0] will try to align the x axis of the body with
   the target direction.
   * \param targetPos Position of target frame to look towards in world frame
   * \param robots Robots controlled by this task
   * \param robotIndex Index of the robot controlled by this task
   * \param stiffness Task stiffness
   * \param weight Task weight
   *
   */
  LookAtTask(const std::string & bodyName,
             const Eigen::Vector3d & bodyVector,
             const Eigen::Vector3d & targetPos,
             const mc_rbdyn::Robots & robots,
             unsigned int robotIndex,
             double stiffness = 2.0,
             double weight = 500);

  /*! \brief Reset the task */
  void reset() override;

  /**
   * \brief Look towards the target frame position
   *
   * \param pos Target position in world frame
   */
  void target(const Eigen::Vector3d & pos);
  /**
   * @brief Gets the target frame position
   * See targetVector() to obtain the gaze vector
   *
   * @return Target vector in world frame
   */
  Eigen::Vector3d target() const;

private:
  void addToLogger(mc_rtc::Logger & logger) override;
  void removeFromLogger(mc_rtc::Logger & logger) override;
  void addToGUI(mc_rtc::gui::StateBuilder & gui) override;

private:
  /*! Target position in world frame */
  Eigen::Vector3d target_pos_;
};

} // namespace mc_tasks
