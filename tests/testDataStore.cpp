/*
 * Copyright 2015-2019 CNRS-UM LIRMM, CNRS-AIST JRL
 */

#include <mc_rbdyn/RobotLoader.h>
#include <mc_rbdyn/Robots.h>
#include <mc_rbdyn/lipm_stabilizer/StabilizerConfiguration.h>
#include <mc_rtc/DataStore.h>
#include <boost/test/unit_test.hpp>
#include "utils.h"
#include <Eigen/Core>

using DataStore = mc_rtc::DataStore;

BOOST_AUTO_TEST_CASE(TestDataStore)
{
  DataStore store;
  store.make<std::vector<double>>("data", 4, 42.0);
  auto & data = store.get<std::vector<double>>("data");
  BOOST_REQUIRE(data.size() == 4);
  BOOST_CHECK_CLOSE(data[0], 42, 1e-10);
  BOOST_CHECK_CLOSE(data[1], 42, 1e-10);
  BOOST_CHECK_CLOSE(data[2], 42, 1e-10);
  BOOST_CHECK_CLOSE(data[3], 42, 1e-10);
  BOOST_CHECK_THROW(store.get<double>("data"), std::runtime_error);
  BOOST_CHECK_THROW(store.get<std::vector<int>>("data"), std::runtime_error);
  BOOST_CHECK_THROW(store.get<std::vector<double>>("non-existing key"), std::runtime_error);

  data.resize(100);
  auto & data2 = store.get<std::vector<double>>("data");
  BOOST_REQUIRE(data2.size() == data.size());
  const auto & constData = store.get<std::vector<double>>("data");
  BOOST_REQUIRE(data2.size() == constData.size());

  // Test make_initializer
  struct Test
  {
    int a;
    std::string name;
  };
  store.make_initializer<Test>("Test", 42, "Test");
  BOOST_REQUIRE(store.get<Test>("Test").a == 42);
  BOOST_REQUIRE(store.get<Test>("Test").name == "Test");

  // Try creating an object that already exists on the datastore
  BOOST_CHECK_THROW(store.make_initializer<Test>("Test", 42, "Test"), std::runtime_error);
  // Check that existing object has not been modified
  BOOST_REQUIRE(store.get<Test>("Test").a == 42);
  BOOST_REQUIRE(store.get<Test>("Test").name == "Test");
  // Remove object
  store.remove("Test");
  BOOST_CHECK(!store.has("Test"));
  BOOST_CHECK(store.has("data"));
  // Recreate it with the same name and directly assign some value to it
  // Recreate it with the same name and directly assign some value to it
  store.make_initializer<Test>("Test", 42, "Test").name = "Test2";
  BOOST_REQUIRE(store.get<Test>("Test").name == "Test2");

  // Check creating object of a different type with same name
  store.remove("Test");
  store.make_initializer<std::vector<double>>("Test", 1., 2.);
  auto & v = store.get<std::vector<double>>("Test");
  BOOST_REQUIRE(v.size() == 2);
  BOOST_REQUIRE(v[0] == 1);
  BOOST_REQUIRE(v[1] == 2);

  // Test operators
  store.make<double>("TestAssign", 42);
  double value = 0;
  store.get("TestAssign", value);
  BOOST_REQUIRE(value == 42);
  // Test modifying existing value from non-existing key in the datastore
  store.get("TestAssignNonExisting", value);
  BOOST_REQUIRE(value == 42);
  // Test getting non-exising value with default
  BOOST_REQUIRE(store.get("TestAssignNonExisting", 12) == 12);
  BOOST_REQUIRE(store.get("HasFeature", false) == false);
}

BOOST_AUTO_TEST_CASE(TestDataStoreInheritance)
{
  struct A
  {
    A(const std::string & name) : name_(name) {}
    virtual std::string hello() const
    {
      return "A::Hello " + name_;
    }
    void printHello() const
    {
      std::cout << hello() << std::endl;
    }
    std::string type() const
    {
      return "A";
    }
    std::string name_;
  };

  struct B : public A
  {
    B(const std::string & name) : A(name) {}
    std::string hello() const override
    {
      return "B::Hello " + name_;
    }
    std::string type() const
    {
      return "B";
    }
  };

  DataStore store;

  // Creating an inherited object and checking virtual inheritance
  store.make<B, A>("b", "World");
  auto & a = store.get<A>("b");
  // Checking member function
  BOOST_REQUIRE(a.type() == "A");
  // Checking virtual function
  BOOST_REQUIRE(a.hello() == "B::Hello World");
  auto & b = store.get<B>("b");
  BOOST_REQUIRE(b.type() == "B");
  BOOST_REQUIRE(b.hello() == "B::Hello World");
}

BOOST_AUTO_TEST_CASE(TestRobotDataStore)
{
  configureRobotLoader();
  auto rm = mc_rbdyn::RobotLoader::get_robot_module("JVRC1");
  auto env = mc_rbdyn::RobotLoader::get_robot_module("env", std::string(mc_rtc::MC_ENV_DESCRIPTION_PATH),
                                                     std::string("ground"));

  DataStore store;
  auto & robots = store.make<mc_rbdyn::Robots>("robots");
  robots.load({rm, env});
  BOOST_REQUIRE(robots.size() == 2);

  // Get another reference to robots
  auto & robots2 = store.get<mc_rbdyn::Robots>("robots");
  BOOST_REQUIRE(robots2.size() == 2);

  robots.robot().posW(sva::PTransformd(Eigen::Vector3d{42, 42, 42}));
  BOOST_CHECK_CLOSE(robots.robot().posW().translation().x(), 42, 1e-10);
  BOOST_CHECK_CLOSE(robots.robot().posW().translation().y(), 42, 1e-10);
  BOOST_CHECK_CLOSE(robots.robot().posW().translation().z(), 42, 1e-10);
  BOOST_CHECK_CLOSE(robots2.robot().posW().translation().x(), 42, 1e-10);
  BOOST_CHECK_CLOSE(robots2.robot().posW().translation().y(), 42, 1e-10);
  BOOST_CHECK_CLOSE(robots2.robot().posW().translation().z(), 42, 1e-10);
}

BOOST_AUTO_TEST_CASE(Lambda)
{
  DataStore store;
  struct A
  {
    double val;
    double compute(double t)
    {
      return val * t;
    }
  };
  A a;
  a.val = 42;
  store.make<std::function<void(double)>>("lambda_setter", [&a](double val) { a.val = val; });
  auto & setter = store.get<std::function<void(double)>>("lambda_setter");
  setter(33);
  BOOST_REQUIRE(a.val == 33);

  store.make<std::function<double()>>("lambda_getter", [&a]() { return a.val; });
  auto & getter = store.get<std::function<double()>>("lambda_getter");
  auto val = getter();
  BOOST_REQUIRE(val == 33);

  double val2 = 33;
  store.get<double>("Value", val2);

  store.make<std::function<double(double)>>("lambda_compute", [&a](double t) { return a.compute(t); });
  auto res = store.get<std::function<double(double)>>("lambda_compute")(2);
  BOOST_REQUIRE(res == 66);

  // Dummy example of a footstepplan decalring a callback mechanism to trigger
  // recomputation of the footstep plan.
  struct FootstepPlan
  {
    FootstepPlan(DataStore & store) : store_(store)
    {
      store.make<std::function<std::vector<double>()>>("compute_footstep", [this]() {
        recompute();
        return plan_;
      });
    }

    ~FootstepPlan()
    {
      store_.remove("compute_footstep");
    }

    // In practice this would do actual computations, just add a dummy number to
    // the footstep vector for this test
    void recompute()
    {
      plan_.push_back(plan_.back() + 1);
    }

    std::vector<double> plan_{1, 2, 3};
    DataStore & store_;
  };

  {
    FootstepPlan plan(store);
    // XXX can we retrieve it without specifying type explicitely?
    auto & computeFootStep = store.get<std::function<std::vector<double>()>>("compute_footstep");
    BOOST_REQUIRE(computeFootStep().back() == 4);
    BOOST_REQUIRE(computeFootStep().back() == 5);
    BOOST_REQUIRE(computeFootStep().back() == 6);
  }
  // Check that lambda was automatically removed from datastore
  BOOST_CHECK(!store.has("compute_footstep"));
}

BOOST_AUTO_TEST_CASE(TestRemove)
{
  struct Object
  {
    Object(const std::string & name) : name_(name)
    {
      LOG_SUCCESS("Object " << name_ << " constructed");
    }

    ~Object()
    {
      LOG_SUCCESS("Object " << name_ << " destructed");
    }
    std::string name_;
  };

  DataStore store;
  store.make<Object>("TestObject", "TestObject");
  {
    auto & r = store.get<Object>("TestObject");
    BOOST_REQUIRE(r.name_ == "TestObject");
    BOOST_CHECK(store.has("TestObject"));
    store.remove("TestObject");
    BOOST_CHECK(!store.has("TestObject"));
  }
}

BOOST_AUTO_TEST_CASE(PointerSharing)
{
  DataStore store;
  struct State
  {
    State(DataStore & store)
    {
      store.make<State *>("APtr", this); // Do not do this, dangerous (see below)
      store.make<std::function<State &(void)>>("ARef", [this]() -> State & { return *this; }); // This is ok
    }
    ~State()
    {
      v = 0;
    }
    double v = 42;
  };
  State state(store);

  // Gets a raw pointer to the state. this is dangerous, as one can misuse the
  // pointer easily
  auto aptr = store.get<State *>("APtr");
  BOOST_REQUIRE(aptr != nullptr);
  BOOST_REQUIRE(aptr->v == 42);
  // delete aptr; any use of state will likely segfault + double free

  // Ok, get a reference to the state through a lambda
  auto & aref = store.get<std::function<State &(void)>>("ARef")();
  BOOST_REQUIRE(aref.v == 42);
  aref.v = 12;
  BOOST_REQUIRE(state.v == 12);
}

BOOST_AUTO_TEST_CASE(TestStabilizer)
{

  using namespace mc_rbdyn::lipm_stabilizer;

  DataStore store;
  store.make<StabilizerConfiguration>("conf");
  BOOST_CHECK_NO_THROW(store.get<StabilizerConfiguration>("conf"));

  store.make<std::function<StabilizerConfiguration(void)>>("getConf");
  BOOST_CHECK_NO_THROW(store.get<std::function<StabilizerConfiguration(void)>>("getConf"));
}
