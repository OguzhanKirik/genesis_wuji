cmake_minimum_required(VERSION 3.16)
project(genesis_policy NONE)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# genesis_policy: the runtime (level loader, URDF model, observation assembly,
# IK, phase machine, actor) as an importable package, and the node.
ament_python_install_package(${PROJECT_NAME})

# policy_node.py is an ordinary rclpy node; make_stub_policy.py writes a
# zero-action actor.onnx and a placeholder level.yaml so the whole path runs
# before adam-rl's artefacts arrive.
install(PROGRAMS
  scripts/policy_node.py
  scripts/selector_node.py
  scripts/make_stub_policy.py
  scripts/grasp_cycle.py
  DESTINATION lib/${PROJECT_NAME}
)

# launch/policy.launch.py; config/defaults.yaml: the node's parameters.
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  # No ROS runtime in any of these: the level schema, FK/Jacobian against an
  # independent URDF walk, the IK and the phase machine on a scripted actor,
  # and the parity replay of adam-rl's recorded episode (skips while the
  # fixture is absent). The launch_testing suite on the loopback stack is
  # genesis_integration_tests/test/test_policy_node.py.
  ament_add_pytest_test(test_level test/test_level.py TIMEOUT 120)
  ament_add_pytest_test(test_model test/test_model.py TIMEOUT 300)
  ament_add_pytest_test(test_runtime test/test_runtime.py TIMEOUT 300)
  ament_add_pytest_test(test_parity test/test_parity.py TIMEOUT 300)
  ament_add_pytest_test(test_selector test/test_selector.py TIMEOUT 120)
endif()

ament_package()
