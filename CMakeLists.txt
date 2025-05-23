cmake_minimum_required(VERSION 3.8)
project(ar4_practice)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
#find_package(rosidl_default_generators REQUIRED)  # Required for service generation

# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
find_package(geometry_msgs REQUIRED)
#rosidl_generate_interfaces(${PROJECT_NAME}
 # "srv/GetPose.srv"
 # DEPENDENCIES geometry_msgs
#)
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()
# Install the launch directory
install(
    DIRECTORY launch/
    DESTINATION share/${PROJECT_NAME}/launch
)
install (PROGRAMS 
  scripts/joint_commander.py
  scripts/movement_commander.py
  scripts/movement_reader.py
  scripts/work_envelope_planner.py
  scripts/work_envelope_analyzer.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
