add_executable(
        unit_tests
        tiny_solver_autodiff_function_test.cc
        tiny_solver_test.cc
)

target_link_libraries(
        unit_tests
        gtest_main
)
