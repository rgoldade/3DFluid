add_executable(TestInitialGeometry TestInitialGeometry.cpp)

target_link_libraries(TestInitialGeometry
						PRIVATE
						RenderTools
						SimTools
						SurfaceTrackers
						Utilities)

file( RELATIVE_PATH REL ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR} )						

install(TARGETS TestInitialGeometry RUNTIME DESTINATION ${REL})

set_target_properties(TestInitialGeometry PROPERTIES FOLDER ${TEST_FOLDER})