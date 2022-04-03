add_executable(TestEnrightFlow TestEnrightFlow.cpp)

target_link_libraries(TestEnrightFlow
						PRIVATE
						MeshRebuilderLibrary
						RenderTools
						SimTools
						SurfaceTrackers
						Utilities)

file(RELATIVE_PATH REL ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

install(TARGETS TestEnrightFlow RUNTIME DESTINATION ${REL})

set_target_properties(TestEnrightFlow PROPERTIES FOLDER ${TEST_FOLDER}/TestProjects)