add_library(SurfaceTrackers
				InitialGeometry.cpp
				LevelSet.cpp
				Predicates.cpp
				TriMesh.cpp)

target_link_libraries(SurfaceTrackers
						PRIVATE
						RenderTools
						Utilities)

find_package(EIGEN3 REQUIRED)
if (EIGEN3_FOUND)
    include_directories(${EIGEN3_INCLUDE_DIR})
endif()

target_include_directories(SurfaceTrackers PUBLIC
							  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
							  $<INSTALL_INTERFACE:include>)

set_target_properties(SurfaceTrackers PROPERTIES FOLDER ${SOURCE_FOLDER})