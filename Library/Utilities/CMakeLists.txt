add_library(Utilities
				ClosestPointComputer.h
				ClosestPointComputer.cpp
				GridUtilities.h
				Integrator.h
				ScalarGrid.h
				SparseUniformGrid.h
				Timer.h
				Transform.h
				UniformGrid.h
				Utilities.h
				Utilities.cpp
				VectorGrid.h)

target_include_directories(Utilities PUBLIC
								$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
								$<INSTALL_INTERFACE:include>)

target_link_libraries(Utilities
						PUBLIC
						RenderTools)

set_target_properties(Utilities PROPERTIES FOLDER ${SOURCE_FOLDER})