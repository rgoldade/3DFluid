add_library(SimTools
				ComputeWeights.cpp
				ComputeWeights.h
				ExtrapolateField.h
				FieldAdvector.h
				GeometricConjugateGradientSolver.h
				GeometricMultigridOperators.h
				GeometricMultigridOperators.cpp
				GeometricMultigridPoissonSolver.h
				GeometricMultigridPoissonSolver.cpp
				GeometricPressureProjection.h
				GeometricPressureProjection.cpp
				PressureProjection.cpp
				PressureProjection.h
				TestVelocityFields.h
				ViscositySolver.cpp
				ViscositySolver.h)

target_include_directories(SimTools PUBLIC
						$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
						$<INSTALL_INTERFACE:include>)

target_link_libraries(SimTools
						PRIVATE
						SurfaceTrackers
						Utilities)

set_target_properties(SimTools PROPERTIES FOLDER ${SOURCE_FOLDER})