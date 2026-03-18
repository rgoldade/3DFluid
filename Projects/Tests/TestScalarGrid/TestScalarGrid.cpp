#include <memory>

#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"

#include "imgui.h"
#include "polyscope/polyscope.h"

#include "ScalarGrid.h"
#include "Transform.h"
#include "UniformGrid.h"
#include "Utilities.h"
#include "VectorGrid.h"

using namespace FluidSim3D;

int main()
{
    double dx = .25;
    Vec3d topRightCorner = Vec3d::Constant(10);
    Vec3d bottomLeftCorner = Vec3d::Constant(-10);
    Vec3i gridSize = ((topRightCorner - bottomLeftCorner) / dx).cast<int>();
    Transform xform(dx, bottomLeftCorner);

    auto testScalarGrid = std::make_unique<ScalarGrid<double>>(xform, gridSize);
    auto testVectorGrid = std::make_unique<VectorGrid<double>>(xform, gridSize, VectorGridSettings::SampleType::STAGGERED);

    tbb::parallel_for(tbb::blocked_range<int>(0, testScalarGrid->voxelCount(), tbbLightGrainSize),
                      [&](const tbb::blocked_range<int>& range)
        {
            for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
            {
                Vec3i cell = testScalarGrid->unflatten(cellIndex);
                Vec3d worldPoint = testScalarGrid->indexToWorld(cell.cast<double>());
                (*testScalarGrid)(cell) = worldPoint.norm() - 5.;
            }
        });

    for (int axis : {0, 1, 2})
    {
        tbb::parallel_for(tbb::blocked_range<int>(0, testVectorGrid->grid(axis).voxelCount(), tbbLightGrainSize),
                          [&](const tbb::blocked_range<int>& range) {
                              for (int faceIndex = range.begin(); faceIndex != range.end(); ++faceIndex)
                              {
                                  Vec3i face = testVectorGrid->grid(axis).unflatten(faceIndex);
                                  Vec3d worldPoint = testVectorGrid->indexToWorld(face.cast<double>(), axis);
                                  Vec3d gradVector = testScalarGrid->triLerpGradient(worldPoint);
                                  (*testVectorGrid)(face, axis) = gradVector[axis];
                              }
                          });
    }

    float planePosition = .5f;
    int planeAxisInt = 2;

    polyscope::init();
    polyscope::options::groundPlaneEnabled = false;

    polyscope::state::userCallback = [&]()
    {
        ImGui::SliderFloat("Plane position", &planePosition, 0.f, 1.f);
        ImGui::Combo("Plane axis", &planeAxisInt, "X\0Y\0Z\0");

        Axis planeAxis = planeAxisInt == 0 ? Axis::XAXIS : (planeAxisInt == 1 ? Axis::YAXIS : Axis::ZAXIS);

        testScalarGrid->drawGridPlane("scalar grid", planeAxis, planePosition);
        testScalarGrid->drawSupersampledValuesPlane("scalar grid", planeAxis, planePosition, .5, 3, .001);
        testVectorGrid->drawSamplePointVectors("vector grid", planeAxis, planePosition, Vec3d(1.0, 0.0, 0.0), 1.0);
    };

    polyscope::show();
}
