#include <iostream>
#include <memory>

#include "imgui.h"
#include "polyscope/polyscope.h"

#include "EulerianLiquidSimulator.h"
#include "InitialGeometry.h"
#include "LevelSet.h"
#include "TestVelocityFields.h"
#include "Transform.h"
#include "TriMesh.h"
#include "Utilities.h"
#include "VectorGrid.h"

using namespace FluidSim3D;

int main()
{
    double dx = .0125;
    Vec3d topRightCorner(.75, 1., .5);
    Vec3d bottomLeftCorner(-.75, -1., -.5);
    Vec3i gridSize = ((topRightCorner - bottomLeftCorner) / dx).cast<int>();
    Transform xform(dx, bottomLeftCorner);
    Vec3d center = .5 * (topRightCorner + bottomLeftCorner);

    constexpr double dt = 1. / 30.;

    // Build static boundary geometry
    double solidThickness = 10;
    Vec3d solidScale = .5 * (topRightCorner - bottomLeftCorner) - Vec3d::Constant(solidThickness * dx);
    TriMesh staticSolidMesh = makeCubeMesh(center, solidScale);
    staticSolidMesh.reverse();
    assert(staticSolidMesh.unitTestMesh());

    // Build moving boundary geometry
    TriMesh movingSolidMesh = makeSphereMesh(center + Vec3d(.4, -.25, 0), .1, 3);

    LevelSet movingSolidSurface(xform, gridSize, 5);
    movingSolidSurface.initFromMesh(movingSolidMesh, false);

    LevelSet combinedSolidSurface(xform, gridSize, 5);
    combinedSolidSurface.setBackgroundNegative();
    combinedSolidSurface.initFromMesh(staticSolidMesh, false);
    combinedSolidSurface.unionSurface(movingSolidSurface);

    // Build seeding liquid geometry
    TriMesh seedLiquidMesh = makeCubeMesh(center + Vec3d(0, .4, 0), .5 * Vec3d(.075, .3, .075));
    assert(seedLiquidMesh.unitTestMesh());

    LevelSet seedLiquidSurface(xform, gridSize, 5);
    seedLiquidSurface.initFromMesh(seedLiquidMesh, false);

    // Set up simulator
    auto simulator = std::make_unique<EulerianLiquidSimulator>(xform, gridSize, 5);
    simulator->setLiquidSurface(seedLiquidSurface);
    simulator->setSolidSurface(combinedSolidSurface);
    simulator->setViscosity(1.);

    auto solidVelocityField = std::make_unique<CircularField>(center, 1, Axis::ZAXIS);

    int frameCount = 0;
    bool runSimulation = false;
    bool runSingleTimestep = false;
    polyscope::init();
    polyscope::options::groundPlaneEnabled = false;

    simulator->drawLiquidSurface("sim");
    movingSolidMesh.drawMesh("sim");

    polyscope::state::userCallback = [&]()
    {
        if (ImGui::Button("Run/Pause")) runSimulation = !runSimulation;
        ImGui::SameLine();
        if (ImGui::Button("Step")) runSingleTimestep = true;

        if (runSimulation || runSingleTimestep)
        {
            std::cout << "\nStart of frame: " << frameCount << std::endl;
            ++frameCount;

            double frameTime = 0.;
            while (frameTime < dt)
            {
                double speed = simulator->maxVelocityMagnitude();
                double localDt = dt - frameTime;
                assert(localDt >= 0);

                if (speed > 1E-6)
                {
                    double cflDt = 3. * xform.dx() / speed;
                    if (localDt > cflDt)
                    {
                        localDt = cflDt;
                        std::cout << "\n  Throttling frame with substep: " << localDt << "\n" << std::endl;
                    }
                }

                if (localDt <= 0) break;

                simulator->addForce(localDt, Vec3d(0, -9.8, 0));

                movingSolidMesh.advectMesh(localDt, *solidVelocityField, IntegrationOrder::RK3);

                LevelSet localMovingSolidSurface(xform, gridSize, 5);
                localMovingSolidSurface.initFromMesh(movingSolidMesh, false);

                VectorGrid<double> movingSolidVelocity(xform, gridSize, Vec3d::Zero(), VectorGridSettings::SampleType::STAGGERED);

                for (int axis : {0, 1, 2})
                {
                    tbb::parallel_for(
                        tbb::blocked_range<int>(0, movingSolidVelocity.grid(axis).voxelCount(), tbbLightGrainSize),
                        [&](const tbb::blocked_range<int>& range) {
                            for (int faceIndex = range.begin(); faceIndex != range.end(); ++faceIndex)
                            {
                                Vec3i face = movingSolidVelocity.grid(axis).unflatten(faceIndex);
                                Vec3d worldPosition = movingSolidVelocity.indexToWorld(face.cast<double>(), axis);

                                if (localMovingSolidSurface.triLerp(worldPosition) < xform.dx())
                                    movingSolidVelocity(face, axis) = (*solidVelocityField)(0, worldPosition)[axis];
                            }
                        });
                }

                LevelSet localCombinedSolidSurface(xform, gridSize, 5);
                localCombinedSolidSurface.setBackgroundNegative();
                localCombinedSolidSurface.initFromMesh(staticSolidMesh, false);
                localCombinedSolidSurface.unionSurface(localMovingSolidSurface);

                simulator->setSolidSurface(localCombinedSolidSurface);
                simulator->setSolidVelocity(movingSolidVelocity);

                simulator->runTimestep(localDt);

                simulator->unionLiquidSurface(seedLiquidSurface);

                frameTime += localDt;
            }

            runSingleTimestep = false;
        }

        simulator->drawLiquidSurface("sim");
        movingSolidMesh.drawMesh("sim");
    };

    polyscope::show();
}
