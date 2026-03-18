#include <iostream>
#include <memory>

#include "imgui.h"
#include "polyscope/polyscope.h"

#include "EulerianLiquidSimulator.h"
#include "InitialGeometry.h"
#include "LevelSet.h"
#include "Transform.h"
#include "TriMesh.h"
#include "Utilities.h"

using namespace FluidSim3D;

int main()
{
    double dx = .05;
    double solidSphereRadius = 2;
    Vec3d topRightCorner = Vec3d::Constant(solidSphereRadius + 15 * dx);
    Vec3d bottomLeftCorner = Vec3d::Constant(-solidSphereRadius - 15 * dx);
    Vec3i gridSize = ((topRightCorner - bottomLeftCorner) / dx).cast<int>();
    Transform xform(dx, bottomLeftCorner);
    Vec3d center = .5 * (topRightCorner + bottomLeftCorner);

    constexpr double dt = 1. / 90.;
    constexpr double cfl = 1.;

    TriMesh liquidMesh = makeSphereMesh(center - Vec3d(0, .65, 0), 1, 3);
    assert(liquidMesh.unitTestMesh());

    LevelSet liquidSurface(xform, gridSize, 5);
    liquidSurface.initFromMesh(liquidMesh, false);

    TriMesh solidMesh = makeSphereMesh(center, 2, 3);
    solidMesh.reverse();
    assert(solidMesh.unitTestMesh());

    LevelSet solidSurface(xform, gridSize, 5);
    solidSurface.setBackgroundNegative();
    solidSurface.initFromMesh(solidMesh, false);

    auto simulator = std::make_unique<EulerianLiquidSimulator>(xform, gridSize, 5);
    simulator->setLiquidSurface(liquidSurface);
    simulator->setSolidSurface(solidSurface);

    int frameCount = 0;
    bool runSimulation = false;
    bool runSingleTimestep = false;
    bool drawSolidBoundaries = true;
    bool drawLiquidVelocities = true;
    bool writeLiquidSurface = true;
    float planePosition = .5f;
    int planeAxisInt = 2;

    polyscope::init();
    polyscope::options::groundPlaneEnabled = false;

    simulator->drawLiquidSurface("sim");
    if (drawSolidBoundaries) simulator->drawSolidSurface("sim");

    polyscope::state::userCallback = [&]()
    {
        if (ImGui::Button("Run/Pause")) runSimulation = !runSimulation;
        ImGui::SameLine();
        if (ImGui::Button("Step")) runSingleTimestep = true;

        ImGui::Checkbox("Solid boundaries", &drawSolidBoundaries);
        ImGui::Checkbox("Liquid velocities", &drawLiquidVelocities);
        ImGui::Checkbox("Write surface OBJ", &writeLiquidSurface);
        ImGui::SliderFloat("Plane position", &planePosition, 0.f, 1.f);
        ImGui::Combo("Plane axis", &planeAxisInt, "X\0Y\0Z\0");

        Axis planeAxis = planeAxisInt == 0 ? Axis::XAXIS : (planeAxisInt == 1 ? Axis::YAXIS : Axis::ZAXIS);

        if (runSimulation || runSingleTimestep)
        {
            std::cout << "\nStart of frame: " << frameCount << std::endl;

            double frameTime = 0.;
            while (frameTime < dt)
            {
                double speed = simulator->maxVelocityMagnitude();
                double localDt = dt - frameTime;
                assert(localDt >= 0);

                if (speed > 1E-6)
                {
                    double cflDt = cfl * xform.dx() / speed;
                    if (localDt > cflDt)
                    {
                        localDt = cflDt;
                        std::cout << "\n  Throttling frame with substep: " << localDt << "\n" << std::endl;
                    }
                }

                if (localDt <= 0) break;

                simulator->addForce(localDt, Vec3d(0, -9.8, 0));
                simulator->runTimestep(localDt);

                frameTime += localDt;
            }
            std::cout << "\n\nEnd of frame: " << frameCount << "\n" << std::endl;
            ++frameCount;

            runSingleTimestep = false;
        }

        simulator->drawLiquidSurface("sim");

        if (drawSolidBoundaries) simulator->drawSolidSurface("sim");

        if (drawLiquidVelocities) simulator->drawLiquidVelocity("sim", planeAxis, planePosition, 1);

        if (writeLiquidSurface)
        {
            std::string surfaceName = std::string("liquid_surface_") + std::to_string(frameCount) + std::string(".obj");
            simulator->writeLiquidSurface(surfaceName);
        }
    };

    polyscope::show();
}
