#include <iostream>
#include <memory>

#include "imgui.h"
#include "polyscope/polyscope.h"

#include "InitialGeometry.h"
#include "Integrator.h"
#include "LevelSet.h"
#include "TestVelocityFields.h"
#include "Transform.h"
#include "TriMesh.h"
#include "Utilities.h"

using namespace FluidSim3D;

int main()
{
    double dx = .005;
    Vec3d topRightCorner = Vec3d::Ones();
    Vec3d bottomLeftCorner = Vec3d::Zero();
    Vec3i gridSize = ((topRightCorner - bottomLeftCorner) / dx).cast<int>();
    Transform xform(dx, bottomLeftCorner);

    constexpr double dt = 1. / 50.;

    auto simulator = std::make_unique<DeformationField>(0., 3.);

    TriMesh triMesh = makeSphereMesh(Vec3d::Constant(.35), .15, 4);

    auto implicitSurface = std::make_unique<LevelSet>(xform, gridSize, 5);
    implicitSurface->initFromMesh(triMesh, false);

    int frameCount = 0;
    bool runSimulation = false;
    bool runSingleTimestep = false;

    polyscope::init();
    polyscope::options::groundPlaneEnabled = false;

    triMesh.drawMesh("level set mesh", Vec3d::Constant(.5));

    polyscope::state::userCallback = [&]()
    {
        if (ImGui::Button("Run/Pause")) runSimulation = !runSimulation;
        ImGui::SameLine();
        if (ImGui::Button("Step")) runSingleTimestep = true;

        if (runSimulation || runSingleTimestep)
        {
            std::cout << "\nStart of frame: " << frameCount << std::endl;
            ++frameCount;

            triMesh.advectMesh(dt, *simulator, IntegrationOrder::RK3);
            simulator->advanceField(dt);

            implicitSurface->initFromMesh(triMesh, false);
            triMesh = implicitSurface->buildMesh();

            runSingleTimestep = false;
        }

        triMesh.drawMesh("level set mesh", Vec3d::Constant(.5));
    };

    polyscope::show();
}
