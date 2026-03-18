#include <iostream>
#include <memory>

#include "imgui.h"
#include "polyscope/polyscope.h"

#include "InitialGeometry.h"
#include "TriMesh.h"
#include "Utilities.h"

using namespace FluidSim3D;

int main()
{
    int geometryIndex = 0;
    TriMesh triMesh = makeDiamondMesh(Vec3d::Zero(), 1.);

    polyscope::init();
    polyscope::options::groundPlaneEnabled = false;

    triMesh.drawMesh("geometry", Vec3d::Constant(.5));

    polyscope::state::userCallback = [&]()
    {
        int prevGeometryIndex = geometryIndex;
        ImGui::Combo("Geometry", &geometryIndex, "Diamond\0Cube\0Icosahedron\0Sphere\0");

        if (geometryIndex != prevGeometryIndex)
        {
            if (geometryIndex == 0)
                triMesh = makeDiamondMesh(Vec3d::Zero(), 1.);
            else if (geometryIndex == 1)
                triMesh = makeCubeMesh(Vec3d::Zero(), Vec3d::Ones());
            else if (geometryIndex == 2)
                triMesh = makeIcosahedronMesh();
            else if (geometryIndex == 3)
                triMesh = makeSphereMesh(Vec3d::Zero(), 1., 1);
        }

        triMesh.drawMesh("geometry", Vec3d::Constant(.5));
    };

    polyscope::show();
}
