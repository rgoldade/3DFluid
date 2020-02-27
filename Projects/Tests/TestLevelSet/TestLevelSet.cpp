#include <iostream>
#include <memory>

#include "Camera3D.h"
#include "InitialGeometry.h"
#include "Integrator.h"
#include "LevelSet.h"
#include "Renderer.h"
#include "TestVelocityFields.h"
#include "Transform.h"
#include "TriMesh.h"
#include "Utilities.h"
#include "Vec.h"

using namespace FluidSim3D::RenderTools;
using namespace FluidSim3D::SimTools;
using namespace FluidSim3D::SurfaceTrackers;
using namespace FluidSim3D::Utilities;

static std::unique_ptr<Renderer> renderer;
static std::unique_ptr<Camera3D> camera;

static int frameCount = 0;
static bool runSimulation = false;
static bool runSingleTimestep = false;
static bool isDisplayDirty = true;

static float planePosition = .5;
static float planeDX = .05;
static Axis planeAxis = Axis::ZAXIS;

static std::unique_ptr<DeformationField> simulator;
static std::unique_ptr<LevelSet> implicitSurface;
static TriMesh triMesh;

static constexpr float dt = 1. / 50.;

void keyboard(unsigned char key, int x, int y)
{
    if (key == '+')
        planePosition += planeDX;
    else if (key == '-')
        planePosition -= planeDX;
    else if (key == 'x')
        planeAxis = Axis::XAXIS;
    else if (key == 'y')
        planeAxis = Axis::YAXIS;
    else if (key == 'z')
        planeAxis = Axis::ZAXIS;
    else if (key == ' ')
        runSimulation = !runSimulation;
    else if (key == 'n')
        runSingleTimestep = true;

    isDisplayDirty = true;
}

void display()
{
    if (runSimulation || runSingleTimestep)
    {
        std::cout << "\nStart of frame: " << frameCount << std::endl;
        ++frameCount;

        // Advect mesh through deformation field
        triMesh.advectMesh(dt, *simulator, IntegrationOrder::RK3);

        simulator->advanceField(dt);

        implicitSurface->initFromMesh(triMesh, false);
        triMesh = implicitSurface->buildMesh();

        isDisplayDirty = true;
    }

    if (isDisplayDirty)
    {
        renderer->clear();

        triMesh.drawMesh(*renderer, true /* draw triangle faces */, Vec3f(.5), false /* don't render triangle normals*/,
                         Vec3f(1, 0, 0), false /*don't render mesh vertices*/, Vec3f(0), true /* draw triangle edges*/,
                         Vec3f(0));

        isDisplayDirty = false;

        glutPostRedisplay();
    }
    runSingleTimestep = false;
}

int main(int argc, char** argv)
{
    float dx = .005;
    Vec3f topRightCorner(1);
    Vec3f bottomLeftCorner(0);
    Vec3i gridSize = Vec3i((topRightCorner - bottomLeftCorner) / dx);
    Transform xform(dx, bottomLeftCorner);

    planeDX =
        std::min(float(1) / float(gridSize[0]), std::min(float(1) / float(gridSize[1]), float(1) / float(gridSize[2])));

    renderer =
        std::make_unique<Renderer>("Level set test", Vec2i(1000), Vec2f(bottomLeftCorner[0], bottomLeftCorner[1]),
                                   topRightCorner[1] - bottomLeftCorner[1], &argc, argv);
    camera = std::make_unique<Camera3D>(.5 * (topRightCorner + bottomLeftCorner), 2.5, 0., 0.);
    renderer->setCamera(camera.get());

    simulator = std::make_unique<DeformationField>(0., 3.);

    triMesh = makeSphereMesh(Vec3f(.35), .15, dx);

    // Scene settings
    implicitSurface = std::make_unique<LevelSet>(xform, gridSize, 5);
    implicitSurface->initFromMesh(triMesh, false);

    std::function<void()> displayFunc = display;
    renderer->setUserDisplay(displayFunc);

    std::function<void(unsigned char, int, int)> keyboardFunc = keyboard;
    renderer->setUserKeyboard(keyboardFunc);
    renderer->run();
}