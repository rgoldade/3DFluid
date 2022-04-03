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

using namespace FluidSim3D;

static std::unique_ptr<Renderer> renderer;
static std::unique_ptr<Camera3D> camera;
static std::unique_ptr<DeformationField> simulator;

static int frameCount = 0;
static bool runSimulation = false;
static bool runSingleTimestep = false;
static bool isDisplayDirty = true;

static LevelSet surface;

static Transform xform;
static constexpr double dt = 1. / 100.;

void keyboard(unsigned char key, int x, int y)
{
    if (key == ' ')
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
        TriMesh mesh = surface.buildMesh();
        mesh.advectMesh(dt, *simulator, IntegrationOrder::RK3);
        surface.initFromMesh(mesh, false);

        simulator->advanceField(dt);

        isDisplayDirty = true;
    }

    if (isDisplayDirty)
    {
        renderer->clear();

        TriMesh mesh = surface.buildMesh();
        mesh.drawMesh(*renderer, true /* draw triangle faces */, Vec3d::Constant(.5), false /* don't render triangle normals*/,
                         Vec3d(1, 0, 0), false /*don't render mesh vertices*/, Vec3d::Zero(), true /* draw triangle edges*/, Vec3d::Zero());

        std::string surfaceName = std::string("enright_level_set_surface_") + std::to_string(frameCount) + std::string(".obj");
        mesh.writeAsOBJ(surfaceName);

        isDisplayDirty = false;

        glutPostRedisplay();
    }
    runSingleTimestep = false;
}

int main(int argc, char** argv)
{
    double dx = .01;
    Vec3d topRightCorner = Vec3d::Ones();
    Vec3d bottomLeftCorner = Vec3d::Zero();
    Vec3i gridSize = ((topRightCorner - bottomLeftCorner) / dx).cast<int>();
    xform = Transform(dx, bottomLeftCorner);

    renderer = std::make_unique<Renderer>("Level set test", Vec2i::Constant(1000), Vec2d(bottomLeftCorner[0], bottomLeftCorner[1]),
                                   topRightCorner[1] - bottomLeftCorner[1], &argc, argv);

    camera = std::make_unique<Camera3D>(.5 * (topRightCorner + bottomLeftCorner), 2.5, 0., 0.);
    renderer->setCamera(camera.get());

    simulator = std::make_unique<DeformationField>(0., 3.);

    TriMesh triMesh = makeSphereMesh(Vec3d::Constant(.35), .15, 4);

    surface = LevelSet(xform, Vec3i::Constant(100));
    surface.initFromMesh(triMesh, false);

    std::function<void()> displayFunc = display;
    renderer->setUserDisplay(displayFunc);

    std::function<void(unsigned char, int, int)> keyboardFunc = keyboard;
    renderer->setUserKeyboard(keyboardFunc);
    renderer->run();
}