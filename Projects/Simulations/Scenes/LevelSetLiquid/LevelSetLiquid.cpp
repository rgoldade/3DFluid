#include <iostream>
#include <memory>

#include "Camera3D.h"
#include "EulerianLiquidSimulator.h"
#include "InitialGeometry.h"
#include "LevelSet.h"
#include "Renderer.h"
#include "Transform.h"
#include "TriMesh.h"
#include "Utilities.h"

using namespace FluidSim3D;

static std::unique_ptr<Renderer> renderer;
static std::unique_ptr<Camera3D> camera;
static std::unique_ptr<EulerianLiquidSimulator> simulator;

static int frameCount = 0;
static bool runSimulation = false;
static bool runSingleTimestep = false;
static bool isDisplayDirty = true;

static bool drawSolidBoundaries = true;
static bool drawLiquidVelocities = true;
static bool writeLiquidSurface = true;

static double planePosition = .5;
static double planeDX = .05;
static Axis planeAxis = Axis::ZAXIS;

static Transform xform;

static constexpr double dt = 1. / 90.;

static constexpr double cfl = 1.;

void display()
{
    if (runSimulation || runSingleTimestep)
    {
        std::cout << "\nStart of frame: " << frameCount << std::endl;

        double frameTime = 0.;
        while (frameTime < dt)
        {
            // Set CFL condition
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

            // Add gravity
            simulator->addForce(localDt, Vec3d(0, -9.8, 0));

            // Projection set unfortunately includes viscosity at the moment
            simulator->runTimestep(localDt);

            frameTime += localDt;
        }
        std::cout << "\n\nEnd of frame: " << frameCount << "\n" << std::endl;
        ++frameCount;

        runSingleTimestep = false;
        isDisplayDirty = true;
    }
    if (isDisplayDirty)
    {
        renderer->clear();

        simulator->drawLiquidSurface(*renderer);

        if (drawSolidBoundaries) simulator->drawSolidSurface(*renderer);

        if (drawLiquidVelocities) simulator->drawLiquidVelocity(*renderer, planeAxis, planePosition, 1);

        if (writeLiquidSurface)
        {
            std::string surfaceName = std::string("liquid_surface_") + std::to_string(frameCount) + std::string(".obj");
            simulator->writeLiquidSurface(surfaceName);
        }

        isDisplayDirty = false;

        glutPostRedisplay();
    }
}

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
    else if (key == 'p')
        writeLiquidSurface = !writeLiquidSurface;
    else if (key == 's')
        drawSolidBoundaries = !drawSolidBoundaries;
    else if (key == 'v')
        drawLiquidVelocities = !drawLiquidVelocities;

    isDisplayDirty = true;
}

int main(int argc, char** argv)
{
    double dx = .05;
    double solidSphereRadius = 2;
    Vec3d topRightCorner = Vec3d::Constant(solidSphereRadius + 15 * dx);
    Vec3d bottomLeftCorner = Vec3d::Constant(-solidSphereRadius - 15 * dx);
    Vec3i gridSize = ((topRightCorner - bottomLeftCorner) / dx).cast<int>();
    xform = Transform(dx, bottomLeftCorner);
    Vec3d center = .5 * (topRightCorner + bottomLeftCorner);

    renderer = std::make_unique<Renderer>("Levelset liquid simulator", Vec2i::Constant(1000),
                                          Vec2d(bottomLeftCorner[0], bottomLeftCorner[1]),
                                          topRightCorner[1] - bottomLeftCorner[1], &argc, argv);
    camera = std::make_unique<Camera3D>(.5 * (topRightCorner + bottomLeftCorner), 6, 0., 0.);
    renderer->setCamera(camera.get());

    // Build initial liquid geometry

    TriMesh liquidMesh = makeSphereMesh(center - Vec3d(0, .65, 0), 1, 3);
    assert(liquidMesh.unitTestMesh());

    LevelSet liquidSurface(xform, gridSize, 5);
    liquidSurface.initFromMesh(liquidMesh, false);

    // Build solid boundary

    TriMesh solidMesh = makeSphereMesh(center, 2, 3);
    solidMesh.reverse();
    assert(solidMesh.unitTestMesh());

    LevelSet solidSurface(xform, gridSize, 5);
    solidSurface.setBackgroundNegative();
    solidSurface.initFromMesh(solidMesh, false);

    // Set up simulator

    simulator = std::make_unique<EulerianLiquidSimulator>(xform, gridSize, 5);
    simulator->setLiquidSurface(liquidSurface);
    simulator->setSolidSurface(solidSurface);

    std::function<void()> displayFunc = display;
    renderer->setUserDisplay(displayFunc);

    std::function<void(unsigned char, int, int)> keyboardFunc = keyboard;
    renderer->setUserKeyboard(keyboardFunc);

    renderer->run();
}