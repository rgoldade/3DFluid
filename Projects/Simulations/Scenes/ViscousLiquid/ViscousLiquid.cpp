#include <iostream>
#include <memory>

#include "Camera3D.h"
#include "EulerianLiquidSimulator.h"
#include "InitialGeometry.h"
#include "LevelSet.h"
#include "Renderer.h"
#include "TestVelocityFields.h"
#include "Transform.h"
#include "TriMesh.h"
#include "Utilities.h"
#include "VectorGrid.h"

using namespace FluidSim3D;

static std::unique_ptr<Renderer> renderer;
static std::unique_ptr<Camera3D> camera;

static int frameCount = 0;
static bool runSimulation = false;
static bool runSingleTimestep = false;
static bool isDisplayDirty = true;

static bool drawSolidBoundaries = true;
static bool drawLiquidVelocities = true;

static double planePosition = .5;
static double planeDX = .05;
static Axis planeAxis = Axis::ZAXIS;

static std::unique_ptr<EulerianLiquidSimulator> simulator;
static std::unique_ptr<CircularField> solidVelocityField;

static TriMesh movingSolidMesh;
static TriMesh staticSolidMesh;

static LevelSet seedLiquidSurface;

static Transform xform;
static Vec3i gridSize;

static constexpr double dt = 1. / 30.;

void display()
{
    if (runSimulation || runSingleTimestep)
    {
        std::cout << "\nStart of frame: " << frameCount << std::endl;
        ++frameCount;

        double frameTime = 0.;
        while (frameTime < dt)
        {
            // Set CFL condition
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

            // Add gravity
            simulator->addForce(localDt, Vec3d(0, -9.8, 0));

            // Update moving solid
            movingSolidMesh.advectMesh(localDt, *solidVelocityField, IntegrationOrder::RK3);

            // Need moving solid volume to build sampled velocity
            LevelSet movingSolidSurface(xform, gridSize, 5);
            movingSolidSurface.initFromMesh(movingSolidMesh, false);

            VectorGrid<double> movingSolidVelocity(xform, gridSize, Vec3d::Zero(), VectorGridSettings::SampleType::STAGGERED);

            // Set moving solid velocity
            for (int axis : {0, 1, 2})
            {
                tbb::parallel_for(
                    tbb::blocked_range<int>(0, movingSolidVelocity.grid(axis).voxelCount(), tbbLightGrainSize),
                    [&](const tbb::blocked_range<int>& range) {
                        for (int faceIndex = range.begin(); faceIndex != range.end(); ++faceIndex)
                        {
                            Vec3i face = movingSolidVelocity.grid(axis).unflatten(faceIndex);

                            Vec3d worldPosition = movingSolidVelocity.indexToWorld(face.cast<double>(), axis);

                            if (movingSolidSurface.triLerp(worldPosition) < xform.dx())
                                movingSolidVelocity(face, axis) = (*solidVelocityField)(0, worldPosition)[axis];
                        }
                    });
            }

            LevelSet combinedSolidSurface(xform, gridSize, 5);
            combinedSolidSurface.setBackgroundNegative();
            combinedSolidSurface.initFromMesh(staticSolidMesh, false);
            combinedSolidSurface.unionSurface(movingSolidSurface);

            simulator->setSolidSurface(combinedSolidSurface);
            simulator->setSolidVelocity(movingSolidVelocity);

            // Projection set unfortunately includes viscosity at the moment
            simulator->runTimestep(localDt);

            simulator->unionLiquidSurface(seedLiquidSurface);

            frameTime += localDt;
        }

        runSingleTimestep = false;
        isDisplayDirty = true;
    }
    if (isDisplayDirty)
    {
        renderer->clear();

        simulator->drawLiquidSurface(*renderer);

        if (drawSolidBoundaries) simulator->drawSolidSurface(*renderer);

        if (drawLiquidVelocities) simulator->drawLiquidVelocity(*renderer, planeAxis, planePosition, 1);

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
    else if (key == 's')
        drawSolidBoundaries = !drawSolidBoundaries;
    else if (key == 'v')
        drawLiquidVelocities = !drawLiquidVelocities;

    isDisplayDirty = true;
}

int main(int argc, char** argv)
{
    double dx = .025;
    Vec3d topRightCorner(.75, 1., .5);
    Vec3d bottomLeftCorner(-.75, -1., -.5);
    gridSize = ((topRightCorner - bottomLeftCorner) / dx).cast<int>();
    xform = Transform(dx, bottomLeftCorner);
    Vec3d center = .5 * (topRightCorner + bottomLeftCorner);

    renderer = std::make_unique<Renderer>("Viscous liquid simulator", Vec2i::Constant(1000), Vec2d(bottomLeftCorner[0], bottomLeftCorner[1]),
                                          topRightCorner[1] - bottomLeftCorner[1], &argc, argv);

    camera = std::make_unique<Camera3D>(.5 * (topRightCorner + bottomLeftCorner), 2.5, 0., 0.);
    renderer->setCamera(camera.get());

    // Build static boundary geometry
    double solidThickness = 10;
    Vec3d solidScale = .5 * (topRightCorner - bottomLeftCorner) - Vec3d::Constant(solidThickness * dx);
    staticSolidMesh = makeCubeMesh(center, solidScale);
    staticSolidMesh.reverse();
    assert(staticSolidMesh.unitTestMesh());

    // Build moving boundary geometry

    movingSolidMesh = makeSphereMesh(center + Vec3d(.4, -.25, 0), .1, 3);

    LevelSet movingSolidSurface(xform, gridSize, 5);
    movingSolidSurface.initFromMesh(movingSolidMesh, false);

    LevelSet combinedSolidSurface(xform, gridSize, 5);
    combinedSolidSurface.setBackgroundNegative();
    combinedSolidSurface.initFromMesh(staticSolidMesh, false);
    combinedSolidSurface.unionSurface(movingSolidSurface);

    // Build seeding liquid geometry

    TriMesh seedLiquidMesh = makeCubeMesh(center + Vec3d(0, .4, 0), Vec3d(.075, .3, .075));
    assert(seedLiquidMesh.unitTestMesh());

    seedLiquidSurface = LevelSet(xform, gridSize, 5);
    seedLiquidSurface.initFromMesh(seedLiquidMesh, false);

    // Set up simulator
    simulator = std::make_unique<EulerianLiquidSimulator>(xform, gridSize, 5);
    simulator->setLiquidSurface(seedLiquidSurface);

    simulator->setSolidSurface(combinedSolidSurface);

    simulator->setLiquidSurface(seedLiquidSurface);

    simulator->setViscosity(1.);

    // Set up moving solid field
    solidVelocityField = std::make_unique<CircularField>(center, 1, Axis::ZAXIS);

    std::function<void()> displayFunc = display;
    renderer->setUserDisplay(displayFunc);

    std::function<void(unsigned char, int, int)> keyboardFunc = keyboard;
    renderer->setUserKeyboard(keyboardFunc);
    renderer->run();
}