#include <memory>

#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"

#include "Camera3D.h"
#include "Renderer.h"
#include "ScalarGrid.h"
#include "Transform.h"
#include "UniformGrid.h"
#include "Utilities.h"
#include "VectorGrid.h"

using namespace FluidSim3D;

static std::unique_ptr<Renderer> renderer;
static std::unique_ptr<Camera3D> camera;

static bool isDisplayDirty = true;

static std::unique_ptr<ScalarGrid<double>> testScalarGrid;
static std::unique_ptr<VectorGrid<double>> testVectorGrid;

static double planePosition = .5;
static double planeDX = .05;
static Axis planeAxis = Axis::ZAXIS;

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

    isDisplayDirty = true;
}

void display()
{
    if (isDisplayDirty)
    {
        renderer->clear();
        testScalarGrid->drawGridPlane(*renderer, planeAxis, planePosition);
        testScalarGrid->drawSupersampledValuesPlane(*renderer, planeAxis, planePosition, .5, 3, 5);
        testVectorGrid->drawSamplePointVectors(*renderer, planeAxis, planePosition);

        glutPostRedisplay();
    }
}

int main(int argc, char** argv)
{
    double dx = .25;
    Vec3d topRightCorner = Vec3d::Constant(10);
    Vec3d bottomLeftCorner = Vec3d::Constant(-10);
    Vec3i gridSize = ((topRightCorner - bottomLeftCorner) / dx).cast<int>();
    Transform xform(dx, bottomLeftCorner);

    planeDX = std::min(double(1) / double(gridSize[0]), std::min(double(1) / double(gridSize[1]), double(1) / double(gridSize[2])));

    renderer = std::make_unique<Renderer>("Scalar grid test", Vec2i::Constant(1000), Vec2d(bottomLeftCorner[0], bottomLeftCorner[1]),
                                   topRightCorner[1] - bottomLeftCorner[1], &argc, argv);
    camera = std::make_unique<Camera3D>(.5 * (topRightCorner + bottomLeftCorner), 20., 0., 0.);
    renderer->setCamera(camera.get());

    testScalarGrid = std::make_unique<ScalarGrid<double>>(xform, gridSize);
    testVectorGrid = std::make_unique<VectorGrid<double>>(xform, gridSize, VectorGridSettings::SampleType::STAGGERED);

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

    std::function<void()> displayFunc = display;
    renderer->setUserDisplay(displayFunc);

    std::function<void(unsigned char, int, int)> keyboardFunc = keyboard;
    renderer->setUserKeyboard(keyboardFunc);

    renderer->run();
}
