#include <iostream>
#include <memory>

#include "Camera3D.h"
#include "InitialGeometry.h"
#include "Renderer.h"
#include "TriMesh.h"
#include "Utilities.h"

using namespace FluidSim3D;

static std::unique_ptr<Renderer> renderer;
static std::unique_ptr<Camera3D> camera;

static int geometryIndex = 0;
static bool geometryDirty = true;
static bool isDisplayDirty = true;

static TriMesh triMesh;

void keyboard(unsigned char key, int x, int y)
{
    if (key == 'g')
    {
        geometryIndex = (geometryIndex + 1) % 4;
        geometryDirty = true;
    }

    isDisplayDirty = true;
}

void display()
{
    if (geometryDirty)
    {
        if (geometryIndex == 0)
        {
            triMesh = makeDiamondMesh(Vec3d::Zero(), 1.);
        }
        else if (geometryIndex == 1)
        {
            triMesh = makeCubeMesh(Vec3d::Zero(), Vec3d::Ones());
        }
        else if (geometryIndex == 2)
        {
            triMesh = makeIcosahedronMesh();
        }
        else if (geometryIndex == 3)
        {
            triMesh = makeSphereMesh(Vec3d::Zero(), 1., 1);
        }
    }

    if (isDisplayDirty)
    {
        renderer->clear();

        triMesh.drawMesh(*renderer, true /* draw triangle faces */, Vec3d::Constant(.5), true/* don't render triangle normals*/,
                         Vec3d(1, 0, 0), false /*don't render mesh vertices*/, Vec3d::Zero(), true /* draw triangle edges*/,
                         Vec3d::Zero());

        isDisplayDirty = false;

        glutPostRedisplay();
    }
}

int main(int argc, char** argv)
{
    float dx = .005;
    Vec3d topRightCorner = Vec3d::Constant(1);
    Vec3d bottomLeftCorner = Vec3d::Constant(-1);
    Vec3i gridSize = Vec3i(((topRightCorner - bottomLeftCorner) / dx).cast<int>());

    renderer = std::make_unique<Renderer>("Level set test", Vec2i::Constant(1000), Vec2d(bottomLeftCorner[0], bottomLeftCorner[1]), topRightCorner[1] - bottomLeftCorner[1], &argc, argv);
    camera = std::make_unique<Camera3D>(.5 * (topRightCorner + bottomLeftCorner), 2.5, 0., 0.);
    renderer->setCamera(camera.get());

    std::function<void()> displayFunc = display;
    renderer->setUserDisplay(displayFunc);

    std::function<void(unsigned char, int, int)> keyboardFunc = keyboard;
    renderer->setUserKeyboard(keyboardFunc);
    renderer->run();
}