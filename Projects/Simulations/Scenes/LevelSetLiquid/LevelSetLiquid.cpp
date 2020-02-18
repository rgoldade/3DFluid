#include <memory>

#include "Camera3D.h"
#include "EulerianLiquidSimulator.h"
#include "InitialGeometry.h"
#include "LevelSet.h"
#include "Renderer.h"
#include "ScalarGrid.h"
#include "Transform.h"
#include "TriMesh.h"
#include "Utilities.h"
#include "Vec.h"
#include "VectorGrid.h"

using namespace FluidSim3D::RenderTools;
using namespace FluidSim3D::SurfaceTrackers;
using namespace FluidSim3D::SimTools;
using namespace FluidSim3D::Utilities;

static std::unique_ptr<Renderer> renderer;
static std::unique_ptr<Camera3D> camera;

static int frameCount = 0;
static bool runSimulation = false;
static bool runSingleTimestep = false;
static bool isDisplayDirty = true;

static bool drawSolidBoundaries = true;
static bool drawLiquidVelocities = true;

static float planePosition = .5;
static float planeDX = .05;
static Axis planeAxis = Axis::ZAXIS;

static std::unique_ptr<EulerianLiquidSimulator> simulator;

static LevelSet seedSurface;

static Transform xform;

static float seedTime = 0;
static float seedPeriod = 1;
static constexpr float dt = 1. / 30.;

void display()
{
	if (runSimulation || runSingleTimestep)
	{
		std::cout << "\nStart of frame: " << frameCount << std::endl;
		++frameCount;

		float frameTime = 0.;
		while (frameTime < dt)
		{
			// Set CFL condition
			float speed = simulator->maxVelocityMagnitude();
			float localDt = dt - frameTime;
			assert(localDt >= 0);

			if (speed > 1E-6)
			{
				float cflDt = 3. * xform.dx() / speed;
				if (localDt > cflDt)
				{
					localDt = cflDt;
					std::cout << "\n  Throttling frame with substep: " << localDt << "\n" << std::endl;
				}
			}

			if (localDt <= 0)
				break;

			if (seedTime > seedPeriod)
			{
				simulator->unionLiquidSurface(seedSurface);
				seedTime = 0;
			}

			// Add gravity
			simulator->addForce(localDt, Vec3f(0, -9.8, 0));

			// Projection set unfortunately includes viscosity at the moment
			simulator->runTimestep(localDt);

			seedTime += localDt;
			frameTime += localDt;
		}

		runSingleTimestep = false;
		isDisplayDirty = true;

	}
	if (isDisplayDirty)
	{
		renderer->clear();

		simulator->drawLiquidSurface(*renderer);

		if (drawSolidBoundaries)
			simulator->drawSolidSurface(*renderer);

		if (drawLiquidVelocities)
			simulator->drawLiquidVelocity(*renderer, planeAxis, planePosition, 1);

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

int main(int argc, char **argv)
{
	float dx = .05;
	float solidSphereRadius = 2;
	Vec3f topRightCorner(solidSphereRadius + 15 * dx);
	Vec3f bottomLeftCorner(-solidSphereRadius - 15 * dx);
	Vec3i gridSize = Vec3i((topRightCorner - bottomLeftCorner) / dx);
	xform = Transform(dx, bottomLeftCorner);
	Vec3f center = .5 * (topRightCorner + bottomLeftCorner);

	renderer = std::make_unique<Renderer>("Levelset liquid simulator", Vec2i(1000), Vec2f(bottomLeftCorner[0], bottomLeftCorner[1]), topRightCorner[1] - bottomLeftCorner[1], &argc, argv);
	camera = std::make_unique<Target3D>(.5 * (topRightCorner + bottomLeftCorner), 6, 0., 0.);
	renderer->setCamera(camera.get());

	// Build initial liquid geometry

	TriMesh liquidMesh = makeSphereMesh(center - Vec3f(0, .65, 0), 1, .5 * dx);
	assert(liquidMesh.unitTestMesh());

	LevelSet liquidSurface(xform, gridSize, 5);
	liquidSurface.initFromMesh(liquidMesh, false);

	// Build seed liquid geometry

	Vec3f seedCenter = center + Vec3f(.6, .6, 0);
	TriMesh seedMesh = makeCubeMesh(seedCenter, Vec3f(.5));

	seedSurface = LevelSet(xform, gridSize, 5);
	seedSurface.initFromMesh(seedMesh, false);

	// Build solid boundary

	TriMesh solidMesh = makeSphereMesh(center, 2, .5 * dx);
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