#include <memory>

#include "Camera3D.h"
#include "Renderer.h"
#include "ScalarGrid.h"
#include "Transform.h"
#include "UniformGrid.h"
#include "Utilities.h"
#include "Vec.h"
#include "VectorGrid.h"

using namespace FluidSim3D::RenderTools;
using namespace FluidSim3D::Utilities;

static std::unique_ptr<Renderer> renderer;
static std::unique_ptr<Camera3D> camera;

static bool isDisplayDirty = true;

static std::unique_ptr<ScalarGrid<float>> testScalarGrid;
static std::unique_ptr<VectorGrid<float>> testVectorGrid;

static float planePosition = .5;
static float planeDX = .05;
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
	float dx = .25;
	Vec3f topRightCorner(10);
	Vec3f bottomLeftCorner(-10);
	Vec3i gridSize = Vec3i((topRightCorner - bottomLeftCorner) / dx);
	Transform xform(dx, bottomLeftCorner);
	
	planeDX = std::min(float(1) / float(gridSize[0]), std::min(float(1) / float(gridSize[1]), float(1) / float(gridSize[2])));

	renderer = std::make_unique<Renderer>("Scalar grid test", Vec2i(1000), Vec2f(bottomLeftCorner[0], bottomLeftCorner[1]), topRightCorner[1] - bottomLeftCorner[1], &argc, argv);
	camera = std::make_unique<Camera3D>(.5 * (topRightCorner + bottomLeftCorner), 20., 0., 0.);
	renderer->setCamera(camera.get());
	
	testScalarGrid = std::make_unique<ScalarGrid<float>>(xform, gridSize);
	testVectorGrid = std::make_unique<VectorGrid<float>>(xform, gridSize, VectorGridSettings::SampleType::STAGGERED);

	tbb::parallel_for(tbb::blocked_range<int>(0, testScalarGrid->voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
	{
		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec3i cell = testScalarGrid->unflatten(cellIndex);

			Vec3f worldPoint = testScalarGrid->indexToWorld(Vec3f(cell));
			(*testScalarGrid)(cell) = sqrt(sqr(worldPoint[0]) + sqr(worldPoint[1]) + sqr(worldPoint[2])) - 5.;
		}
	});

	for (int axis : {0, 1, 2})
	{
		tbb::parallel_for(tbb::blocked_range<int>(0, testVectorGrid->grid(axis).voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
		{
			for (int faceIndex = range.begin(); faceIndex != range.end(); ++faceIndex)
			{
				Vec3i face = testVectorGrid->grid(axis).unflatten(faceIndex);

				Vec3f worldPoint = testVectorGrid->indexToWorld(Vec3f(face), axis);

				Vec3f gradVector = testScalarGrid->gradient(worldPoint);
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

