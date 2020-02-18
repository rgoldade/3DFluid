#ifndef LIBRARY_CAMERA3D_H
#define LIBRARY_CAMERA3D_H

#include <GL/glut.h>

#include "Utilities.h"
#include "Vec.h"

///////////////////////////////////
//
// Camera3D.h/cpp
// Ryan Goldade 2017
//
// Modified from gluvi.h/cpp
// in the El Topo library
//
////////////////////////////////////

namespace FluidSim3D::RenderTools
{

using namespace Utilities;

class Camera3D
{
public:
	Camera3D(const Vec3f& target = Vec3f(0),
				float targetDistance = 1,
				float heading = 0,
				float pitch = 0,
				float fieldOfView = 45,
				float nearClippingDistance = 0.01,
				float farClippingDistance = 100.);

	void mouse(int button, int state, int x, int y);
	void drag(int x, int y);

	void reset();
	void transform(const Vec2i& windowSize);

private:
	enum class MouseAction { INACTIVE, ROTATE, TRUCK, DOLLY };

	Vec3f myTarget, myDefaultTarget;
	float myDistance, myDefaultDistance;
	float myHeading, myDefaultHeading;
	float myPitch, myDefaultPitch;
	float myFieldOfView;

	float myNearClippingDistance, myFaceClippingDistance;

	MouseAction myMouseAction;
	Vec2i myOldMouseCoord;
};

}
#endif