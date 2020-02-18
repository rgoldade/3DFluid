#include "Camera3D.h"

namespace FluidSim3D::RenderTools
{

Camera3D::Camera3D(const Vec3f& target,
					float targetDistance,
					float heading,
					float pitch,
					float fieldOfView,
					float nearClippingDistance,
					float farClippingDistance)
	: myTarget(target)
	, myDefaultTarget(target)
	, myDistance(targetDistance)
	, myDefaultDistance(targetDistance)
	, myHeading(heading)
	, myDefaultHeading(heading)
	, myPitch(pitch)
	, myDefaultPitch(pitch)
	, myFieldOfView(fieldOfView)
	, myNearClippingDistance(nearClippingDistance)
	, myFaceClippingDistance(farClippingDistance)
	{}

void Camera3D::mouse(int button, int state, int x, int y)
{
	if (state == GLUT_UP)
		myMouseAction = MouseAction::INACTIVE;
	else if (button == GLUT_LEFT_BUTTON)
		myMouseAction = MouseAction::ROTATE;
	else if (button == GLUT_MIDDLE_BUTTON)
		myMouseAction = MouseAction::TRUCK;
	else if (button == GLUT_RIGHT_BUTTON)
		myMouseAction = MouseAction::DOLLY;

	myOldMouseCoord = Vec2i(x, y);
}

void Camera3D::drag(int x, int y)
{
	switch (myMouseAction)
	{
	case MouseAction::INACTIVE:
		return;
	case MouseAction::ROTATE:
		myHeading += 0.007 * (myOldMouseCoord[0] - x);

		if (myHeading < -PI) myHeading += 2.0 * PI;
		else if (myHeading > PI) myHeading -= 2.0 * PI;

		myPitch += 0.007 * (myOldMouseCoord[1] - y);

		if (myPitch < -0.5 * PI) myPitch = -0.5 * PI;
		else if (myPitch > 0.5 * PI) myPitch = 0.5 * PI;

		break;
	case MouseAction::TRUCK:
		myTarget[0] += (0.002 * myDistance) * std::cos(myHeading) * (myOldMouseCoord[0] - x);
		myTarget[1] -= (0.002 * myDistance) * (myOldMouseCoord[1] - y);
		myTarget[2] -= (0.002 * myDistance) * std::sin(myHeading) * (myOldMouseCoord[0] - x);
		break;
	case MouseAction::DOLLY:
		myDistance *= std::pow(1.01, myOldMouseCoord[1] - y + x - myOldMouseCoord[0]);
		break;
	}

	myOldMouseCoord[0] = x;
	myOldMouseCoord[1] = y;

	glutPostRedisplay();
}

void Camera3D::reset()
{
	myTarget = myDefaultTarget;
	myDistance = myDefaultDistance;
	myHeading = myDefaultHeading;
	myPitch = myDefaultPitch;
}

void Camera3D::transform(const Vec2i& windowSize)
{
	glViewport(0, 0, GLsizei(windowSize[0]), GLsizei(windowSize[1]));

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(myFieldOfView, windowSize[0] / float(windowSize[1]), myNearClippingDistance, myFaceClippingDistance);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0, 0, -myDistance); // translate target dist away in the z direction
	glRotatef(-180.f / PI * myPitch, 1, 0, 0); // rotate pitch in the yz plane
	glRotatef(-180.f / PI * myHeading, 0, 1, 0); // rotate heading in the xz plane
	glTranslatef(-myTarget[0], -myTarget[1], -myTarget[2]); // translate target to origin
}

}