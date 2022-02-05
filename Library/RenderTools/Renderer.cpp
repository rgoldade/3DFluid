#include "Renderer.h"

namespace FluidSim3D
{
// Helper struct because glut is a pain.
// This is probably a very bad design choice
// but glut doesn't make life easy.
class GlutHelper
{
public:
    static void init(Renderer* render)
    {
        // Safety check to only ever take one instance
        // of a Renderer
        if (!myRenderer) myRenderer = render;

        glutReshapeFunc(reshape);
        glutDisplayFunc(display);
        glutMouseFunc(mouse);
        glutMotionFunc(drag);
        glutKeyboardFunc(keyboard);
    }

private:
    static void reshape(int w, int h)
    {
        if (myRenderer) myRenderer->reshape(w, h);
    }

    static void display()
    {
        if (myRenderer) myRenderer->display();
    }

    static void mouse(int button, int state, int x, int y)
    {
        if (myRenderer) myRenderer->mouse(button, state, x, y);
    }

    static void drag(int x, int y)
    {
        if (myRenderer) myRenderer->drag(x, y);
    }

    static void keyboard(unsigned char key, int x, int y)
    {
        if (myRenderer) myRenderer->keyboard(key, x, y);
    }

    static Renderer* myRenderer;
};

Renderer* GlutHelper::myRenderer;

Renderer::Renderer(const char* title, const Vec2i& windowSize, const Vec2d& screenOrigin, double screenHeight, int* argc,
                   char** argv)
    : myWindowSize(windowSize),
      myCurrentScreenOrigin(screenOrigin),
      myCurrentScreenHeight(screenHeight),
      myDefaultScreenOrigin(screenOrigin),
      myDefaultScreenHeight(screenHeight),
      myMouseAction(MouseAction::INACTIVE),
      myMouseHasMoved(false)
{
    glutInit(argc, argv);

    // TODO: review if these flags are needed
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_ALPHA | GLUT_DEPTH | GLUT_STENCIL);
    glutInitWindowSize(myWindowSize[0], myWindowSize[1]);
    glutCreateWindow(title);

    GlutHelper::init(this);

    glEnable(GL_DEPTH_TEST);
    glClearDepth(1);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    glBlendFunc(GL_SRC_ALPHA_SATURATE, GL_ONE);
    glClearColor(1, 1, 1, 1);
}

void Renderer::display()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // draw the scene
    if (myCamera) myCamera->transform(myWindowSize);

    if (myUserDisplayFunction) myUserDisplayFunction();

    // Draw primitives that were added to the renderer
    drawPrimitives();

    glutSwapBuffers();
    glutPostRedisplay();
}

void Renderer::mouse(int button, int state, int x, int y)
{
    if (myCamera)
        myCamera->mouse(button, state, x, y);
    else if (myUserMouseClickFunction)
        myUserMouseClickFunction(button, state, x, y);
}

void Renderer::drag(int x, int y)
{
    if (myCamera)
        myCamera->drag(x, y);
    else if (myUserMouseDragFunction)
        myUserMouseDragFunction(x, y);
}

void Renderer::keyboard(unsigned char key, int x, int y)
{
    if (myUserKeyboardFunction) myUserKeyboardFunction(key, x, y);

    // r triggers return to defaults
    if (key == 'r')
    {
        myCurrentScreenOrigin = myDefaultScreenOrigin;
        myCurrentScreenHeight = myDefaultScreenHeight;
        myCamera->reset();
        glutPostRedisplay();
    }
}

void Renderer::reshape(int width, int height)
{
    myWindowSize[0] = width;
    myWindowSize[1] = height;
    glutPostRedisplay();
}

void Renderer::setUserMouseClick(const std::function<void(int, int, int, int)>& mouseClickFunction)
{
    myUserMouseClickFunction = mouseClickFunction;
}

void Renderer::setUserKeyboard(const std::function<void(unsigned char, int, int)>& keyboardFunction)
{
    myUserKeyboardFunction = keyboardFunction;
}

void Renderer::setUserMouseDrag(const std::function<void(int, int)>& mouseDragFunction)
{
    myUserMouseDragFunction = mouseDragFunction;
}

void Renderer::setUserDisplay(const std::function<void()>& displayFunction) { myUserDisplayFunction = displayFunction; }

// These helpers make it easy to render out basic primitives without having to write
// a custom loop outside of this class
void Renderer::addPoint(const Vec3d& point, const Vec3d& colour, double size)
{
    myPoints.emplace_back(1, point);
    myPointColours.push_back(colour);
    myPointSizes.push_back(size);
}

void Renderer::addPoints(const VecVec3d& points, const Vec3d& colour, double size)
{
    myPoints.push_back(points);
    myPointColours.push_back(colour);
    myPointSizes.push_back(size);
}

void Renderer::addLine(const Vec3d& startingPoint, const Vec3d& endingPoint, const Vec3d& colour, double lineWidth)
{
    myLineStartingPoints.emplace_back(1, startingPoint);
    myLineEndingPoints.emplace_back(1, endingPoint);

    myLineColours.push_back(colour);
    myLineWidths.push_back(lineWidth);
}

void Renderer::addLines(const VecVec3d& startingPoints, const VecVec3d& endingPoints,
                        const Vec3d& colour, double lineWidth)
{
    assert(startingPoints.size() == endingPoints.size());
    myLineStartingPoints.push_back(startingPoints);
    myLineEndingPoints.push_back(endingPoints);

    myLineColours.push_back(colour);
    myLineWidths.push_back(lineWidth);
}

void Renderer::addTriFaces(const VecVec3d& vertices, const VecVec3d& normals,
                           const VecVec3i& faces, const Vec3d& faceColour)
{
    assert(vertices.size() == normals.size());

    myTriVertices.push_back(vertices);
    myTriNormals.push_back(normals);

    myTriFaces.push_back(faces);
    myTriFaceColours.push_back(faceColour);
}

void Renderer::drawPrimitives() const
{
    glDisable(GL_LIGHTING);
    glDepthFunc(GL_LEQUAL);

    assert(myPoints.size() == myPointColours.size() && myPoints.size() == myPointSizes.size());

    for (int pointListIndex = 0; pointListIndex < myPoints.size(); ++pointListIndex)
    {
        Vec3t<GLfloat> pointColour = myPointColours[pointListIndex].cast<GLfloat>();
        glColor3f(pointColour[0], pointColour[1], pointColour[2]);
        glPointSize(GLfloat(myPointSizes[pointListIndex]));

        glBegin(GL_POINTS);

        for (int pointIndex = 0; pointIndex < myPoints[pointListIndex].size(); ++pointIndex)
        {
            Vec3t<GLfloat> point = myPoints[pointListIndex][pointIndex].cast<GLfloat>();
            glVertex3f(point[0], point[1], point[2]);
        }

        glEnd();
    }

    assert(myLineStartingPoints.size() == myLineEndingPoints.size() &&
           myLineStartingPoints.size() == myLineColours.size() && myLineStartingPoints.size() == myLineWidths.size());

    for (int lineListIndex = 0; lineListIndex < myLineStartingPoints.size(); ++lineListIndex)
    {
        Vec3t<GLfloat> lineColour = myLineColours[lineListIndex].cast<GLfloat>();
        glColor3f(lineColour[0], lineColour[1], lineColour[2]);

        glLineWidth(GLfloat(myLineWidths[lineListIndex]));

        glBegin(GL_LINES);

        assert(myLineStartingPoints[lineListIndex].size() == myLineEndingPoints[lineListIndex].size());

        for (int lineIndex = 0; lineIndex < myLineStartingPoints[lineListIndex].size(); ++lineIndex)
        {
            Vec3t<GLfloat> startPoint = myLineStartingPoints[lineListIndex][lineIndex].cast<GLfloat>();
            glVertex3f(startPoint[0], startPoint[1], startPoint[2]);

            Vec3t<GLfloat> endPoint = myLineEndingPoints[lineListIndex][lineIndex].cast<GLfloat>();
            glVertex3f(endPoint[0], endPoint[1], endPoint[2]);
        }

        glEnd();
    }

    // Render tris
    // glEnable(GL_LIGHTING);
    // glShadeModel(GL_SMOOTH);

    //// Set simple lights and materials
    // set_generic_lights();
    // set_generic_material(1.0f, 1.0f, 1.0f, GL_FRONT);   // exterior surface colour
    // set_generic_material(1.0f, 1.0f, 1.0f, GL_BACK);
    //

    // TEMP
    glDisable(GL_LIGHTING);
    glColor3d(1, 1, 1);

    // Cull back faces
    // glEnable(GL_CULL_FACE);

    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(1.0f, 1.0f);  //  allow the wireframe to show through

    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);

    assert(myTriVertices.size() == myTriNormals.size() && myTriVertices.size() == myTriFaces.size() &&
           myTriVertices.size() == myTriFaceColours.size());

    for (int triListIndex = 0; triListIndex != myTriFaceColours.size(); ++triListIndex)
    {
        Vec3t<GLfloat> triFaceColour = myTriFaceColours[triListIndex].cast<GLfloat>();
        glColor3f(triFaceColour[0], triFaceColour[1], triFaceColour[2]);

        glBegin(GL_TRIANGLES);

        assert(myTriVertices[triListIndex].size() == myTriNormals[triListIndex].size());

        for (int triIndex = 0; triIndex != myTriFaces[triListIndex].size(); ++triIndex)
        {
            for (int pointIndex : {0, 1, 2})
            {
                int trianglePointIndex = myTriFaces[triListIndex][triIndex][pointIndex];

                assert(trianglePointIndex >= 0 && trianglePointIndex < myTriVertices[triListIndex].size());

                Vec3t<GLfloat> vertexPoint = myTriVertices[triListIndex][trianglePointIndex].cast<GLfloat>();
                glVertex3f(vertexPoint[0], vertexPoint[1], vertexPoint[2]);

                Vec3t<GLfloat> normal = myTriNormals[triListIndex][trianglePointIndex].cast<GLfloat>();
                glNormal3f(normal[0], normal[1], normal[2]);
            }
        }

        glEnd();
    }

    glDisable(GL_POLYGON_OFFSET_FILL);
    glDisable(GL_LIGHTING);
}

void Renderer::clear()
{
    myPoints.clear();
    myPointColours.clear();
    myPointSizes.clear();

    myLineStartingPoints.clear();
    myLineEndingPoints.clear();
    myLineColours.clear();
    myLineWidths.clear();

    myTriVertices.clear();
    myTriNormals.clear();
    myTriFaces.clear();
    myTriFaceColours.clear();
}

void Renderer::run() { glutMainLoop(); }

void Renderer::setGenericLights(void) const
{
    glEnable(GL_LIGHTING);
    {
        GLfloat ambient[4] = {.3f, .3f, .3f, 1.0f};
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, ambient);
    }
    {
        GLfloat color[4] = {.4f, .4f, .4f, 1.0f};
        glLightfv(GL_LIGHT0, GL_DIFFUSE, color);
        glLightfv(GL_LIGHT0, GL_SPECULAR, color);
        glEnable(GL_LIGHT0);
    }
    {
        GLfloat color[4] = {.4f, .4f, .4f, 1.0f};
        glLightfv(GL_LIGHT1, GL_DIFFUSE, color);
        glLightfv(GL_LIGHT1, GL_SPECULAR, color);
        glEnable(GL_LIGHT1);
    }
    {
        GLfloat color[4] = {.2f, .2f, .2f, 1.0f};
        glLightfv(GL_LIGHT2, GL_DIFFUSE, color);
        glLightfv(GL_LIGHT2, GL_SPECULAR, color);
        glEnable(GL_LIGHT2);
    }
}

void Renderer::setGenericMaterial(double r, double g, double b, GLenum face) const
{
    GLfloat ambient[4], diffuse[4], specular[4];

    ambient[0] = GLfloat(0.1 * r + 0.03);
    ambient[1] = GLfloat(0.1 * g + 0.03);
    ambient[2] = GLfloat(0.1 * b + 0.03);
    ambient[3] = GLfloat(1.0);

    diffuse[0] = GLfloat(0.7 * r);
    diffuse[1] = GLfloat(0.7 * g);
    diffuse[2] = GLfloat(0.7 * b);
    diffuse[3] = GLfloat(1.0);

    specular[0] = GLfloat(0.1 * r + 0.1);
    specular[1] = GLfloat(0.1 * g + 0.1);
    specular[2] = GLfloat(0.1 * b + 0.1);
    specular[3] = GLfloat(1.0);

    glMaterialfv(face, GL_AMBIENT, ambient);
    glMaterialfv(face, GL_DIFFUSE, diffuse);
    glMaterialfv(face, GL_SPECULAR, specular);
    glMaterialf(face, GL_SHININESS, 32);
}

}  // namespace FluidSim3D::RenderTools