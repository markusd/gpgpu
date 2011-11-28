/**
 * @author Markus Holtermann
 * @date May 14, 2011
 * @file gui/renderwidget.cpp
 */

#ifdef __WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include <QtGui/QMessageBox>
#include <viewport.hpp>
#include <qutils.hpp>

#include <m3d/m3d.hpp>

#include <opengl/shader.hpp>
#include <opengl/texture.hpp>

#include <util/inputadapters.hpp>
#include <util/config.hpp>
#include <util/tostring.hpp>

#include <QtCore/QString>
#include <QtGui/QWheelEvent>
#include <QtGui/QWidget>
#include <QtOpenGL/QGLWidget>

#include <CL/cl.hpp>
#include <opencl/oclutil.hpp>

#include <iostream>


using namespace m3d;

#define WIDTH 768
#define HEIGHT 768

// OpenCL vars
/*
cl_platform_id clPlatform;
cl_device_id clDevice;
cl_context clContext;
cl_command_queue clQueue;
cl_program clProgram;
cl_kernel clKernel;
*/

cl::Platform clPlatform;
std::vector<cl::Device> clDevices;
cl::Context clContext;
cl::CommandQueue clQueue;
cl::Program clProgram;
cl::Kernel clKernel;
cl::Buffer clOut;
cl::Image2DGL clTexture;
cl::Buffer clPos;
cl::BufferRenderGL clRenderBuffer;

GLuint glFBO;
GLuint glRB;


Viewport::Viewport(QWidget* parent) :
	QGLWidget(parent)
{
	setFocusPolicy(Qt::WheelFocus);

	m_mouseAdapter.addListener(this);
	setMinimumWidth(WIDTH);
	setMinimumHeight(HEIGHT);
	setMaximumWidth(WIDTH);
	setMaximumHeight(HEIGHT);

	m_k = 1;
	m_iterations = 10;
	m_seedingAlgorithm = RANDOM;

	m_timer = new QTimer(this);
	connect(m_timer, SIGNAL(timeout()), this, SLOT(updateGL()));
}

void Viewport::initializeGL()
{
	GLenum err = glewInit();
	if (err != GLEW_OK) {
		util::ErrorAdapter::instance().displayErrorMessage("Could not initialize GLEW!");
		exit(1);
	}

	glShadeModel(GL_SMOOTH);
	glClearColor(1.0f, 1.0f, 1.0f, 0.0f);
	glClearDepth(1.0f);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);

	m_clock.reset();
}

void Viewport::resizeGL(int width, int height)
{
	glViewport(0, 0, width, height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glViewport(0.0f, 0.0f, width, height);
	glOrtho(0.0f, width, 0.0f, height, 0.0f, 32.0f);
	glMatrixMode(GL_MODELVIEW);

}

void Viewport::paintGL()
{
	static int frames = 0;
	static float fps_time = 0.0f;

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	float dt = m_clock.get();

	// input
	glColor3f(0.0f, 0.0f, 0.0f);
	glPointSize(2.0f);
	glBegin(GL_POINTS);
	for (int i = 0; i < m_input.size(); ++i) {
		glVertex2d(m_input[i].x * WIDTH, m_input[i].y * HEIGHT);
	}
	glEnd();

	// centroids
	glColor3f(1.0f, 0.0f, 0.0f);
	glPointSize(6.0f);
	glBegin(GL_POINTS);
	for (int i = 0; i < m_centroids.size(); ++i) {
		glVertex2d(m_centroids[i].x * WIDTH, m_centroids[i].y * HEIGHT);
	}
	glEnd();

	// seed
	glColor3f(0.0f, 0.0f, 1.0f);
	glPointSize(4.0f);
	glBegin(GL_POINTS);
	for (int i = 0; i < m_seed.size(); ++i) {
		glVertex2d(m_seed[i].x * WIDTH, m_seed[i].y * HEIGHT);
	}
	glEnd();

	// mean
	glColor3f(0.0f, 1.0f, 1.0f);
	glPointSize(5.0f);
	glBegin(GL_POINTS);
		glVertex2d(m_mean.x * WIDTH, m_mean.y * HEIGHT);
	glEnd();
	
	frames++;
	if (dt - fps_time >= 1.0f) {
		//std::cout << dt << ", " << fps_time << ", " << frames << std::endl;
		fps_time = dt;
		emit framesPerSecondChanged(frames);
		frames = 0;
	}
}

void Viewport::doCluster()
{
	//for (int i = 0; i < INPUT_SIZE; ++i) {
	//	input.push_back(Vec2d(((double)(rand()%10000)/10000.0), ((double)(rand()%10000)/10000.0)));
	//	std::cout << input[i] << std::endl;
	//}

	if (m_input.size() < m_k) {
		QMessageBox msgBox;
		msgBox.setText("Error: The input size is smaller than the number of clusters.");
		msgBox.exec();
		return;
	}

	switch (m_seedingAlgorithm) {
		case RANDOM:
			// hide mean
			m_mean = Vec2d(-100.0, -100.0);
			m_seed = random_seed(m_k, m_input);
			break;

		case MANUAL:
			break;

		case HARTIGAN_WONG:
			std::pair<Vec2d, std::vector<Vec2d> > seed_result = hartigan_wong(m_k, m_input);
			m_mean = seed_result.first;
			m_seed = seed_result.second;
			break;
	}
	
	m_centroids = kmeans(m_iterations, m_k, m_input, m_seed);

	//for (int i = 0; i < centroids.size(); ++i) {
	//	std::cout << centroids[i] << std::endl;
	//}
}

void Viewport::setSeedingAlgorithm(SeedingAlgorithm s)
{
	m_seedingAlgorithm = s;
}


void Viewport::mouseMove(int x, int y)
{
}

void Viewport::mouseButton(util::Button button, bool down, int x, int y)
{
	if (button == util::LEFT && !down) {
		m_input.push_back(Vec2d((double)x / (double)WIDTH, ((double)HEIGHT - (double)y) / (double)HEIGHT));
	}

	if (button == util::RIGHT && !down) {


	}
}

void Viewport::mouseDoubleClick(util::Button button, int x, int y)
{
}

void Viewport::mouseWheel(int delta)
{
}



void Viewport::keyPressEvent(QKeyEvent* event)
{
	m_keyAdapter.keyEvent(event);
}

void Viewport::keyReleaseEvent(QKeyEvent* event)
{
	m_keyAdapter.keyEvent(event);
}

void Viewport::mouseMoveEvent(QMouseEvent* event)
{
	m_mouseAdapter.mouseEvent(event);
}

void Viewport::mousePressEvent(QMouseEvent* event)
{
	m_mouseAdapter.mouseEvent(event);
}

void Viewport::mouseReleaseEvent(QMouseEvent* event)
{
	m_mouseAdapter.mouseEvent(event);
}

void Viewport::wheelEvent(QWheelEvent* event)
{
	m_mouseAdapter.mouseWheelEvent(event);
}

void Viewport::mouseDoubleClickEvent(QMouseEvent* event)
{
	m_mouseAdapter.mouseEvent(event);
}