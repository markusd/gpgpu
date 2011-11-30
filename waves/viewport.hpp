#ifndef VIEWPORT_HPP_
#define VIEWPORT_HPP_

#include <waves.hpp>
#include <util/clock.hpp>
#include <qutils.hpp>
#include <opengl/texture.hpp>
#include <opengl/shader.hpp>
#include <GL/glew.h>
#include <QtOpenGL/QGLWidget>
#include <QtCore/QTimer>
#include <m3d/m3d.hpp>

#ifdef USE_CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#endif

class QTimer;

class Viewport: public QGLWidget, public util::MouseListener {
Q_OBJECT
public:
	Viewport(QWidget* parent = 0);

protected:
	virtual void initializeGL();
	virtual void resizeGL(int width, int height);
	virtual void paintGL();

	virtual void keyPressEvent(QKeyEvent* event);
	virtual void keyReleaseEvent(QKeyEvent* event);
	virtual void mouseMoveEvent(QMouseEvent* event);
	virtual void mousePressEvent(QMouseEvent* event);
	virtual void mouseReleaseEvent(QMouseEvent* event);
	virtual void wheelEvent(QWheelEvent* event);
	virtual void mouseDoubleClickEvent(QMouseEvent* event);

public:
	util::Clock m_clock;

	ComputationMode m_mode;
	GLuint m_displayList;
	ogl::Texture m_waveData;
	ogl::Texture m_texture;
	ogl::Shader m_shader;

	std::vector<Wave> m_waves;

	float m_xoffs;
	float m_yoffs;

#ifndef CPU_FLOAT
	unsigned char* m_textureData;
#else
	float* m_textureData;
#endif
	float* m_clTextureData;

#ifdef USE_CUDA
	ogl::Texture m_cuTexture;
	float* m_cuHostTextureData;
	float* m_cuTextureData;
	float* m_cuPosData;
#ifdef CUDA_GL_INTEROP
	cudaGraphicsResource* m_cuTextureResource;
#endif
#endif

	QTimer* m_timer;
	
	QtMouseAdapter m_mouseAdapter;
	QtKeyAdapter m_keyAdapter;

	void createKernel();
	void createShader();
	void createTextureCPU(float dt);
	void createTextureOpenCL(float dt);
#ifdef USE_CUDA
	void createTextureCuda(float dt);
#endif

	virtual void mouseMove(int x, int y);
	virtual void mouseButton(util::Button button, bool down, int x, int y);
	virtual void mouseDoubleClick(util::Button button, int x, int y);
	virtual void mouseWheel(int delta);
public slots:
	void setComputationMode(ComputationMode mode);
signals:
	void framesPerSecondChanged(int);
};


#endif /* RENDERWIDGET_HPP_ */
