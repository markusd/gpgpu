#ifndef VIEWPORT_HPP_
#define VIEWPORT_HPP_

#include <kmeans.hpp>
#include <util/clock.hpp>
#include <qutils.hpp>
#include <opengl/texture.hpp>
#include <opengl/shader.hpp>
#include <GL/glew.h>
#include <QtOpenGL/QGLWidget>
#include <QtCore/QTimer>
#include <m3d/m3d.hpp>

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
	QTimer* m_timer;
	
	QtMouseAdapter m_mouseAdapter;
	QtKeyAdapter m_keyAdapter;

	// input parameters
	int m_k;
	int m_iterations;
	int m_runs;
	SeedingAlgorithm m_seedingAlgorithm;


	std::vector<Vec2d> m_input, m_seed, m_centroids;
	Vec2d m_mean;

	virtual void mouseMove(int x, int y);
	virtual void mouseButton(util::Button button, bool down, int x, int y);
	virtual void mouseDoubleClick(util::Button button, int x, int y);
	virtual void mouseWheel(int delta);
public slots:
	void setSeedingAlgorithm(SeedingAlgorithm s);
	void setK(int i) { m_k = i; };
	void setIterations(int i) { m_iterations = i; };
	void setRuns(int i) { m_runs = i; };
	void doCluster();
	void findK();
	void doClear();
	void doClearSeed();
signals:
	void framesPerSecondChanged(int);
};


#endif /* RENDERWIDGET_HPP_ */
