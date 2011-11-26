#ifndef MAINWINDOW_HPP_
#define MAINWINDOW_HPP_

#include <toolbox.hpp>
#include <viewport.hpp>
#include <QtGui/QMainWindow>

class QLabel;

class MainWindow: public QMainWindow {
Q_OBJECT
public:
	MainWindow(QApplication* app);

private slots:
	void updateFramesPerSecond(int frames);

private:
	Viewport* m_viewport;
	ToolBox* m_toolBox;
	QLabel* m_framesPerSec;
};


#endif /* MAINWINDOW_HPP_ */
