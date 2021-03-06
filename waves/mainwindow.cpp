#include <mainwindow.hpp>

#include <QtGui/QApplication>
#include <QtCore/QList>
#include <QtCore/QTextCodec>
#include <QtGui/QLabel>
#include <QtGui/QSplitter>
#include <QtGui/QStatusBar>

MainWindow::MainWindow(QApplication* app)
{
	QTextCodec::setCodecForCStrings(QTextCodec::codecForName("UTF-8"));
	this->setWindowTitle("waves");

	// create status bar
	m_framesPerSec = new QLabel("nA");
	m_framesPerSec->setMinimumSize(m_framesPerSec->sizeHint());
	m_framesPerSec->setAlignment(Qt::AlignLeft);
	m_framesPerSec->setToolTip("Current frames per second not yet initialized.");
	statusBar()->addWidget(m_framesPerSec);

	m_viewport = new Viewport(this);
	m_viewport->show();

	m_toolBox = new ToolBox();
	
	QSplitter* splitter = new QSplitter(Qt::Horizontal);
	splitter->insertWidget(0, m_toolBox);
	splitter->insertWidget(1, m_viewport);

	QList<int> sizes;
	sizes.append(200);
	sizes.append(1300);

	splitter->setSizes(sizes);
	splitter->setStretchFactor(0, 1);
	splitter->setStretchFactor(1, 1);
	splitter->setChildrenCollapsible(false);

	setCentralWidget(splitter);

	connect(m_viewport, SIGNAL(framesPerSecondChanged(int)), this, SLOT(updateFramesPerSecond(int)));
	connect(m_toolBox, SIGNAL(setComputationMode(ComputationMode)), m_viewport, SLOT(setComputationMode(ComputationMode)));
	
	show();
this->adjustSize();
	m_viewport->updateGL();
	m_viewport->m_timer->start();
}

void MainWindow::updateFramesPerSecond(int frames)
{
	m_framesPerSec->setText(QString("%1 fps   ").arg(frames));
}
