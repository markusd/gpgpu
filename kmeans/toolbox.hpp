#ifndef TOOLBOX_HPP_
#define TOOLBOX_HPP_

#include <kmeans.hpp>
#include <QtGui/QWidget>


class QGroupBox;
class QRadioButton;

class ToolBox: public QWidget {
Q_OBJECT
public:
	ToolBox(QWidget* parent = 0);

public slots:
	//void typeChanged();

signals:
	//void setComputationMode(ComputationMode mode);

private:
	/*
	QGroupBox* m_typeGroup;
	QRadioButton* m_cpuButton;
	QRadioButton* m_shaderButton;
	QRadioButton* m_openclButton;*/
};

#endif /* TOOLBOX_HPP_ */