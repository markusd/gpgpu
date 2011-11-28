#ifndef TOOLBOX_HPP_
#define TOOLBOX_HPP_

#include <kmeans.hpp>
#include <QtGui/QWidget>


class QGroupBox;
class QRadioButton;
class QPushButton;
class QSpinBox;

class ToolBox: public QWidget {
Q_OBJECT
public:
	ToolBox(QWidget* parent = 0);

public slots:
	void seedingAlgorithmChanged();

signals:
	void setSeedingAlgorithm(SeedingAlgorithm s);

public:
	QGroupBox* m_seedingGroup;
	QRadioButton* m_randomButton;
	QRadioButton* m_hartiganWongButton;

	QGroupBox* m_parametersGroup;
	QSpinBox* m_kBox;
	QSpinBox* m_iterationsBox;
	QPushButton* m_doClusterButton;
};

#endif /* TOOLBOX_HPP_ */