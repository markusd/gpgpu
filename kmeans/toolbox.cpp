#include <toolbox.hpp>

#include <QtGui/QVBoxLayout>
#include <QtGui/QGroupBox>
#include <QtGui/QRadioButton>
#include <QtGui/QPushButton>
#include <QSpinBox>

#include <iostream>


ToolBox::ToolBox(QWidget *parent)
{
	setMaximumWidth(250);

	QVBoxLayout* layout = new QVBoxLayout();

	/* Seeding group */
	m_seedingGroup = new QGroupBox("Seeding Algorithm");

	m_randomButton = new QRadioButton("Random");
	m_hartiganWongButton = new QRadioButton("Hartigan Wong");
	m_randomButton->setChecked(true);

	connect(m_randomButton, SIGNAL(clicked()), this, SLOT(seedingAlgorithmChanged()));
	connect(m_hartiganWongButton, SIGNAL(clicked()), this, SLOT(seedingAlgorithmChanged()));


	QVBoxLayout* group_layout = new QVBoxLayout();
	group_layout->addWidget(m_randomButton);
	group_layout->addWidget(m_hartiganWongButton);
	group_layout->addStretch(1);

	m_seedingGroup->setLayout(group_layout);

	/* Parameters goup */
	m_parametersGroup = new QGroupBox("Parameters");

	m_doClusterButton = new QPushButton("Cluster!");
	m_kBox = new QSpinBox();
	m_kBox->setSuffix(" clusters");
	m_kBox->setRange(1, 100000);

	m_iterationsBox = new QSpinBox();
	m_iterationsBox->setSuffix(" iteratios");
	m_iterationsBox->setRange(1, 1000000);

	group_layout = new QVBoxLayout();
	group_layout->addWidget(m_kBox);
	group_layout->addWidget(m_iterationsBox);
	group_layout->addWidget(m_doClusterButton);
	group_layout->addStretch(1);

	m_parametersGroup->setLayout(group_layout);

	layout->addWidget(m_seedingGroup);
	layout->addWidget(m_parametersGroup);
	layout->addStretch(1);
	setLayout(layout);
}

void ToolBox::seedingAlgorithmChanged()
{
	SeedingAlgorithm s = RANDOM;
	if (m_hartiganWongButton->isChecked())
		s = HARTIGAN_WONG;
	
	emit setSeedingAlgorithm(s);
}
