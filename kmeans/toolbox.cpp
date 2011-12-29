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
	m_manualButton = new QRadioButton("Manual");
	m_hartiganWongButton = new QRadioButton("Hartigan Wong");
	m_astrahanButton = new QRadioButton("Astrahan");
	m_randomButton->setChecked(true);

	connect(m_randomButton, SIGNAL(clicked()), this, SLOT(seedingAlgorithmChanged()));
	connect(m_manualButton, SIGNAL(clicked()), this, SLOT(seedingAlgorithmChanged()));
	connect(m_hartiganWongButton, SIGNAL(clicked()), this, SLOT(seedingAlgorithmChanged()));
	connect(m_astrahanButton, SIGNAL(clicked()), this, SLOT(seedingAlgorithmChanged()));


	QVBoxLayout* group_layout = new QVBoxLayout();
	group_layout->addWidget(m_randomButton);
	group_layout->addWidget(m_manualButton);
	group_layout->addWidget(m_hartiganWongButton);
	group_layout->addWidget(m_astrahanButton);
	group_layout->addStretch(1);

	m_seedingGroup->setLayout(group_layout);

	/* Parameters goup */
	m_parametersGroup = new QGroupBox("Parameters");

	m_kBox = new QSpinBox();
	m_kBox->setSuffix(" clusters");
	m_kBox->setRange(1, 100000);

	m_iterationsBox = new QSpinBox();
	m_iterationsBox->setSuffix(" iterations");
	m_iterationsBox->setRange(1, 1000000);
	
	m_runBox = new QSpinBox();
	m_runBox->setSuffix(" runs");
	m_runBox->setRange(1, 1000000);

	m_doClusterButton = new QPushButton("Cluster!");
	m_findKButton = new QPushButton("Find k!");
	m_clearButton = new QPushButton("Clear!");
	m_clearSeedButton = new QPushButton("Clear Seed!");
	m_clearSeedButton->hide();

	group_layout = new QVBoxLayout();
	group_layout->addWidget(m_kBox);
	group_layout->addWidget(m_iterationsBox);
	group_layout->addWidget(m_runBox);
	group_layout->addWidget(m_doClusterButton);
	group_layout->addWidget(m_findKButton);
	group_layout->addWidget(m_clearButton);
	group_layout->addWidget(m_clearSeedButton);
	group_layout->addStretch(1);

	m_parametersGroup->setLayout(group_layout);

	layout->addWidget(m_seedingGroup);
	layout->addWidget(m_parametersGroup);
	layout->addStretch(1);
	setLayout(layout);
}

void ToolBox::seedingAlgorithmChanged()
{
	m_clearSeedButton->hide();
	m_runBox->show();
	SeedingAlgorithm s = RANDOM;
	if (m_manualButton->isChecked()) {
		m_clearSeedButton->show();
		m_runBox->hide();
		s = MANUAL;
	} else if (m_hartiganWongButton->isChecked()) {
		m_runBox->hide();
		s = HARTIGAN_WONG;
	} else if (m_astrahanButton->isChecked()) {
		m_runBox->hide();
		s = ASTRAHAN;
	}
	
	emit setSeedingAlgorithm(s);
}
