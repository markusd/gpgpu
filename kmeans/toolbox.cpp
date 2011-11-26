#include <toolbox.hpp>

#include <QtGui/QVBoxLayout>
#include <QtGui/QGroupBox>
#include <QtGui/QRadioButton>

#include <iostream>


ToolBox::ToolBox(QWidget *parent)
{
	setMaximumWidth(250);
/*
	QVBoxLayout* layout = new QVBoxLayout();

	m_typeGroup = new QGroupBox("Computation");

	m_cpuButton = new QRadioButton("CPU");
	m_shaderButton = new QRadioButton("Shader");
	m_openclButton = new QRadioButton("OpenCL");
	m_cpuButton->setChecked(true);

	connect(m_cpuButton, SIGNAL(clicked()), this, SLOT(typeChanged()));
	connect(m_shaderButton, SIGNAL(clicked()), this, SLOT(typeChanged()));
	connect(m_openclButton, SIGNAL(clicked()), this, SLOT(typeChanged()));

	QVBoxLayout* group_layout = new QVBoxLayout();
	group_layout->addWidget(m_cpuButton);
	group_layout->addWidget(m_shaderButton);
	group_layout->addWidget(m_openclButton);
	group_layout->addStretch(1);

	m_typeGroup->setLayout(group_layout);

	layout->addWidget(m_typeGroup);
	layout->addStretch(1);
	setLayout(layout);
	*/
}
/*
void ToolBox::typeChanged()
{
	ComputationMode mode = CPU;
	if (m_shaderButton->isChecked())
		mode = SHADER;
	else if (m_openclButton->isChecked())
		mode = OPENCL;
	
	emit setComputationMode(mode);
}
*/