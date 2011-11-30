#include <toolbox.hpp>

#include <QtGui/QVBoxLayout>
#include <QtGui/QGroupBox>
#include <QtGui/QRadioButton>

#include <iostream>


ToolBox::ToolBox(QWidget *parent)
{
	setMaximumWidth(250);

	QVBoxLayout* layout = new QVBoxLayout();

	m_typeGroup = new QGroupBox("Computation");

	m_cpuButton = new QRadioButton("CPU");
	m_shaderButton = new QRadioButton("Shader");
	m_openclButton = new QRadioButton("OpenCL");
#ifdef USE_CUDA
	m_cudaButton = new QRadioButton("CUDA");
#endif
	m_cpuButton->setChecked(true);

	connect(m_cpuButton, SIGNAL(clicked()), this, SLOT(typeChanged()));
	connect(m_shaderButton, SIGNAL(clicked()), this, SLOT(typeChanged()));
	connect(m_openclButton, SIGNAL(clicked()), this, SLOT(typeChanged()));
#ifdef USE_CUDA
	connect(m_cudaButton, SIGNAL(clicked()), this, SLOT(typeChanged()));
#endif

	QVBoxLayout* group_layout = new QVBoxLayout();
	group_layout->addWidget(m_cpuButton);
	group_layout->addWidget(m_shaderButton);
	group_layout->addWidget(m_openclButton);
#ifdef USE_CUDA
	group_layout->addWidget(m_cudaButton);
#endif
	group_layout->addStretch(1);

	m_typeGroup->setLayout(group_layout);

	layout->addWidget(m_typeGroup);
	layout->addStretch(1);
	setLayout(layout);
}

void ToolBox::typeChanged()
{
	ComputationMode mode = CPU;
	if (m_shaderButton->isChecked())
		mode = SHADER;
	else if (m_openclButton->isChecked())
		mode = OPENCL;
#ifdef USE_CUDA
	else if (m_cudaButton->isChecked())
		mode = CUDA;
#endif
	
	emit setComputationMode(mode);
}
