/**
 * @author Markus Doellinger
 * @date November 16, 2011
 * @file main.cpp
 */

#include <waves.hpp>
#include <mainwindow.hpp>

#include <QtGui/QApplication>


#include <iostream>

int main(int argc, char **argv)
{
	QApplication app(argc, argv);
	MainWindow mainwindow(&app);

	int result = app.exec();
	return result;
}
