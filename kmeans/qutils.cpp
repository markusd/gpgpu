/**
 * @author Markus Holtermann
 * @date Jun 7, 2011
 * @file gui/qutils.cpp
 */

#include <qutils.hpp>
#include <QtGui/QKeyEvent>
#include <QtGui/QMouseEvent>
#include <QtGui/QWheelEvent>



void QtErrorListerner::displayError(const std::string& message)
{
	//gui::MessageDialog("Error", message, gui::MessageDialog::QERROR);
}

void QtKeyAdapter::keyEvent(QKeyEvent* event)
{
	// do nothing if the key was already pressed
	if (event->isAutoRepeat())
		return;

	// if the text is empty, it was a special key
	if (!event->text().isEmpty()) {
		QString text = event->text().toLower();
		unsigned char key = text.toAscii().at(0);
		if (event->type() == QEvent::KeyPress)
			m_pressed.insert(key);
		else
		if (event->type() == QEvent::KeyRelease)
			m_pressed.erase(key);
	} else {
		m_alt = event->key() == Qt::Key_Alt && event->type() == QEvent::KeyPress;
		m_ctrl = event->key() == Qt::Key_Control && event->type() == QEvent::KeyPress;
		m_shift = event->key() == Qt::Key_Shift && event->type() == QEvent::KeyPress;
	}
}

void QtMouseAdapter::mouseEvent(QMouseEvent* event)
{
	util::Button button = util::LEFT;
	switch (event->button()) {
	case Qt::LeftButton:
		button = util::LEFT;
		break;

	case Qt::RightButton:
		button = util::RIGHT;
		break;

	case Qt::MidButton:
		button = util::MIDDLE;
		break;

	default:
		break;
	}

	switch (event->type()) {

	case QEvent::MouseButtonPress:
	case QEvent::MouseButtonRelease:
		m_down[button] = event->type() == QEvent::MouseButtonPress;
		for (std::list<util::MouseListener*>::iterator itr = m_listeners.begin();
				itr != m_listeners.end(); ++itr)
			(*itr)->mouseButton(button, m_down[button], event->x(), event->y());
		break;

	case QEvent::MouseButtonDblClick:
		for (std::list<util::MouseListener*>::iterator itr = m_listeners.begin();
				itr != m_listeners.end(); ++itr)
			(*itr)->mouseDoubleClick(button, event->x(), event->y());
		break;

	case QEvent::MouseMove:
		for (std::list<util::MouseListener*>::iterator itr = m_listeners.begin();
				itr != m_listeners.end(); ++itr)
			(*itr)->mouseMove(event->x(), event->y());
		break;

	default:
		break;
	}

	m_x = event->x();
	m_y = event->y();
}

void QtMouseAdapter::mouseWheelEvent(QWheelEvent* event) {
	for (std::list<util::MouseListener*>::iterator itr = m_listeners.begin();
			itr != m_listeners.end(); ++itr)
		(*itr)->mouseWheel(event->delta());
}

