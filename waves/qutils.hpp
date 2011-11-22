/**
 * @author Markus Holtermann
 * @date Jun 17, 2011
 * @file gui/qutils.hpp
 */

#ifndef QUTILSS_HPP_
#define QUTILSS_HPP_

#include <util/inputadapters.hpp>
#include <util/erroradapters.hpp>

class QKeyEvent;
class QMouseEvent;
class QWheelEvent;



class QtErrorListerner: public util::ErrorListener {
public:
	void displayError(const std::string& message);
};

/**
 * A key adapter for the Qt windowing system
 */
class QtKeyAdapter : public util::KeyAdapter {
public:
	/**
	 * Must be called if a new Qt key event is received. The method
	 * will set the appropriate modifier and key states.
	 *
	 * @param event The received Qt key event
	 */
	void keyEvent(QKeyEvent* event);
};

/**
 * A mouse adapter for the Qt windowing system.
 */
class QtMouseAdapter : public util::MouseAdapter {
public:
	/**
	 * Must be called if a new Qt mouse event is received. The
	 * method will set the appropriate button and position states.
	 *
	 * @param event The received Qt mouse event
	 */
	void mouseEvent(QMouseEvent* event);

	/**
	 *
	 * @param event The received Qt wheel event
	 */
	void mouseWheelEvent(QWheelEvent* event);
};


#endif /* QUTILSS_HPP_ */
