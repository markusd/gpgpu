/**
 * @author Markus Holtermann, Robert Waury
 * @date Jun 30, 2011
 * @file util/erroradapters.hpp
 */

#ifndef ERRORADAPTERS_HPP_
#define ERRORADAPTERS_HPP_

#include <util/adapter.hpp>
#include <vector>
#include <string>
#include <stdexcept>
#include <xml/rapidxml.hpp>

namespace util {

class ErrorListener {
public:
	ErrorListener();
	virtual ~ErrorListener();
	virtual void displayError(const std::string& message);
};

class ErrorAdapter: public Adapter<ErrorListener> {
private:
	static ErrorAdapter* s_instance;

public:
	static void createInstance();
	static void destroyInstance();
	static ErrorAdapter& instance();
	void displayErrorMessage(const std::string& message);
	void displayErrorMessage(const std::string& function,
			const std::vector<std::string>& args);
	void displayErrorMessage(const std::string& function,
			const std::vector<std::string>& args, rapidxml::parse_error& e);
	void displayErrorMessage(const std::string& function,
			const std::vector<std::string>& args, std::runtime_error& e);

};

inline ErrorAdapter& ErrorAdapter::instance() {
	if (!s_instance)
		createInstance();
	return *s_instance;
}

}

#endif /* END ERRORADAPTERS_HPP_ */
