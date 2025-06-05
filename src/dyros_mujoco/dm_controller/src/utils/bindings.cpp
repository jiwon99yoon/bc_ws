#define BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS
// #define BOOST_MPL_LIMIT_LIST_SIZE 40

#include <boost/python.hpp>
#include <eigenpy/eigenpy.hpp>
#include <Eigen/Dense>

#include "dm_controller/controller.h"

namespace bp = boost::python;
using namespace DmController;

// Converter for std::vector<std::string>
struct VectorString_to_python
{
    static PyObject* convert(const std::vector<std::string>& vec)
    {
        boost::python::list py_list;
        for (const auto& str : vec)
        {
            py_list.append(str);
        }   
        return bp::incref(py_list.ptr());
    }
};

struct VectorString_from_python
{
    VectorString_from_python()
    {
        bp::converter::registry::push_back(&convertible, &construct, boost::python::type_id<std::vector<std::string>>());
    }

    static void* convertible(PyObject* obj_ptr)
    {
        if (!PySequence_Check(obj_ptr)) return nullptr;
        return obj_ptr;
    }

    static void construct(PyObject* obj_ptr, bp::converter::rvalue_from_python_stage1_data* data)
    {
        void* storage = ((bp::converter::rvalue_from_python_storage<std::vector<std::string>>*)data)->storage.bytes;
        new (storage) std::vector<std::string>();
        std::vector<std::string>& vec = *(std::vector<std::string>*)(storage);

        int len = PySequence_Size(obj_ptr);
        if (len < 0) bp::throw_error_already_set();
        vec.reserve(len);

        for (int i = 0; i < len; ++i)
        {
            vec.push_back(bp::extract<std::string>(PySequence_GetItem(obj_ptr, i)));
        }

        data->convertible = storage;
    }
};


BOOST_PYTHON_MODULE(dm_controller_wrapper_cpp) 
{
    eigenpy::enableEigenPy();
    eigenpy::enableEigenPySpecific<Eigen::Matrix<double, Eigen::Dynamic, 1>>();
    eigenpy::enableEigenPySpecific<Eigen::Matrix<double, 4, 4>>();
    eigenpy::enableEigenPySpecific<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>();
    
    bp::to_python_converter<std::vector<std::string>, VectorString_to_python>();
    VectorString_from_python();

    // Bind Manipulator class
    bp::class_<Controller, boost::noncopyable>("Controller", bp::init<const std::string>())
        // ----- <State update Func> -----
        .def("updateModel", &Controller::updateModel)
        .def("gripperOpen", &Controller::gripperOpen)
        .def("setIdleConfig", &Controller::setIdleConfig)
        .def("taskMove", &Controller::taskMove)
        .def("jointMove", &Controller::jointMove)
        .def("idleState", &Controller::idleState)
        .def("initState", &Controller::initState)
        .def("updateInitialValues", &Controller::updateInitialValues);
}
