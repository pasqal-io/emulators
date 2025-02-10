#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "torch/csrc/autograd/python_variable.h"

#include "evolve_single.hpp"

static PyObject *evolve_single(
        PyObject *self,
        PyObject *args,
        PyObject *kwargs
    ) {
    PyObject *state_factor;
    PyObject *left_bath;
    PyObject *right_bath;
    PyObject *ham_factor;
    double dt = 0;
    int is_hermitian = true; // Warning: don't use bool type here, it causes parsing with "p" to fail.
    PyObject *config = Py_None; // FIXME

    static char *kwlist[] = {"state_factor", "baths", "ham_factor", "dt", "is_hermitian", "config", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "$O(OO)OdpO", kwlist,
                                     &state_factor, &left_bath, &right_bath, &ham_factor, &dt, &is_hermitian, &config)) {
        return NULL;
    }

    if (!THPVariable_CheckExact(state_factor)) {
        PyErr_SetString(PyExc_RuntimeError, "state should be a tensor");
        return NULL;
    }

    auto cpp_state_factor = THPVariable_Unpack(state_factor);

    auto cpp_left_bath = THPVariable_Unpack(left_bath);
    auto cpp_right_bath = THPVariable_Unpack(right_bath);

    if (!THPVariable_CheckExact(ham_factor)) {
        PyErr_SetString(PyExc_RuntimeError, "ham_factor should be a tensor");
        return NULL;
    }
    auto cpp_ham_factor = THPVariable_Unpack(ham_factor);

    return THPVariable_Wrap(emulators_cpp::evolve_single(cpp_state_factor, cpp_left_bath, cpp_right_bath, cpp_ham_factor, dt, is_hermitian));
}

static PyMethodDef emulators_cpp_module_methods[] = {
    {   "evolve_single",
        (PyCFunction) evolve_single,
        METH_VARARGS | METH_KEYWORDS,
        "Evolve single blablabla" },
    {nullptr}
};

static PyModuleDef emulators_cpp_module = [] {
    PyModuleDef result{ PyModuleDef_HEAD_INIT };
    result.m_name = "emulators_cpp";
    result.m_doc = "Module containing some C++ primitives";
    result.m_size = -1;

    return result;
}();


PyMODINIT_FUNC
PyInit_emulators_cpp(void) {
    PyObject *m;
    m = PyModule_Create(&emulators_cpp_module);
    if (m == nullptr) {
        return nullptr;
    }

    if (PyModule_AddFunctions(m, emulators_cpp_module_methods) < 0) {
        Py_DECREF(m);
        return nullptr;
    }

    return m;
}
