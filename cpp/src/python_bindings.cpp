#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "torch/csrc/autograd/python_variable.h"

#include "core.hpp"

namespace emulators_py_cpp_bindings {

int get_tensor(PyObject* object, void* result) {
    if (!THPVariable_CheckExact(object)) {
        PyErr_SetString(PyExc_RuntimeError, "not a tensor");
        return 0;
    }

    *(static_cast<at::Tensor*>(result)) = THPVariable_Unpack(object);
    return 1;
}

int get_config(PyObject* config, void* address) {
    auto py_precision = PyObject_GetAttrString(config, "precision");
    if (!py_precision) {
        return 0;
    }
    double precision = PyFloat_AsDouble(py_precision);
    if (PyErr_Occurred()) {
        return 0;
    }
    Py_DECREF(py_precision);

    auto py_extra_krylov_tolerance = PyObject_GetAttrString(config, "extra_krylov_tolerance");
    if (!py_extra_krylov_tolerance) {
        return 0;
    }
    double extra_krylov_tolerance = PyFloat_AsDouble(py_extra_krylov_tolerance);
    if (PyErr_Occurred()) {
        return 0;
    }
    Py_DECREF(py_extra_krylov_tolerance);

    auto py_max_krylov_dim = PyObject_GetAttrString(config, "max_krylov_dim");
    if (!py_max_krylov_dim) {
        return 0;
    }

    long max_krylov_dim = PyLong_AsLong(py_max_krylov_dim);
    if (PyErr_Occurred()) {
        return 0;
    }
    Py_DECREF(py_max_krylov_dim);

    auto py_max_bond_dim = PyObject_GetAttrString(config, "max_bond_dim");
    if (!py_max_bond_dim) {
        return 0;
    }
    long max_bond_dim = PyLong_AsLong(py_max_bond_dim);
    if (PyErr_Occurred()) {
        return 0;
    }
    Py_DECREF(py_max_bond_dim);

    auto& result = *static_cast<emulators_cpp::Config*>(address);
    result.precision = precision;
    result.extra_krylov_tolerance = extra_krylov_tolerance;
    result.max_krylov_dim = max_krylov_dim;
    result.max_bond_dim = max_bond_dim;

    return 1;
}

static PyObject *evolve_single(
        PyObject *self,
        PyObject *args,
        PyObject *kwargs
    ) {
    at::Tensor state_factor;
    at::Tensor left_bath;
    at::Tensor right_bath;
    at::Tensor ham_factor;
    double dt = 0;
    int is_hermitian = true; // Warning: don't use bool type here, it causes parsing with "p" to fail.
    emulators_cpp::Config config{};

    static char *kwlist[] = {"state_factor", "baths", "ham_factor", "dt", "is_hermitian", "config", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "$O&(O&O&)O&dpO&", kwlist,
                                     &get_tensor, &state_factor,
                                     &get_tensor, &left_bath,
                                     &get_tensor, &right_bath,
                                     &get_tensor, &ham_factor,
                                     &dt, &is_hermitian,
                                     &get_config, &config)) {
        return NULL;
    }


    try {
        auto result = emulators_cpp::evolve_single(state_factor, left_bath, right_bath, ham_factor,
        dt, is_hermitian, config);
        return THPVariable_Wrap(result);
    } catch (std::exception const& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}


static PyObject *evolve_pair(
        PyObject *self,
        PyObject *args,
        PyObject *kwargs
    ) {
    at::Tensor left_state_factor;
    at::Tensor right_state_factor;
    at::Tensor left_bath;
    at::Tensor right_bath;
    at::Tensor left_ham_factor;
    at::Tensor right_ham_factor;
    double dt = 0;
    int orth_center_right = false; // Warning: don't use bool type here, it causes parsing with "p" to fail.
    int is_hermitian = true; // Warning: don't use bool type here, it causes parsing with "p" to fail.
    emulators_cpp::Config config{};

    static char *kwlist[] = {"state_factors", "baths", "ham_factors", "dt", "orth_center_right", "is_hermitian", "config", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "$(O&O&)(O&O&)(O&O&)dppO&", kwlist,
                                     &get_tensor, &left_state_factor,
                                     &get_tensor, &right_state_factor,
                                     &get_tensor, &left_bath,
                                     &get_tensor, &right_bath,
                                     &get_tensor, &left_ham_factor,
                                     &get_tensor, &right_ham_factor,
                                     &dt,
                                     &orth_center_right,
                                     &is_hermitian,
                                     &get_config, &config)) {
        return NULL;
    }

    try {
        auto [l, r] = emulators_cpp::evolve_pair(left_state_factor, right_state_factor,
        left_bath, right_bath, left_ham_factor, right_ham_factor,
        dt, orth_center_right, is_hermitian, config);

        return PyTuple_Pack(2, THPVariable_Wrap(l), THPVariable_Wrap(r));
    } catch (std::exception const& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

int get_ham_parameters(PyObject* py_ham, void* address) {
    auto py_omegas = PyObject_GetAttrString(py_ham, "omegas");
    if (!py_omegas) {
        return 0;
    }
    if (!THPVariable_CheckExact(py_omegas)) {
        PyErr_SetString(PyExc_RuntimeError, "omegas should be tensor");
        return 0;
    }
    auto omegas = THPVariable_Unpack(py_omegas);

    auto py_deltas = PyObject_GetAttrString(py_ham, "deltas");
    if (!py_deltas) {
        return 0;
    }
    if (!THPVariable_CheckExact(py_deltas)) {
        PyErr_SetString(PyExc_RuntimeError, "deltas should be tensor");
        return 0;
    }
    auto deltas = THPVariable_Unpack(py_deltas);

    auto py_interaction_matrix = PyObject_GetAttrString(py_ham, "interaction_matrix");
    if (!py_interaction_matrix) {
        return 0;
    }
    if (!THPVariable_CheckExact(py_interaction_matrix)) {
        PyErr_SetString(PyExc_RuntimeError, "interaction_matrix should be tensor");
        return 0;
    }
    auto interaction_matrix = THPVariable_Unpack(py_interaction_matrix);

    auto& ham_params = *(static_cast<emulators_cpp::HamParameters*>(address));
    ham_params.omegas = omegas;
    ham_params.deltas = deltas;
    ham_params.interaction_matrix = interaction_matrix;

    return 1;
}

static PyObject *evolve_sv_rydberg(
        PyObject *self,
        PyObject *args,
        PyObject *kwargs
    ) {
    double dt = 0;
    emulators_cpp::HamParameters ham_params;
    at::Tensor state_vector;
    double krylov_tolerance;

    static char *kwlist[] = {"dt", "ham", "state_vector", "krylov_tolerance", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "$dO&O&d", kwlist,
                                     &dt,
                                     &get_ham_parameters, &ham_params,
                                     &get_tensor, &state_vector,
                                     &krylov_tolerance)) {
        return NULL;
    }

    try {
        auto result = emulators_cpp::evolve_sv_rydberg(dt, ham_params,
            state_vector, krylov_tolerance);
        return THPVariable_Wrap(result);
    } catch (std::exception const& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

static PyObject *apply_rydberg_sv(
        PyObject *self,
        PyObject *args,
        PyObject *kwargs
    ) {
    emulators_cpp::HamParameters ham_params;
    at::Tensor state_vector;

    static char *kwlist[] = {"hamiltonian", "state_vector", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "$O&O&", kwlist,
                                     &get_ham_parameters, &ham_params,
                                     &get_tensor, &state_vector)) {
        return NULL;
    }

    try {
        auto result = emulators_cpp::apply_rydberg_sv(ham_params, state_vector);
        return THPVariable_Wrap(result);
    } catch (std::exception const& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

static PyMethodDef methods[] = {
    {   "evolve_single",
        (PyCFunction) evolve_single,
        METH_VARARGS | METH_KEYWORDS,
        "Evolve single" },
    {   "evolve_pair",
        (PyCFunction) evolve_pair,
        METH_VARARGS | METH_KEYWORDS,
        "Evolve pair" },
    {   "evolve_sv_rydberg",
        (PyCFunction) evolve_sv_rydberg,
        METH_VARARGS | METH_KEYWORDS,
        "Evolve state vector with rydberg hamiltonian" },
    {   "apply_rydberg_sv",
        (PyCFunction) apply_rydberg_sv,
        METH_VARARGS | METH_KEYWORDS,
        "Apply rydberg hamiltonian to state vector" },
    {nullptr}
};


static PyModuleDef module = [] {
    PyModuleDef result{ PyModuleDef_HEAD_INIT };
    result.m_name = "emulators_cpp";
    result.m_doc = "Module containing some C++ primitives";
    result.m_size = -1;

    return result;
}();

}

PyMODINIT_FUNC
PyInit_emulators_cpp(void) {
    PyObject *m;
    m = PyModule_Create(&emulators_py_cpp_bindings::module);
    if (m == nullptr) {
        return NULL;
    }

    if (PyModule_AddFunctions(m, emulators_py_cpp_bindings::methods) < 0) {
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
