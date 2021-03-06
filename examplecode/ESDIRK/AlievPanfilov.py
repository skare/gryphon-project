# coding:utf8
"""
This demo program solves Aliev-Panfilof equations.
They described in article http://www.sciencedirect.com/science/article/pii/0960077995000895

We write Aliev-Panfilov model in the folowing form 
.. math::
	 \phi_t = div(D*grad(\phi) + c \phi(\phi - \alpha)(1 - \phi) - r\phi
	    r_t = -(\gamma + r\mu_1/(\mu_2 + \phi))(r + c\phi(\phi-b-1))
	
Demo was tested on Ubuntu 14.04 with Fenics library 1.5

"""
# Copyright (C) 2015 Vladimir Zverev (vladimir.zverev@urfu.ru, Ural Federal University)

# This file is part of Gryphon project
# Copyright (C) 2012-2019 - Knut Erik Skare
#
# Gryphon is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Gryphon is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Gryphon. If not, see <http://www.gnu.org/licenses/>.


from datetime import datetime

from dolfin import *
from gryphon import ESDIRK


def set_output_dir():
    return u"./Results/_" + datetime.today().strftime('%y_%m_%d %H_%M')


class InitialConditions(UserExpression):
    def eval(self, values, x):
        values[1] = 0
        if between(x[2], (0.0, 0.25)):
            values[0] = 1
        else:
            values[0] = 0

    def value_shape(self):
        return (2,)


if __name__ == "__main__":
    output_dir = str(set_output_dir())

    mesh = BoxMesh(Point(0, 1, 0), Point(1, 0, 4), 2, 2, 8)

    P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    ME = FunctionSpace(mesh, P1 * P1)
    q1, q2 = TestFunctions(ME)

    # Define and interpolate initial condition
    W = Function(ME)
    W.interpolate(InitialConditions())
    phi, r = split(W)

    # Set parameters of the model
    beta_t = 13  # ms, conversion factor
    beta_phi = Constant(100)  # mV, conversion factor
    delta_phi = Constant(-80)  # mV, potencial difference
    d_along_dimensional = 0.3  # mm^2/ms, diffusion coefficient along a fiber
    d_across_dimensional = 0.033  # mm^2/ms, diffusion coefficient across a fiber
    d_along = d_along_dimensional * beta_t
    d_across = d_across_dimensional * beta_t
    alpha = Constant(0.01)
    b = Constant(0.15)
    c = Constant(8)
    gamma = Constant(0.002)
    mu_1 = Constant(0.2)
    mu_2 = Constant(0.3)

    # Define the time domain
    T = [0, 10]

    # Create mesh functions for coefficients of a diffusion tensor
    d00 = MeshFunction("double", mesh, 3)
    d11 = MeshFunction("double", mesh, 3)
    d22 = MeshFunction("double", mesh, 3)

    # numpy.outer(fiber_v, fiber_v)
    # assume that fiber is directed along Oz
    # Iterate over mesh and set values
    for cell in cells(mesh):
        d00[cell] = d_across
        d11[cell] = d_across
        d22[cell] = d_along

    # Code for C++ evaluation of conductivity
    DiffusionTensor_code = '''
    
    #include <pybind11/pybind11.h>
    #include <pybind11/eigen.h>
    namespace py = pybind11;
    
    #include <dolfin/function/Expression.h>
    #include <dolfin/mesh/MeshFunction.h>
    #include <dolfin/common/Array.h>
    
    class DiffusionTensor : public dolfin::Expression
    {
    public:
    
      // Create expression with 3 components
      DiffusionTensor() : dolfin::Expression(3) {}
    
      // Function for evaluating expression on each cell
      void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x, const ufc::cell& cell) const override
      {
        const int cell_index = cell.index;
        values[0] = (*d00)[cell_index];
        values[1] = (*d11)[cell_index];
        values[2] = (*d22)[cell_index];
      }
    
      // The data stored in mesh functions
      std::shared_ptr<dolfin::MeshFunction<double> > d00;
      std::shared_ptr<dolfin::MeshFunction<double> > d11;
      std::shared_ptr<dolfin::MeshFunction<double> > d22;
    
    };
    
    PYBIND11_MODULE(SIGNATURE, m)
    {
      py::class_<DiffusionTensor, std::shared_ptr<DiffusionTensor>, dolfin::Expression>
        (m, "DiffusionTensor")
        .def(py::init<>())
        .def_readwrite("d00", &DiffusionTensor::d00)
        .def_readwrite("d11", &DiffusionTensor::d11)
        .def_readwrite("d22", &DiffusionTensor::d22);
    }
    '''

    d_temp = CompiledExpression(compile_cpp_code(DiffusionTensor_code).DiffusionTensor(),
                           d00=d00, d11=d11, d22=d22, degree=0)
    D = as_matrix(((d_temp[0], 0, 0), (0, d_temp[1], 0), (0, 0, d_temp[2])))

    # Define the right hand side for each of the PDEs
    F1 = (-inner(D * grad(phi), grad(q1)) + c * phi * (phi - alpha) * (1 - phi) * q1 - r * phi * q1) * dx
    F2 = (gamma + r * mu_1 / (mu_2 + phi)) * (-r - c * phi * (phi - b - 1)) * q2 * dx

    # Create the solver object and adjust tolerance
    obj = ESDIRK(T, W, [F1, F2])
    obj.parameters["timestepping"]["absolute_tolerance"] = 1e-3

    # Turn on some output and save run time
    obj.parameters["verbose"] = True
    obj.parameters["drawplot"] = False
    obj.parameters['output']['reuseoutputfolder'] = True
    obj.parameters["output"]["path"] = output_dir
    obj.parameters["output"]["statistics"] = True
    obj.parameters["output"]["plot"] = True

    # Supress some FEniCS output
    set_log_level(LogLevel.WARNING)

    # Solve the problem
    obj.solve()
