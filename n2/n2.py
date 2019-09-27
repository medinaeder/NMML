# -*- coding: utf-8 -*-
import sys
from   math import floor

from defcon import *
from dolfin import *

import defcon.backend as backend
import os

from subprocess import Popen

import matplotlib.pyplot as plt
from numpy import array
import numpy as np
from petsc4py import PETSc
from slepc4py import SLEPc

class HyperelasticityProblem(BifurcationProblem):
    E = 1.0
    nu = 0.48
    def mesh(self, comm):
        mesh_ = Mesh(comm,"mesh2.xml.gz")

        self.mesh_ = mesh_
        return mesh_

    def function_space(self, mesh):
        V = VectorFunctionSpace(mesh, "CG", 2)

        # Construct rigid body modes used in algebraic multigrid preconditioner later on
        rbms = [Constant((0, 1)),
                Constant((1, 0)),
                Expression(("-x[1]", "x[0]"), degree=1, mpi_comm=mesh.mpi_comm())]
        self.rbms = [interpolate(rbm, V) for rbm in rbms]
        print(V.dim())

        return V

    def parameters(self):
        eps = Constant(0)

        return [(eps, "eps", r"$\epsilon$")]

    def psi(self,u,params): 
        B   = Constant((0.0, 0.000)) # Body force per unit volume

        # Kinematics
        I = Identity(2)             # Identity tensor
        F = I + grad(u)             # Deformation gradient
        C = F.T*F                   # Right Cauchy-Green tensor

        # Invariants of deformation tensors
        Ic = tr(C)
        J  = det(F)

        # Elasticity parameters
        E, nu = self.E, self.nu
        mu, lmbda = Constant(E/(2*(1 + nu))), Constant(E*nu/((1 + nu)*(1 - 2*nu)))

        # Stored strain energy density (compressible neo-Hookean model)
        psi_ = (mu/2)*(Ic - 2) - mu*ln(J) + (lmbda/2)*(ln(J))**2 # PEF
        
        

        return psi_
    def residual(self, u, params, v):
        # Total potential energy
        Energy = self.psi(u,params)*dx  #- dot(T, u)*ds
        F = derivative(Energy, u, v)

        return F

    def boundary_conditions(self, V, params):
        eps = params[0]
        left  = CompiledSubDomain("(std::abs(x[1])       < 1e-4) && on_boundary", mpi_comm=V.mesh().mpi_comm())
        right = CompiledSubDomain("(std::abs(x[1] - 1.0) < 1e-4) && on_boundary", mpi_comm=V.mesh().mpi_comm())

        bcl = DirichletBC(V, (0.0,0.0), left)
        bcr = DirichletBC(V, (0.0,-eps), right)

        return [bcl,bcr]
    def functionals(self):
        def strain_energy(u, params):
            return assemble(self.psi(u,params)*dx)
        def force(u, params):
            V = u.function_space()
            v = TestFunction(V)
            mesh = self.mesh_
            left  = CompiledSubDomain("(std::abs(x[1])< 1e-4) && on_boundary", mpi_comm=V.mesh().mpi_comm())
            colors = MeshFunction("size_t",mesh, mesh.topology().dim()-1)
            colors.set_all(0)
            left.mark(colors,1)
            ds_p = ds(subdomain_data = colors)(1)

            # Kinematics
            I = Identity(2)             # Identity tensor
            F = variable(I + grad(u))   # Deformation gradient
            C = F.T*F                   # Right Cauchy-Green tensor

            # Invariants of deformation tensors
            Ic = tr(C)
            J  = det(F)

            # Elasticity parameters
            E, nu = self.E, self.nu
            mu, lmbda = Constant(E/(2*(1 + nu))), Constant(E*nu/((1 + nu)*(1 - 2*nu)))

            # Stored strain energy density (compressible neo-Hookean model)
            psi = (mu/2)*(Ic - 2) - mu*ln(J) + (lmbda/2)*(ln(J))**2 # PEF

            Pk1 = diff(psi, F)
            n = FacetNormal(mesh)
            fs = assemble(-dot(dot(Pk1/mu, n), n)*ds_p)

            return fs

        return [(strain_energy, "Energy", r"$\psi$"),
                (force, "Force", r"Stress/E")]


    def number_initial_guesses(self, params):
        return 1

    def initial_guess(self, V, params, n):
        return interpolate(Constant((0,0)),V)

    def number_solutions(self, params):
        # Here I know the number of solutions for each value of eps.
        # This cheating allows me to calculate the bifurcation diagram
        # much more quickly. This can be disabled without changing the
        # correctness of the calculations.
        eps = params[0]
        return float("inf")

    def transform_guess(self, state, task, io):
        C  = Constant((0,0))
        state.assign(state+C)

    def squared_norm(self, a, b, params):
        return inner(a - b, a - b)*dx + inner(grad(a - b), grad(a - b))*dx
    def solver(self, problem, params, solver_params, prefix="", **kwargs):
        # Set the rigid body modes for use in AMG

        s = SNUFLSolver(problem, solver_parameters=solver_params, prefix=prefix, **kwargs)
        snes = s.snes
        snes.setFromOptions()

        if snes.ksp.type != "preonly":
            # Convert rigid body modes (computed in self.function_space above) to PETSc Vec
            rbms = list(map(vec, self.rbms))

            # Create the PETSc nullspace
            nullsp = PETSc.NullSpace().create(vectors=rbms, constant=False, comm=snes.comm)

            (A, P) = snes.ksp.getOperators()
            A.setNearNullSpace(nullsp)
            P.setNearNullSpace(nullsp)

        return s
    def compute_stabilityx(self, params, branchid, u, hint=None):
        V = u.function_space()
        trial = TrialFunction(V)
        test  = TestFunction(V)

        bcs = self.boundary_conditions(V, params)
        comm = V.mesh().mpi_comm()

        F = self.residual(u, list(map(Constant, params)), test)
        J = derivative(F, u, trial)
        b = inner(Constant((1, 0)), test)*dx # a dummy linear form, needed to construct the SystemAssembler

        # Build the LHS matrix
        A = PETScMatrix(comm)
        asm = SystemAssembler(J, b, bcs)
        asm.assemble(A)

        # Build the mass matrix for the RHS of the generalised eigenproblem
        B = PETScMatrix(comm)
        asm = SystemAssembler(inner(test, trial)*dx, b, bcs)
        asm.assemble(B)
        [bc.zero(B) for bc in bcs]

        # Create the SLEPc eigensolver
        eps = SLEPc.EPS().create(comm=comm)
        eps.setOperators(A.mat(), B.mat())
        eps.setWhichEigenpairs(eps.Which.SMALLEST_MAGNITUDE)
        eps.setProblemType(eps.ProblemType.GHEP)
        eps.setFromOptions()

        # If we have a hint, use it - it's the eigenfunctions from the previous solve
        if hint is not None:
            initial_space = [vec(x) for x in hint]
            eps.setInitialSpace(initial_space)

        if eps.st.ksp.type != "preonly":
            # Convert rigid body modes (computed in self.function_space above) to PETSc Vec
            rbms = list(map(vec, self.rbms))

            # Create the PETSc nullspace
            nullsp = PETSc.NullSpace().create(vectors=rbms, constant=False, comm=comm)

            (A, P) = eps.st.ksp.getOperators()
            A.setNearNullSpace(nullsp)
            P.setNearNullSpace(nullsp)

        # Solve the eigenproblem
        eps.solve()

        eigenvalues = []
        eigenfunctions = []
        eigenfunction = Function(V, name="Eigenfunction")

        for i in range(eps.getConverged()):
            lmbda = eps.getEigenvalue(i)
            assert lmbda.imag == 0
            eigenvalues.append(lmbda.real)

            eps.getEigenvector(i, vec(eigenfunction))
            eigenfunctions.append(eigenfunction.copy(deepcopy=True))

        if min(eigenvalues) < 0:
            is_stable = False
        else:
            is_stable = True

        d = {"stable": is_stable,
             "eigenvalues": eigenvalues,
             "eigenfunctions": eigenfunctions,
             "hint": eigenfunctions}

        return d

    def solver_parameters(self, params, task, **kwargs):
        return {
               "snes_max_it": 100,
               "snes_atol": 1.0e-8,
               "snes_rtol": 1.0e-10,
               "snes_max_linear_solve_fail": 100,
               "snes_linesearch_type": "l2",
               "snes_monitor": None,
               "snes_converged_reason": None,
               "ksp_type": "preonly",
               "ksp_monitor_cancel": None,
               "ksp_converged_reason": None,
               "ksp_max_it": 20,
               "pc_type": "cholesky",
               "pc_factor_mat_solver_package": "mumps",
               "eps_type": "krylovschur",
               "eps_target": -1,
               "eps_monitor_all": None,
               "eps_converged_reason": None,
               "eps_nev": 1,
               "st_type": "sinvert",
               "st_ksp_type": "preonly",
               "st_pc_type": "lu",
               "st_pc_factor_mat_solver_package": "mumps",
               }
    def postprocess(self, solution, params, branchid,window):
        # fetch the eigenfunctions associated with our point
        io2 = self.io()
        mesh = self.mesh(backend.comm_world)
        parameters = self.parameters()
        functionals = self.functionals()
        io2.setup(parameters,functionals, self.function_space(mesh))
        d_l = io2.fetch_stability(params,[branchid],True)
        sol = io2.fetch_solutions(params,[branchid])
        V = self.function_space(self.mesh_)
        try: os.mkdir('test')
        except OSError: pass 
        pvd_filename  = os.path.join('test', "egfunc.pvd")
        pvd = backend.File(pvd_filename);
        for d in d_l: 
            for i,e_v in enumerate(d["eigenvalues"]):
                e_f=d["eigenfunctions"][i]
                e_f.rename("Solution", "Solution")
                #print(i,e_v,e_f)
                e2 = project(0.01*e_f+sol[0],V)
                #e2 =  0.01*e_f+sol[0]

                # Uncomment to only save eigenfunction
                #self.save_pvd(e_f, pvd);
                # Save the eigenfunction on top of the solution    
                self.save_pvd(e2,pvd)
        
        try: Popen(["paraview", pvd_filename])
        except Exception: pass 
        
    

if __name__ == "__main__":

    problem = HyperelasticityProblem()
    dc = DeflatedContinuation(problem=HyperelasticityProblem(), teamsize=1, verbose=True, clear_output=True)
    e_max = 0.08
    params = list(arange(0.001, e_max, 0.001)) + [e_max]
    dc.run(values={"eps": params})
