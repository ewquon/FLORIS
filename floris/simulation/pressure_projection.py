# Copyright 2019 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import cg
#from pyamg import smoothed_aggregation_solver
#from pyamg.gallery import poisson

class PressureField(object):
    """
    Given a modeled velocity field, solve for the perturbation pressure
    field corresponding to a perturbation velocity field that, when
    combined with the original velocity field, satisfies mass
    conservation.
    """
    def __init__(self,flow_field):
        # setup grid
        self.x = flow_field.x
        self.y = flow_field.y
        self.z = flow_field.z
        self.Nx, self.Ny, self.Nz = self.x.shape
        self.N = self.Nx * self.Ny * self.Nz

        # set initial fields
        self.u0 = flow_field.u
        self.v0 = flow_field.v
        self.w0 = flow_field.w

        # get actual grid spacings
        dx = np.diff(self.x[:,0,0])
        dy = np.diff(self.y[0,:,0])
        dz = np.diff(self.z[0,0,:])
        assert np.max(np.abs(dx - dx[0]) < 1e-8)
        assert np.max(np.abs(dy - dy[0]) < 1e-8)
        assert np.max(np.abs(dz - dz[0]) < 1e-8)
        self.dx = dx[0]
        self.dy = dy[0]
        self.dz = dz[0]

        # setup solver matrices
        self._setup_LHS()
        self.RHS = None

    def _setup_LHS(self):
        """Compressed sparse row (CSR) format appears to be slightly
        more efficient than compressed sparse column (CSC).
        """
        ones = np.ones((self.N,))
        diag = -2*ones/self.dx**2 - 2*ones/self.dy**2 - 2*ones/self.dz**2
        # off diagonal for d/dx operator    
        offx = self.Ny*self.Nz
        offdiagx = ones[:-offx]/self.dx**2
        # off diagonal for d/dy operator    
        offy = self.Nz
        offdiagy = ones[:-offy]/self.dy**2
        for i in range(offx, len(offdiagy), offx):
            offdiagy[i-offy:i] -= 1./self.dy**2
        # off diagonal for d/dz operator    
        offz = 1
        offdiagz = ones[:-offz]/self.dz**2
        offdiagz[self.Nz-1::self.Nz] -= 1./self.dz**2
        # spsolve requires matrix to be in CSC or CSR format
        self.LHS = diags(
            [
                offdiagx,
                offdiagy,
                offdiagz,
                diag,
                offdiagz,
                offdiagy,
                offdiagx,
            ],
            [-offx,-offy,-offz,0,offz,offy,offx],
            format='csr'
        )

    def calc_gradients(self):
        """Calculate RHS of Poisson equation, div(U), from finite
        differences. Second-order central differences are evaluated
        on the interior, first-order one-sided differences on the
        boundaries.
        """
        self.du0_dx = np.zeros(self.u0.shape)
        self.dv0_dy = np.zeros(self.v0.shape)
        self.dw0_dz = np.zeros(self.w0.shape)
        # u, inlet
        self.du0_dx[0,:,:] = (self.u0[1,:,:] - self.u0[0,:,:]) / self.dx
        # u, outlet
        self.du0_dx[-1,:,:] = (self.u0[-1,:,:] - self.u0[-2,:,:]) / self.dx
        # interior
        self.du0_dx[1:-1,:,:] = (self.u0[2:,:,:] - self.u0[:-2,:,:]) / (2*self.dx)
        if self.v0 is not None:
            # v, -y
            self.dv0_dy[:,0,:] = (self.v0[:,1,:] - self.v0[:,0,:]) / self.dy
            # v, +y
            self.dv0_dy[:,-1,:] = (self.v0[:,-1,:] - self.v0[:,-2,:]) / self.dy
            # interior
            self.dv0_dy[:,1:-1,:] = (self.v0[:,2:,:] - self.v0[:,:-2,:]) / (2*self.dy)
        if self.w0 is not None:
            # w, lower
            self.dw0_dz[:,:,0] = (self.w0[:,:,1] - self.w0[:,:,0]) / self.dz
            # w, upper
            self.dw0_dz[:,:,-1] = (self.w0[:,:,-1] - self.w0[:,:,-2]) / self.dz
            # interior
            self.dw0_dz[:,:,1:-1] = (self.w0[:,:,2:] - self.w0[:,:,:-2]) / (2*self.dz)
        # update RHS
        self.set_RHS(self.du0_dx, self.dv0_dy, self.dw0_dz)

    def set_RHS(self,du_dx,dv_dy=None,dw_dz=None):
        """Set the RHS of the Poisson equation, which is the divergence
        of the initial velocity predictor (i.e., the modeled velocity
        field).
        """
        div = du_dx
        if dv_dy is not None:
            div += dv_dy
        if dw_dz is not None:
            div += dw_dz
        self.RHS = div.ravel()

    def _correct_field(self,A=1.0):
        dp_dx = (self.p[2:,:,:] - self.p[:-2,:,:]) / (2*self.dx)
        dp_dy = (self.p[:,2:,:] - self.p[:,:-2,:]) / (2*self.dy)
        dp_dz = (self.p[:,:,2:] - self.p[:,:,:-2]) / (2*self.dz)
        self.u = self.u0.copy()
        if self.v0 is None:
            self.v = np.zeros(self.u.shape)
        else:
            self.v = self.v0.copy()
        if self.w0 is None:
            self.w = np.zeros(self.u.shape)
        else:
            self.w = self.w0.copy()
        self.u[1:-1,:,:] -= A*dp_dx
        self.v[:,1:-1,:] -= A*dp_dy
        self.w[:,:,1:-1] -= A*dp_dz

    def solve(self,A=1.0,tol=1e-5):
        """Solve Poisson equation for perturbation pressure field,
        according to the formulation in Tannehill, Anderson, and Pletcher
        (1997).
        
        Note: For a fictitious timestep (dt), A == dt/rho [m^3-s/kg]
        """
        assert(self.RHS is not None)
        self._setup_LHS()
        soln = cg(self.LHS, self.RHS/A, x0=np.zeros((self.N,)), tol=tol, atol=tol)
        assert(soln[1] == 0) # success
        self.p = soln[0].reshape(self.u0.shape)
        self._correct_field(A)
