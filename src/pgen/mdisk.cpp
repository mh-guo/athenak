//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file mdisk.cpp
//! \brief (@mhguo) A model of Magnetized Disk Accretion onto Black Holes

// Athena++ headers
#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "pgen/pgen.hpp"

namespace {
struct mdisk_pgen {
  Real iso_cs;              // isothermal sound speed
  Real sink_d;              // sink density
  Real sink_p;              // sink pressure
  Real dens_0;              // gas density
  Real j_z;                 // specific angular momentum
  Real b_ini;               // initial magnetic field strength
  Real dt_floor;            // minimum time step
  int  ndiag = 0;           // number of cycles between diagnostics
  bool bc_flag = 0;         // boundary condition flag
};

  mdisk_pgen mdisk;

KOKKOS_INLINE_FUNCTION
static Real Acceleration(const Real r) {
  return -1.0/SQR(r);
}
KOKKOS_INLINE_FUNCTION
static void ComputePrimitiveSingle(Real x, Real y, Real z, struct mdisk_pgen pgen,
                                   Real& rho, Real& uu1, Real& uu2, Real& uu3);

// prototypes for user-defined BCs and srcterm functions
void UserBoundary(Mesh *pm);
void AddUserSrcs(Mesh *pm, const Real bdt);
void MDiskHistory(HistoryData *pdata, Mesh *pm);
Real MDiskTimeStep(Mesh *pm);
void Diagnostic(Mesh *pm);

} // namespace

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem()
//! \brief set initial conditions for Bondi accretion test
//  Compile with '-D PROBLEM=disk' to enroll as user-specific problem generator

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  user_bcs_func = UserBoundary;
  user_srcs_func = AddUserSrcs;
  user_hist_func = MDiskHistory;
  user_dt_func = MDiskTimeStep;

  bool is_mhd = (pmbp->pmhd != nullptr);
  bool is_gr = pmbp->pcoord->is_general_relativistic;
  auto &eos = is_mhd ? pmbp->pmhd->peos->eos_data : pmbp->phydro->peos->eos_data;

  mdisk.iso_cs = eos.iso_cs;
  mdisk.sink_d = pin->GetReal("problem","sink_d");
  mdisk.sink_p = pin->GetReal("problem","sink_p");
  mdisk.dens_0 = pin->GetReal("problem","dens_0");
  mdisk.j_z = pin->GetOrAddReal("problem","j_z",0.0);
  Real b_ini = mdisk.b_ini = pin->GetOrAddReal("problem","b_ini",0.0);
  mdisk.dt_floor = eos.dt_floor;
  mdisk.ndiag = pin->GetOrAddInteger("problem","ndiag",0);
  mdisk.bc_flag = pin->GetOrAddInteger("problem","bc_flag",0);

  // Spherical Grid for user-defined history
  auto &grids = spherical_grids;
  int hist_nr = pin->GetOrAddInteger("problem","hist_nr",4);
  for (int i=0; i<hist_nr; i++) {
    Real rmin = 1.0;
    Real rmax = pin->GetReal("mesh","x1max");
    Real r_i = std::pow(rmax/rmin,static_cast<Real>(i)/static_cast<Real>(hist_nr-1))*rmin;
    if (i==0) {
      r_i = is_gr ? 1.0+sqrt(1.0-SQR(pmbp->pcoord->coord_data.bh_spin)) : r_i;
    }
    grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, r_i));
  }

  if (restart) return;

  // capture variables for the kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  auto mdisk_ = mdisk;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int n1m1 = n1 - 1, n2m1 = n2 - 1, n3m1 = n3 - 1;
  int is = indcs.is, nx1 = indcs.nx1;
  int js = indcs.js, nx2 = indcs.nx2;
  int ks = indcs.ks, nx3 = indcs.nx3;
  int nmb = pmbp->nmb_thispack;
  auto w0 = is_mhd ? pmbp->pmhd->w0 : pmbp->phydro->w0;
  auto u0 = is_mhd ? pmbp->pmhd->u0 : pmbp->phydro->u0;

  // set initial conditions
  par_for("pgen_mdisk", DevExeSpace(), 0,(nmb-1),0,n3m1,0,n2m1,0,n1m1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real rho, uu1, uu2, uu3;
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

    ComputePrimitiveSingle(x1v,x2v,x3v,mdisk_,rho,uu1,uu2,uu3);
    w0(m,IDN,k,j,i) = rho;
    w0(m,IM1,k,j,i) = uu1;
    w0(m,IM2,k,j,i) = uu2;
    w0(m,IM3,k,j,i) = uu3;
  });

  // Add magnetic field
  if (is_mhd && b_ini>0.0) {
    auto &bcc0_ = pmbp->pmhd->bcc0;
    auto &b0_ = pmbp->pmhd->b0;
    par_for("pgen_mdisk_bfield", DevExeSpace(), 0,(nmb-1),0,n3m1,0,n2m1,0,n1m1,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      b0_.x1f(m,k,j,i) = 0.0;
      if (i==n1m1) b0_.x1f(m,k,j,i+1) = 0.0;
      b0_.x2f(m,k,j,i) = 0.0;
      if (j==n2m1) b0_.x2f(m,k,j+1,i) = 0.0;
      b0_.x3f(m,k,j,i) = b_ini;
      if (k==n3m1) b0_.x3f(m,k+1,j,i) = b_ini;
      bcc0_(m,IBX,k,j,i) = 0.0;
      bcc0_(m,IBY,k,j,i) = 0.0;
      bcc0_(m,IBZ,k,j,i) = b_ini;
    });
  }
  // Convert primitives to conserved
  if (pmbp->phydro != nullptr) {
    pmbp->phydro->peos->PrimToCons(w0, u0, 0, n1m1, 0, n2m1, 0, n3m1);
  } else if (pmbp->pmhd != nullptr) {
    auto &bcc0_ = pmbp->pmhd->bcc0;
    pmbp->pmhd->peos->PrimToCons(w0, bcc0_, u0, 0, n1m1, 0, n2m1, 0, n3m1);
  }

  return;
}

namespace {
//----------------------------------------------------------------------------------------
// Function for returning corresponding Boyer-Lindquist coordinates of point
// Inputs:
//   x1,x2,x3: global coordinates to be converted
// Outputs:
//   pr,ptheta,pphi: variables pointed to set to Boyer-Lindquist coordinates

KOKKOS_INLINE_FUNCTION
static void ComputePrimitiveSingle(Real x, Real y, Real z, struct mdisk_pgen pgen,
                                   Real& rho, Real& uu1, Real& uu2, Real& uu3) {
  Real r = sqrt(SQR(x) + SQR(y) + SQR(z));
  Real r_cyl = sqrt(SQR(x) + SQR(y));
  Real j_z = pgen.j_z;
  // gamma = 1.0
  Real iso_cs = pgen.iso_cs;
  rho = pgen.dens_0*fmax(exp((1.0/r-SQR(j_z/r_cyl)/2.0)/SQR(iso_cs)),0.01);
  uu1 = - j_z * y / SQR(r);
  uu2 = j_z * x / SQR(r);
  uu3 = 0.0;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn UserBoundary(Mesh *pm)
//  \brief Sets boundary condition on surfaces of computational domain
// Note quantities at boundaryies are held fixed to initial condition values

void UserBoundary(Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int n1m1 = n1 - 1, n2m1 = n2 - 1, n3m1 = n3 - 1;
  int &is = indcs.is; int &ie = indcs.ie, nx1 = indcs.nx1;
  int &js = indcs.js; int &je = indcs.je, nx2 = indcs.nx2;
  int &ks = indcs.ks; int &ke = indcs.ke, nx3 = indcs.nx3;
  auto &mb_bcs = pmbp->pmb->mb_bcs;
  int nmb = pmbp->nmb_thispack;

  bool is_mhd = (pmbp->pmhd != nullptr);
  auto u0_ = is_mhd ? pmbp->pmhd->u0 : pmbp->phydro->u0;
  auto w0_ = is_mhd ? pmbp->pmhd->w0 : pmbp->phydro->w0;
  int nvar = u0_.extent_int(1);
  auto &mdisk_ = mdisk;

  Real sink_d = mdisk.sink_d;
  bool bc_flag = mdisk.bc_flag;
  par_for("initial_radial", DevExeSpace(),0,nmb-1,ks-ng,ke+ng,js-ng,je+ng,is-ng,ie+ng,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

    Real rad = sqrt(SQR(x1v)+SQR(x2v)+SQR(x3v));

    if (rad < 1.0) {
      u0_(m,IDN,k,j,i) = sink_d;
      u0_(m,IM1,k,j,i) = 0.0;
      u0_(m,IM2,k,j,i) = 0.0;
      u0_(m,IM3,k,j,i) = 0.0;
    }
  });

  // B-field boundary conditions
  if (is_mhd) {
    auto &b0 = pm->pmb_pack->pmhd->b0;
    Real dt_floor = mdisk.dt_floor;
    par_for("initial_radial", DevExeSpace(),0,nmb-1,ks-ng,ke+ng,js-ng,je+ng,is-ng,ie+ng,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

      Real rad = sqrt(SQR(x1v)+SQR(x2v)+SQR(x3v));

      if (dt_floor > 0.0 && rad < 1.0) {
        Real va1_ceil = size.d_view(m).dx1/dt_floor;
        Real va2_ceil = size.d_view(m).dx2/dt_floor;
        Real va3_ceil = size.d_view(m).dx3/dt_floor;
        Real bx = 0.5*(b0.x1f(m,k,j,i) + b0.x1f(m,k,j,i+1));
        Real by = 0.5*(b0.x2f(m,k,j,i) + b0.x2f(m,k,j+1,i));
        Real bz = 0.5*(b0.x3f(m,k,j,i) + b0.x3f(m,k+1,j,i));
        u0_(m,IDN,k,j,i) = fmax(u0_(m,IDN,k,j,i),SQR(bx/va1_ceil));
        u0_(m,IDN,k,j,i) = fmax(u0_(m,IDN,k,j,i),SQR(by/va2_ceil));
        u0_(m,IDN,k,j,i) = fmax(u0_(m,IDN,k,j,i),SQR(bz/va3_ceil));
        // TODO(@mhguo): not working for ideal gas!
      }
    });
    // X1-Boundary
    par_for("noinflow_field_x1", DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n2-1),
    KOKKOS_LAMBDA(int m, int k, int j) {
      if (mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::user) {
        for (int i=0; i<ng; ++i) {
          b0.x1f(m,k,j,is-i-1) = b0.x1f(m,k,j,is);
          b0.x2f(m,k,j,is-i-1) = b0.x2f(m,k,j,is);
          if (j == n2-1) {b0.x2f(m,k,j+1,is-i-1) = b0.x2f(m,k,j+1,is);}
          b0.x3f(m,k,j,is-i-1) = b0.x3f(m,k,j,is);
          if (k == n3-1) {b0.x3f(m,k+1,j,is-i-1) = b0.x3f(m,k+1,j,is);}
        }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x1) == BoundaryFlag::user) {
        for (int i=0; i<ng; ++i) {
          b0.x1f(m,k,j,ie+i+2) = b0.x1f(m,k,j,ie+1);
          b0.x2f(m,k,j,ie+i+1) = b0.x2f(m,k,j,ie);
          if (j == n2-1) {b0.x2f(m,k,j+1,ie+i+1) = b0.x2f(m,k,j+1,ie);}
          b0.x3f(m,k,j,ie+i+1) = b0.x3f(m,k,j,ie);
          if (k == n3-1) {b0.x3f(m,k+1,j,ie+i+1) = b0.x3f(m,k+1,j,ie);}
        }
      }
    });
    // X2-Boundary
    par_for("noinflow_field_x2", DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n1-1),
    KOKKOS_LAMBDA(int m, int k, int i) {
      if (mb_bcs.d_view(m,BoundaryFace::inner_x2) == BoundaryFlag::user) {
        for (int j=0; j<ng; ++j) {
          b0.x1f(m,k,js-j-1,i) = b0.x1f(m,k,js,i);
          if (i == n1-1) {b0.x1f(m,k,js-j-1,i+1) = b0.x1f(m,k,js,i+1);}
          b0.x2f(m,k,js-j-1,i) = b0.x2f(m,k,js,i);
          b0.x3f(m,k,js-j-1,i) = b0.x3f(m,k,js,i);
          if (k == n3-1) {b0.x3f(m,k+1,js-j-1,i) = b0.x3f(m,k+1,js,i);}
        }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x2) == BoundaryFlag::user) {
        for (int j=0; j<ng; ++j) {
          b0.x1f(m,k,je+j+1,i) = b0.x1f(m,k,je,i);
          if (i == n1-1) {b0.x1f(m,k,je+j+1,i+1) = b0.x1f(m,k,je,i+1);}
          b0.x2f(m,k,je+j+2,i) = b0.x2f(m,k,je+1,i);
          b0.x3f(m,k,je+j+1,i) = b0.x3f(m,k,je,i);
          if (k == n3-1) {b0.x3f(m,k+1,je+j+1,i) = b0.x3f(m,k+1,je,i);}
        }
      }
    });
    // X3-Boundary
    par_for("noinflow_field_x3", DevExeSpace(),0,(nmb-1),0,(n2-1),0,(n1-1),
    KOKKOS_LAMBDA(int m, int j, int i) {
      if (mb_bcs.d_view(m,BoundaryFace::inner_x3) == BoundaryFlag::user) {
        for (int k=0; k<ng; ++k) {
          b0.x1f(m,ks-k-1,j,i) = b0.x1f(m,ks,j,i);
          if (i == n1-1) {b0.x1f(m,ks-k-1,j,i+1) = b0.x1f(m,ks,j,i+1);}
          b0.x2f(m,ks-k-1,j,i) = b0.x2f(m,ks,j,i);
          if (j == n2-1) {b0.x2f(m,ks-k-1,j+1,i) = b0.x2f(m,ks,j+1,i);}
          b0.x3f(m,ks-k-1,j,i) = b0.x3f(m,ks,j,i);
        }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x3) == BoundaryFlag::user) {
        for (int k=0; k<ng; ++k) {
          b0.x1f(m,ke+k+1,j,i) = b0.x1f(m,ke,j,i);
          if (i == n1-1) {b0.x1f(m,ke+k+1,j,i+1) = b0.x1f(m,ke,j,i+1);}
          b0.x2f(m,ke+k+1,j,i) = b0.x2f(m,ke,j,i);
          if (j == n2-1) {b0.x2f(m,ke+k+1,j+1,i) = b0.x2f(m,ke,j+1,i);}
          b0.x3f(m,ke+k+2,j,i) = b0.x3f(m,ke+1,j,i);
        }
      }
    });
  }

  // Primitive boundary conditions
  if (is_mhd) {
    auto &bcc0_ = pmbp->pmhd->bcc0;
    auto &b0_ = pmbp->pmhd->b0;
    pmbp->pmhd->peos->ConsToPrim(u0_,b0_,w0_,bcc0_,false,is-ng,is-1,0,n2m1,0,n3m1);
    pmbp->pmhd->peos->ConsToPrim(u0_,b0_,w0_,bcc0_,false,ie+1,ie+ng,0,n2m1,0,n3m1);
  } else {
    pmbp->phydro->peos->ConsToPrim(u0_,w0_,false,is-ng,is-1,0,n2m1,0,n3m1);
    pmbp->phydro->peos->ConsToPrim(u0_,w0_,false,ie+1,ie+ng,0,n2m1,0,n3m1);
  }
  if (bc_flag == 0) {
    // Set X1-BCs on w0 if Meshblock face is at the edge of computational domain
    par_for("noinflow_hydro_x1", DevExeSpace(),0,(nmb-1),0,(nvar-1),0,(n3-1),0,(n2-1),
    KOKKOS_LAMBDA(int m, int n, int k, int j) {
      if (mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::user) {
        for (int i=0; i<ng; ++i) {
          w0_(m,n,k,j,is-i-1) = w0_(m,n,k,j,is);
        }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x1) == BoundaryFlag::user) {
        for (int i=0; i<ng; ++i) {
          w0_(m,n,k,j,ie+i+1) = w0_(m,n,k,j,ie);
        }
      }
    });
  } else {
    par_for("fixed_x1", DevExeSpace(),0,(nmb-1),0,n3m1,0,n2m1,0,(ng-1),
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      // inner x1 boundary
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

      Real rho, uu1, uu2, uu3;
      if (mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::user) {
        ComputePrimitiveSingle(x1v,x2v,x3v,mdisk_,rho,uu1,uu2,uu3);
        w0_(m,IDN,k,j,i) = rho;
        w0_(m,IM1,k,j,i) = uu1;
        w0_(m,IM2,k,j,i) = uu2;
        w0_(m,IM3,k,j,i) = uu3;
      }

      // outer x1 boundary
      x1v = CellCenterX((ie+i+1)-is, indcs.nx1, x1min, x1max);

      if (mb_bcs.d_view(m,BoundaryFace::outer_x1) == BoundaryFlag::user) {
        ComputePrimitiveSingle(x1v,x2v,x3v,mdisk_, rho,uu1,uu2,uu3);
        w0_(m,IDN,k,j,(ie+i+1)) = rho;
        w0_(m,IM1,k,j,(ie+i+1)) = uu1;
        w0_(m,IM2,k,j,(ie+i+1)) = uu2;
        w0_(m,IM3,k,j,(ie+i+1)) = uu3;
      }
    });
  }
  // PrimToCons on X1 physical boundary ghost zones
  if (is_mhd) {
    auto &bcc0_ = pmbp->pmhd->bcc0;
    pmbp->pmhd->peos->PrimToCons(w0_,bcc0_,u0_,is-ng,is-1,0,n2m1,0,n3m1);
    pmbp->pmhd->peos->PrimToCons(w0_,bcc0_,u0_,ie+1,ie+ng,0,n2m1,0,n3m1);
  } else {
    pmbp->phydro->peos->PrimToCons(w0_,u0_,is-ng,is-1,0,n2m1,0,n3m1);
    pmbp->phydro->peos->PrimToCons(w0_,u0_,ie+1,ie+ng,0,n2m1,0,n3m1);
  }

  if (is_mhd) {
    auto &bcc0_ = pmbp->pmhd->bcc0;
    auto &b0_ = pmbp->pmhd->b0;
    pmbp->pmhd->peos->ConsToPrim(u0_,b0_,w0_,bcc0_,false,0,n1m1,js-ng,js-1,0,n3m1);
    pmbp->pmhd->peos->ConsToPrim(u0_,b0_,w0_,bcc0_,false,0,n1m1,je+1,je+ng,0,n3m1);
  } else {
    pmbp->phydro->peos->ConsToPrim(u0_,w0_,false,0,n1m1,js-ng,js-1,0,n3m1);
    pmbp->phydro->peos->ConsToPrim(u0_,w0_,false,0,n1m1,je+1,je+ng,0,n3m1);
  }
  if (bc_flag == 0) {
    // Set X2-BCs on w0 if Meshblock face is at the edge of computational domain
    par_for("noinflow_hydro_x2", DevExeSpace(),0,(nmb-1),0,(nvar-1),0,(n3-1),0,(n1-1),
    KOKKOS_LAMBDA(int m, int n, int k, int i) {
      if (mb_bcs.d_view(m,BoundaryFace::inner_x2) == BoundaryFlag::user) {
        for (int j=0; j<ng; ++j) {
          w0_(m,n,k,js-j-1,i) = w0_(m,n,k,js,i);
        }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x2) == BoundaryFlag::user) {
        for (int j=0; j<ng; ++j) {
          w0_(m,n,k,je+j+1,i) = w0_(m,n,k,je,i);
        }
      }
    });
  } else {
    par_for("fixed_x2", DevExeSpace(),0,(nmb-1),0,n3m1,0,(ng-1),0,n1m1,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      // inner x2 boundary
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

      Real rho, uu1, uu2, uu3;
      if (mb_bcs.d_view(m,BoundaryFace::inner_x2) == BoundaryFlag::user) {
        ComputePrimitiveSingle(x1v,x2v,x3v,mdisk_,rho,uu1,uu2,uu3);
        w0_(m,IDN,k,j,i) = rho;
        w0_(m,IM1,k,j,i) = uu1;
        w0_(m,IM2,k,j,i) = uu2;
        w0_(m,IM3,k,j,i) = uu3;
      }

      // outer x2 boundary
      x2v = CellCenterX((je+j+1)-js, indcs.nx2, x2min, x2max);

      if (mb_bcs.d_view(m,BoundaryFace::outer_x2) == BoundaryFlag::user) {
        ComputePrimitiveSingle(x1v,x2v,x3v,mdisk_,rho,uu1,uu2,uu3);
        w0_(m,IDN,k,(je+j+1),i) = rho;
        w0_(m,IM1,k,(je+j+1),i) = uu1;
        w0_(m,IM2,k,(je+j+1),i) = uu2;
        w0_(m,IM3,k,(je+j+1),i) = uu3;
      }
    });
  }
  if (is_mhd) {
    auto &bcc0_ = pmbp->pmhd->bcc0;
    pmbp->pmhd->peos->PrimToCons(w0_,bcc0_,u0_,0,n1m1,js-ng,js-1,0,n3m1);
    pmbp->pmhd->peos->PrimToCons(w0_,bcc0_,u0_,0,n1m1,je+1,je+ng,0,n3m1);
  } else {
    pmbp->phydro->peos->PrimToCons(w0_,u0_,0,n1m1,js-ng,js-1,0,n3m1);
    pmbp->phydro->peos->PrimToCons(w0_,u0_,0,n1m1,je+1,je+ng,0,n3m1);
  }

  if (is_mhd) {
    auto &bcc0_ = pmbp->pmhd->bcc0;
    auto &b0_ = pmbp->pmhd->b0;
    pmbp->pmhd->peos->ConsToPrim(u0_,b0_,w0_,bcc0_,false,0,n1m1,0,n2m1,ks-ng,ks-1);
    pmbp->pmhd->peos->ConsToPrim(u0_,b0_,w0_,bcc0_,false,0,n1m1,0,n2m1,ke+1,ke+ng);
  } else {
    pmbp->phydro->peos->ConsToPrim(u0_,w0_,false,0,n1m1,0,n2m1,ks-ng,ks-1);
    pmbp->phydro->peos->ConsToPrim(u0_,w0_,false,0,n1m1,0,n2m1,ke+1,ke+ng);
  }
  if (bc_flag == 0) {
    // Set X3-BCs on w0 if Meshblock face is at the edge of computational domain
    par_for("noinflow_hydro_x3", DevExeSpace(),0,(nmb-1),0,(nvar-1),0,(n2-1),0,(n1-1),
    KOKKOS_LAMBDA(int m, int n, int j, int i) {
      if (mb_bcs.d_view(m,BoundaryFace::inner_x3) == BoundaryFlag::user) {
        for (int k=0; k<ng; ++k) {
          w0_(m,n,ks-k-1,j,i) = w0_(m,n,ks,j,i);
        }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x3) == BoundaryFlag::user) {
        for (int k=0; k<ng; ++k) {
          w0_(m,n,ke+k+1,j,i) = w0_(m,n,ke,j,i);
        }
      }
    });
  } else {
    par_for("fixed_ix3", DevExeSpace(),0,(nmb-1),0,(ng-1),0,n2m1,0,n1m1,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      // inner x3 boundary
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

      Real rho, uu1, uu2, uu3;
      if (mb_bcs.d_view(m,BoundaryFace::inner_x3) == BoundaryFlag::user) {
        ComputePrimitiveSingle(x1v,x2v,x3v,mdisk_,rho,uu1,uu2,uu3);
        w0_(m,IDN,k,j,i) = rho;
        w0_(m,IM1,k,j,i) = uu1;
        w0_(m,IM2,k,j,i) = uu2;
        w0_(m,IM3,k,j,i) = uu3;
      }

      // outer x3 boundary
      x3v = CellCenterX((ke+k+1)-ks, indcs.nx3, x3min, x3max);

      if (mb_bcs.d_view(m,BoundaryFace::outer_x3) == BoundaryFlag::user) {
        ComputePrimitiveSingle(x1v,x2v,x3v,mdisk_,rho,uu1,uu2,uu3);
        w0_(m,IDN,(ke+k+1),j,i) = rho;
        w0_(m,IM1,(ke+k+1),j,i) = uu1;
        w0_(m,IM2,(ke+k+1),j,i) = uu2;
        w0_(m,IM3,(ke+k+1),j,i) = uu3;
      }
    });
  }
  if (is_mhd) {
    auto &bcc0_ = pmbp->pmhd->bcc0;
    pmbp->pmhd->peos->PrimToCons(w0_,bcc0_,u0_,0,n1m1,0,n2m1,ks-ng,ks-1);
    pmbp->pmhd->peos->PrimToCons(w0_,bcc0_,u0_,0,n1m1,0,n2m1,ke+1,ke+ng);
  } else {
    pmbp->phydro->peos->PrimToCons(w0_,u0_,0,n1m1,0,n2m1,ks-ng,ks-1);
    pmbp->phydro->peos->PrimToCons(w0_,u0_,0,n1m1,0,n2m1,ke+1,ke+ng);
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void AddUserSrcs()
//! \brief Add User Source Terms
// NOTE source terms must all be computed using primitive (w0) and NOT conserved (u0) vars
void AddUserSrcs(Mesh *pm, const Real bdt) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  int is_mhd = (pmbp->pmhd != nullptr);
  DvceArray5D<Real> &u0 = is_mhd ? pmbp->pmhd->u0 : pmbp->phydro->u0;
  DvceArray5D<Real> &w0 = is_mhd ? pmbp->pmhd->w0 : pmbp->phydro->w0;
  // capture variables for the kernel
  auto &indcs = pm->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int n1m1 = n1 - 1, n2m1 = n2 - 1, n3m1 = n3 - 1;
  int is = indcs.is, nx1 = indcs.nx1;
  int js = indcs.js, nx2 = indcs.nx2;
  int ks = indcs.ks, nx3 = indcs.nx3;
  int nmb = pmbp->nmb_thispack;
  par_for("add_accel", DevExeSpace(), 0,(nmb-1),0,n3m1,0,n2m1,0,n1m1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

    Real rad = sqrt(SQR(x1v)+SQR(x2v)+SQR(x3v));

    Real accel = Acceleration(rad);
    Real dmomr = bdt*w0(m,IDN,k,j,i)*accel;
    u0(m,IM1,k,j,i) += dmomr*x1v/rad;
    u0(m,IM2,k,j,i) += dmomr*x2v/rad;
    u0(m,IM3,k,j,i) += dmomr*x3v/rad;
  });
  return;
}

//----------------------------------------------------------------------------------------
//! \fn MDiskHistory
//! \brief User-defined history output

void MDiskHistory(HistoryData *pdata, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  int nvars; bool is_ideal = false; Real gamma; bool is_mhd = false;
  DvceArray5D<Real> w0_, bcc0_;
  if (pmbp->phydro != nullptr) {
    nvars = pmbp->phydro->nhydro + pmbp->phydro->nscalars;
    is_ideal = pmbp->phydro->peos->eos_data.is_ideal;
    gamma = pmbp->phydro->peos->eos_data.gamma;
    w0_ = pmbp->phydro->w0;
  } else if (pmbp->pmhd != nullptr) {
    is_mhd = true;
    nvars = pmbp->pmhd->nmhd + pmbp->pmhd->nscalars;
    is_ideal = pmbp->pmhd->peos->eos_data.is_ideal;
    gamma = pmbp->pmhd->peos->eos_data.gamma;
    w0_ = pmbp->pmhd->w0;
    bcc0_ = pmbp->pmhd->bcc0;
  }
  // extract grids, number of radii, number of fluxes, and history appending index
  auto &grids = pm->pgen->spherical_grids;
  int nradii = grids.size();
  //const int nflux = (is_mhd) ? 7 : 6;
  const int nflux = 10;
  // set number of and names of history variables for hydro or mhd
  //  (0) mass
  //  (1) mass accretion rate
  //  (2) energy flux
  //  (3) angular momentum flux * 3
  //  (4) magnetic flux (iff MHD)
  int nsph = nradii * nflux;
  int nuser = nsph;
  pdata->nhist = nuser;
  for (int g=0; g<nradii; ++g) {
    std::string rstr = std::to_string(g);
    pdata->label[nflux*g+0] = "m_" + rstr;
    pdata->label[nflux*g+1] = "mdot" + rstr;
    pdata->label[nflux*g+2] = "mdout" + rstr;
    pdata->label[nflux*g+3] = "mdh" + rstr;
    pdata->label[nflux*g+4] = "edot" + rstr;
    pdata->label[nflux*g+5] = "edout" + rstr;
    pdata->label[nflux*g+6] = "lx" + rstr;
    pdata->label[nflux*g+7] = "ly" + rstr;
    pdata->label[nflux*g+8] = "lz" + rstr;
    pdata->label[nflux*g+9] = "phi" + rstr;
  }

  // go through angles at each radii:
  DualArray2D<Real> interpolated_bcc;  // needed for MHD
  for (int g=0; g<nradii; ++g) {
    // zero fluxes at this radius
    for (int i=0; i<nflux; ++i) {
      pdata->hdata[nflux*g+i] = 0.0;
    }
    // interpolate primitives (and cell-centered magnetic fields iff mhd)
    if (is_mhd) {
      grids[g]->InterpolateToSphere(3, bcc0_);
      Kokkos::realloc(interpolated_bcc, grids[g]->nangles, 3);
      Kokkos::deep_copy(interpolated_bcc, grids[g]->interp_vals);
      interpolated_bcc.template modify<DevExeSpace>();
      interpolated_bcc.template sync<HostMemSpace>();
    }
    grids[g]->InterpolateToSphere(nvars, w0_);

    // compute fluxes
    bool is_gr = pmbp->pcoord->is_general_relativistic;
    if (is_gr) {
      // extract BH parameters
      bool &flat = pmbp->pcoord->coord_data.is_minkowski;
      Real &spin = pmbp->pcoord->coord_data.bh_spin;
      for (int n=0; n<grids[g]->nangles; ++n) {
        // extract coordinate data at this angle
        Real r = grids[g]->radius;
        Real theta = grids[g]->polar_pos.h_view(n,0);
        Real phi = grids[g]->polar_pos.h_view(n,1);
        Real x1 = grids[g]->interp_coord.h_view(n,0);
        Real x2 = grids[g]->interp_coord.h_view(n,1);
        Real x3 = grids[g]->interp_coord.h_view(n,2);
        Real glower[4][4], gupper[4][4];
        ComputeMetricAndInverse(x1,x2,x3,flat,spin,glower,gupper);

        // extract interpolated primitives
        Real &int_dn = grids[g]->interp_vals.h_view(n,IDN);
        Real &int_vx = grids[g]->interp_vals.h_view(n,IVX);
        Real &int_vy = grids[g]->interp_vals.h_view(n,IVY);
        Real &int_vz = grids[g]->interp_vals.h_view(n,IVZ);
        Real int_ie = is_ideal ? grids[g]->interp_vals.h_view(n,IEN) : 0.0;

        // extract interpolated field components (iff is_mhd)
        Real int_bx = 0.0, int_by = 0.0, int_bz = 0.0;
        if (is_mhd) {
          int_bx = interpolated_bcc.h_view(n,IBX);
          int_by = interpolated_bcc.h_view(n,IBY);
          int_bz = interpolated_bcc.h_view(n,IBZ);
        }

        // Compute interpolated u^\mu in CKS
        Real q = glower[1][1]*int_vx*int_vx + 2.0*glower[1][2]*int_vx*int_vy +
                 2.0*glower[1][3]*int_vx*int_vz + glower[2][2]*int_vy*int_vy +
                 2.0*glower[2][3]*int_vy*int_vz + glower[3][3]*int_vz*int_vz;
        Real alpha = sqrt(-1.0/gupper[0][0]);
        Real lor = sqrt(1.0 + q);
        Real u0 = lor/alpha;
        Real u1 = int_vx - alpha * lor * gupper[0][1];
        Real u2 = int_vy - alpha * lor * gupper[0][2];
        Real u3 = int_vz - alpha * lor * gupper[0][3];

        // Lower vector indices
        Real u_0 = glower[0][0]*u0 + glower[0][1]*u1 + glower[0][2]*u2 + glower[0][3]*u3;
        Real u_1 = glower[1][0]*u0 + glower[1][1]*u1 + glower[1][2]*u2 + glower[1][3]*u3;
        Real u_2 = glower[2][0]*u0 + glower[2][1]*u1 + glower[2][2]*u2 + glower[2][3]*u3;
        Real u_3 = glower[3][0]*u0 + glower[3][1]*u1 + glower[3][2]*u2 + glower[3][3]*u3;

        // Calculate 4-magnetic field (returns zero if not MHD)
        Real b0 = u_1*int_bx + u_2*int_by + u_3*int_bz;
        Real b1 = (int_bx + b0 * u1) / u0;
        Real b2 = (int_by + b0 * u2) / u0;
        Real b3 = (int_bz + b0 * u3) / u0;

        // compute b_\mu in CKS and b_sq (returns zero if not MHD)
        Real b_0 = glower[0][0]*b0 + glower[0][1]*b1 + glower[0][2]*b2 + glower[0][3]*b3;
        Real b_1 = glower[1][0]*b0 + glower[1][1]*b1 + glower[1][2]*b2 + glower[1][3]*b3;
        Real b_2 = glower[2][0]*b0 + glower[2][1]*b1 + glower[2][2]*b2 + glower[2][3]*b3;
        Real b_3 = glower[3][0]*b0 + glower[3][1]*b1 + glower[3][2]*b2 + glower[3][3]*b3;
        Real b_sq = b0*b_0 + b1*b_1 + b2*b_2 + b3*b_3;

        // Transform CKS 4-velocity and 4-magnetic field to spherical KS
        Real a2 = SQR(spin);
        Real rad2 = SQR(x1)+SQR(x2)+SQR(x3);
        Real r2 = SQR(r);
        Real sth = sin(theta);
        Real sph = sin(phi);
        Real cph = cos(phi);
        Real drdx = r*x1/(2.0*r2 - rad2 + a2);
        Real drdy = r*x2/(2.0*r2 - rad2 + a2);
        Real drdz = (r*x3 + a2*x3/r)/(2.0*r2-rad2+a2);
        // contravariant r component of 4-velocity
        Real ur = drdx *u1 + drdy *u2 + drdz *u3;
        // contravariant r component of 4-magnetic field (returns zero if not MHD)
        Real br = drdx *b1 + drdy *b2 + drdz *b3;
        // covariant phi component of 4-velocity
        Real u_ph = (-r*sph-spin*cph)*sth*u_1 + (r*cph-spin*sph)*sth*u_2;
        // covariant phi component of 4-magnetic field (returns zero if not MHD)
        Real b_ph = (-r*sph-spin*cph)*sth*b_1 + (r*cph-spin*sph)*sth*b_2;

        // integration params
        Real &domega = grids[g]->solid_angles.h_view(n);
        Real sqrtmdet = (r2+SQR(spin*cos(theta)));

        // compute temperature and compare with hot/cold threshold
        Real is_out = (ur>0.0)? 1.0 : 0.0;
        Real is_in = (ur<0.0)? 1.0 : 0.0;

        // compute mass density
        pdata->hdata[nflux*g+0] += int_dn*sqrtmdet*domega;

        // compute mass flux
        pdata->hdata[nflux*g+1] += 1.0*int_dn*ur*sqrtmdet*domega;
        pdata->hdata[nflux*g+2] += is_out*int_dn*ur*sqrtmdet*domega;
        pdata->hdata[nflux*g+3] += is_in*int_dn*ur*sqrtmdet*domega;

        // compute energy flux
        Real t1_0 = (int_dn + gamma*int_ie + b_sq)*ur*u_0 - br*b_0;
        pdata->hdata[nflux*g+4] += 1.0*t1_0*sqrtmdet*domega;
        pdata->hdata[nflux*g+5] += is_out*t1_0*sqrtmdet*domega;

        // compute angular momentum flux
        // TODO(@mhguo): write a correct function to compute x,y angular momentum flux
        Real t1_1 = 0.0;
        Real t1_2 = 0.0;
        Real t1_3 = (int_dn + gamma*int_ie + b_sq)*ur*u_ph - br*b_ph;
        pdata->hdata[nflux*g+6] += t1_1*sqrtmdet*domega;
        pdata->hdata[nflux*g+7] += t1_2*sqrtmdet*domega;
        pdata->hdata[nflux*g+8] += t1_3*sqrtmdet*domega;

        // compute magnetic flux
        if (is_mhd) {
          pdata->hdata[nflux*g+9] += 0.5*fabs(br*u0 - b0*ur)*sqrtmdet*domega;
        }
      }
    } else {
      for (int n=0; n<grids[g]->nangles; ++n) {
        // extract coordinate data at this angle
        Real r = grids[g]->radius;
        Real x1 = grids[g]->interp_coord.h_view(n,0);
        Real x2 = grids[g]->interp_coord.h_view(n,1);
        Real x3 = grids[g]->interp_coord.h_view(n,2);
        // extract interpolated primitives
        Real &int_dn = grids[g]->interp_vals.h_view(n,IDN);
        Real &int_vx = grids[g]->interp_vals.h_view(n,IVX);
        Real &int_vy = grids[g]->interp_vals.h_view(n,IVY);
        Real &int_vz = grids[g]->interp_vals.h_view(n,IVZ);
        Real int_ie = is_ideal ? grids[g]->interp_vals.h_view(n,IEN) : 0.0;
        // extract interpolated field components (iff is_mhd)
        Real int_bx = 0.0, int_by = 0.0, int_bz = 0.0;
        if (is_mhd) {
          int_bx = interpolated_bcc.h_view(n,IBX);
          int_by = interpolated_bcc.h_view(n,IBY);
          int_bz = interpolated_bcc.h_view(n,IBZ);
        }

        Real v1 = int_vx, v2 = int_vy, v3 = int_vz;
        Real e_k = 0.5*int_dn*(v1*v1+v2*v2+v3*v3);
        Real b1 = int_bx, b2 = int_by, b3 = int_bz;
        Real b_sq = b1*b1 + b2*b2 + b3*b3;
        Real r_sq = SQR(r);
        Real drdx = x1/r;
        Real drdy = x2/r;
        Real drdz = x3/r;
        // v_r
        Real vr = drdx*v1 + drdy*v2 + drdz*v3;
        // b_r
        Real br = drdx*b1 + drdy*b2 + drdz*b3;
        // integration params
        Real &domega = grids[g]->solid_angles.h_view(n);
        Real is_out = (vr>0.0)? 1.0 : 0.0;
        Real is_in = (vr<0.0)? 1.0 : 0.0;

        // compute mass density
        pdata->hdata[nflux*g+0] += int_dn*r_sq*domega;

        // compute mass flux
        pdata->hdata[nflux*g+1] += 1.0*int_dn*vr*r_sq*domega;
        pdata->hdata[nflux*g+2] += is_out*int_dn*vr*r_sq*domega;
        pdata->hdata[nflux*g+3] += is_in*int_dn*vr*r_sq*domega;

        // compute energy flux
        // TODO(@mhguo): check whether this is correct!
        Real t1_0 = (int_ie + e_k + 0.5*b_sq)*vr;
        pdata->hdata[nflux*g+4] += 1.0*t1_0*r_sq*domega;
        pdata->hdata[nflux*g+5] += is_out*t1_0*r_sq*domega;

        // compute angular momentum flux
        // TODO(@mhguo): check whether this is correct!
        pdata->hdata[nflux*g+6] += int_dn*(x2*v3-x3*v2)*r_sq*domega;
        pdata->hdata[nflux*g+7] += int_dn*(x3*v1-x1*v3)*r_sq*domega;
        pdata->hdata[nflux*g+8] += int_dn*(x1*v2-x2*v1)*r_sq*domega;

        // compute magnetic flux
        if (is_mhd) {
          pdata->hdata[nflux*g+9] += 0.5*fabs(br)*r_sq*domega;
        }
      }
    }
  }
  return;
}


//----------------------------------------------------------------------------------------
//! \fn Real MDiskTimeStep()
//! \brief User-defined time step function

Real MDiskTimeStep(Mesh *pm) {
  if (mdisk.ndiag>0 && pm->ncycle % mdisk.ndiag == 0) {
    Diagnostic(pm);
  }
  Real dt = pm->dt/pm->cfl_no;
  return dt;
}

//----------------------------------------------------------------------------------------
//! \fn void Diagnostic()
//! \brief Compute diagnostics.
// NOTE source terms must all be computed using primitive (w0) and NOT conserved (u0) vars

void Diagnostic(Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  DvceArray5D<Real> &w0 = (pmbp->pmhd != nullptr) ? pmbp->pmhd->w0 : pmbp->phydro->w0;
  const EOS_Data &eos_data = (pmbp->pmhd != nullptr) ?
                             pmbp->pmhd->peos->eos_data : pmbp->phydro->peos->eos_data;
  auto &indcs = pmbp->pmesh->mb_indcs;
  int is = indcs.is; int nx1 = indcs.nx1;
  int js = indcs.js; int nx2 = indcs.nx2;
  int ks = indcs.ks; int nx3 = indcs.nx3;
  auto &size = pmbp->pmb->mb_size;
  const int nmkji = (pmbp->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;

  Real gamma = eos_data.gamma;
  Real gm1 = gamma - 1.0;

  Real dtnew = std::numeric_limits<Real>::max();

  Real min_dens = std::numeric_limits<Real>::max();
  Real min_vtot = std::numeric_limits<Real>::max();
  Real min_temp = std::numeric_limits<Real>::max();
  Real min_eint = std::numeric_limits<Real>::max();
  Real max_dens = std::numeric_limits<Real>::min();
  Real max_vtot = std::numeric_limits<Real>::min();
  Real max_temp = std::numeric_limits<Real>::min();
  Real max_eint = std::numeric_limits<Real>::min();
  Real max_bfld = std::numeric_limits<Real>::min();
  Real max_valf = std::numeric_limits<Real>::min();
  Real min_dtva = std::numeric_limits<Real>::max();

  // find smallest (e/cooling_rate) in each cell
  Kokkos::parallel_reduce("diagnostic", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &min_dt, Real &min_d, Real &min_v, Real &min_t,
  Real &min_e, Real &max_d, Real &max_v, Real &max_t, Real &max_e) {
    // compute m,k,j,i indices of thread and call function
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;

    Real dx = fmin(fmin(size.d_view(m).dx1,size.d_view(m).dx2),size.d_view(m).dx3);

    // temperature in cgs unit
    Real temp = w0(m,IEN,k,j,i)/w0(m,IDN,k,j,i)*gm1;
    Real eint = w0(m,IEN,k,j,i);

    Real vtot = sqrt(SQR(w0(m,IVX,k,j,i))+SQR(w0(m,IVY,k,j,i))+SQR(w0(m,IVZ,k,j,i)));
    min_dt = fmin(dx/sqrt(gamma*temp), min_dt);
    min_d = fmin(w0(m,IDN,k,j,i), min_d);
    min_v = fmin(vtot,min_v);
    min_t = fmin(temp, min_t);
    min_e = fmin(eint, min_e);
    max_d = fmax(w0(m,IDN,k,j,i), max_d);
    max_v = fmax(vtot,max_v);
    max_t = fmax(temp, max_t);
    max_e = fmax(eint, max_e);
  }, Kokkos::Min<Real>(dtnew), Kokkos::Min<Real>(min_dens), Kokkos::Min<Real>(min_vtot),
  Kokkos::Min<Real>(min_temp), Kokkos::Min<Real>(min_eint), Kokkos::Max<Real>(max_dens),
  Kokkos::Max<Real>(max_vtot), Kokkos::Max<Real>(max_temp), Kokkos::Max<Real>(max_eint));

  if (pmbp->pmhd != nullptr) {
    auto bcc = pmbp->pmhd->bcc0;
    Kokkos::parallel_reduce("diagnostic", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, Real &max_b, Real &max_va, Real &min_dta) {
      // compute m,k,j,i indices of thread and call function
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks;
      j += js;

      Real dx = fmin(fmin(size.d_view(m).dx1,size.d_view(m).dx2),size.d_view(m).dx3);
      Real btot = sqrt(SQR(bcc(m,IBX,k,j,i))+SQR(bcc(m,IBY,k,j,i))+SQR(bcc(m,IBZ,k,j,i)));
      Real va = btot/sqrt(w0(m,IDN,k,j,i));
      Real dta = dx/fmax(va,1e-30);
      max_b = fmax(btot,max_b);
      max_va = fmax(va,max_va);
      min_dta = fmin(dta,min_dta);
    }, Kokkos::Max<Real>(max_bfld), Kokkos::Max<Real>(max_valf),
    Kokkos::Min<Real>(min_dtva));
  }
#if MPI_PARALLEL_ENABLED
  Real m_min[6] = {dtnew,min_dens,min_vtot,min_temp,min_eint,min_dtva};
  Real m_max[6] = {max_dens,max_vtot,max_temp,max_eint,max_bfld,max_valf};
  Real gm_min[6];
  Real gm_max[6];
  //MPI_Allreduce(MPI_IN_PLACE, &dtnew, 1, MPI_ATHENA_REAL, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(m_min, gm_min, 6, MPI_ATHENA_REAL, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(m_max, gm_max, 6, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
  dtnew = gm_min[0];
  min_dens = gm_min[1];
  min_vtot = gm_min[2];
  min_temp = gm_min[3];
  min_eint = gm_min[4];
  min_dtva = gm_min[5];
  max_dens = gm_max[0];
  max_vtot = gm_max[1];
  max_temp = gm_max[2];
  max_eint = gm_max[3];
  max_bfld = gm_max[4];
  max_valf = gm_max[5];
#endif
  if (global_variable::my_rank == 0) {
    std::cout << " min_d=" << min_dens << " max_d=" << max_dens << std::endl
              << " min_v=" << min_vtot << " max_v=" << max_vtot << std::endl
              << " min_t=" << min_temp << " max_t=" << max_temp << std::endl
              << " min_e=" << min_eint << " max_e=" << max_eint << std::endl
              << " dt_cs=" << dtnew;
    if (pmbp->pmhd != nullptr) {
      std::cout << " dt_va=" << min_dtva << std::endl
                << " max_b=" << max_bfld << " max_va=" << max_valf;
    }
    std::cout << std::endl;
  }
  return;
}

} // namespace
