//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file xray_binary.cpp
//! \brief Problem generator for x-ray binary with donor on +x and accretor at origin.
//! Supports Newtonian Cartesian hydro and GR hydro (+ optional radiation).

#include <stdio.h>
#include <math.h>

#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "geodesic-grid/spherical_grid.hpp"
#include "hydro/hydro.hpp"
#include "radiation/radiation.hpp"

namespace {
KOKKOS_INLINE_FUNCTION
static void GetBoyerLindquistCoordinates(struct xrb_pgen pgen,
                                         Real x1, Real x2, Real x3,
                                         Real *pr, Real *ptheta, Real *pphi);

KOKKOS_INLINE_FUNCTION
Real CalculatePotential(struct xrb_pgen pgen, Real x1, Real x2, Real x3);

KOKKOS_INLINE_FUNCTION
static void CalculateRocheAcceleration(struct xrb_pgen pgen,
                                     Real x1, Real x2, Real x3,
                                     Real vx, Real vy, Real vz,
                                     Real *ax, Real *ay, Real *az);

// Useful container for physical parameters of the binary.
// Geometry: accretor at origin, donor centered at (+a_sep, 0, 0).
struct xrb_pgen {
  bool is_gr;                                 // true: GR Kerr-Schild; false: Newtonian Cartesian
  Real spin;                                  // black hole spin (GR only)
  Real dexcise, pexcise;                      // excision parameters (GR only)
  Real r_soft;                                // softening radius at accretor (Newtonian only)
  Real gamma_adi;                             // EOS adiabatic index
  Real k_adi;                                 // EOS entropy constant for polytrope
  Real a_sep;                                 // binary separation
  Real m_donor;                               // donor star mass
  Real m_accretor;                            // accretor star mass (1 in GR units)
  Real r_donor;                               // donor star radius
  Real mass_ratio;                            // mass ratio: m_accretor / m_total
  Real rho_min, rho_pow, pgas_min, pgas_pow;  // background atmosphere parameters
};

xrb_pgen xrb;

} // namespace

void NoInflowXRB(Mesh *pm);
void XRBSourceTerms(Mesh *pm, const Real bdt);
void AccretorFluxes(HistoryData *pdata, Mesh *pm);

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->phydro == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "xray_binary requires a <hydro> block" << std::endl;
    exit(EXIT_FAILURE);
  }

  const bool is_gr = pmbp->pcoord->is_general_relativistic;
  const bool is_radiation_enabled = (pmbp->prad != nullptr);
  if (!is_gr && is_radiation_enabled) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Newtonian xray_binary does not support <radiation>" << std::endl;
    exit(EXIT_FAILURE);
  }

  user_bcs_func = NoInflowXRB;
  user_srcs_func = XRBSourceTerms;
  user_hist_func = AccretorFluxes;

  auto &indcs = pmy_mesh_->mb_indcs;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int ie = indcs.ie, je = indcs.je, ke = indcs.ke;
  auto &coord = pmbp->pcoord->coord_data;

  xrb.is_gr = is_gr;
  xrb.spin = coord.bh_spin;
  xrb.dexcise = coord.dexcise;
  xrb.pexcise = coord.pexcise;
  xrb.r_soft = pin->GetOrAddReal("problem", "r_soft", 0.01);

  auto &grids = spherical_grids;
  Real rflux_inner;
  if (is_gr) {
    const Real r_excise = coord.rexcise;
    rflux_inner = (is_radiation_enabled) ? ceil(r_excise + 1.0)
                                         : 1.0 + sqrt(1.0 - SQR(xrb.spin));
  } else {
    rflux_inner = pin->GetOrAddReal("problem", "flux_radius_inner", xrb.r_soft);
  }
  grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, rflux_inner));
  grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, 12.0));
  grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, 24.0));

  xrb.gamma_adi = pmbp->phydro->peos->eos_data.gamma;
  xrb.k_adi = pin->GetReal("problem", "k_adi");
  xrb.a_sep = pin->GetReal("problem", "a_sep");
  xrb.m_donor = pin->GetReal("problem", "m_donor");
  xrb.m_accretor = pin->GetOrAddReal("problem", "m_accretor", 1.0);
  xrb.r_donor = pin->GetReal("problem", "r_donor");
  xrb.mass_ratio = xrb.m_accretor / (xrb.m_donor + xrb.m_accretor);
  xrb.rho_min = pin->GetReal("problem", "rho_min");
  xrb.rho_pow = pin->GetReal("problem", "rho_pow");
  xrb.pgas_min = pin->GetReal("problem", "pgas_min");
  xrb.pgas_pow = pin->GetReal("problem", "pgas_pow");

  if (restart) return;

  auto &u0_ = pmbp->phydro->u0;
  auto &w0_ = pmbp->phydro->w0;

  int nangles_ = 0;
  DualArray2D<Real> nh_c_;
  DvceArray6D<Real> norm_to_tet_, tet_c_, tetcov_c_;
  DvceArray5D<Real> i0_;
  if (is_gr && is_radiation_enabled) {
    nangles_ = pmbp->prad->prgeo->nangles;
    nh_c_ = pmbp->prad->nh_c;
    norm_to_tet_ = pmbp->prad->norm_to_tet;
    tet_c_ = pmbp->prad->tet_c;
    tetcov_c_ = pmbp->prad->tetcov_c;
    i0_ = pmbp->prad->i0;
  }

  Real gm1 = xrb.gamma_adi - 1.0;

  auto pgen = xrb;
  auto &size = pmbp->pmb->mb_size;
  const int nmkji = (pmbp->nmb_thispack)*indcs.nx3*indcs.nx2*indcs.nx1;
  const int nkji = indcs.nx3*indcs.nx2*indcs.nx1;
  const int nji  = indcs.nx2*indcs.nx1;

  par_for("pgen_xrb", DevExeSpace(), 0, nmkji,
  KOKKOS_LAMBDA(const int idx) {
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/indcs.nx1;
    int i = (idx - m*nkji - k*nji - j*indcs.nx1) + is;
    k += ks;
    j += js;

    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    Real &dx1 = size.d_view(m).dx1;
    Real &dx2 = size.d_view(m).dx2;
    Real &dx3 = size.d_view(m).dx3;

    Real rho_bg, pgas_bg;
    if (pgen.is_gr) {
      Real r, theta, phi;
      GetBoyerLindquistCoordinates(pgen, x1v, x2v, x3v, &r, &theta, &phi);

      Real r_excise_cell, theta_excise, phi_excise;
      GetBoyerLindquistCoordinates(pgen, x1v + copysign(0.5*dx1, x1v),
                                        x2v + copysign(0.5*dx2, x2v),
                                        x3v + copysign(0.5*dx3, x3v),
                                        &r_excise_cell, &theta_excise, &phi_excise);
      if (r_excise_cell > 1.0) {
        rho_bg = pgen.rho_min * pow(r, pgen.rho_pow);
        pgas_bg = pgen.pgas_min * pow(r, pgen.pgas_pow);
      } else {
        rho_bg = pgen.dexcise;
        pgas_bg = pgen.pexcise;
      }
    } else {
      Real r = sqrt(SQR(x1v) + SQR(x2v) + SQR(x3v));
      Real r_eff = fmax(r, pgen.r_soft);
      rho_bg = pgen.rho_min * pow(r_eff, pgen.rho_pow);
      pgas_bg = pgen.pgas_min * pow(r_eff, pgen.pgas_pow);
    }

    Real rho = rho_bg;
    Real pgas = pgas_bg;
    Real uu1 = 0.0;
    Real uu2 = 0.0;
    Real uu3 = 0.0;
    const Real urad = 0.0;

    Real potential_surface =
      CalculatePotential(pgen, pgen.a_sep - pgen.r_donor, 0.0, 0.0);
    Real potential = CalculatePotential(pgen, x1v, x2v, x3v);
    Real x_d = x1v - pgen.a_sep;
    Real d_donor = sqrt(SQR(x_d) + SQR(x2v) + SQR(x3v));
    Real delta_phi = potential_surface - potential;
    if (d_donor < pgen.r_donor && delta_phi > 0.0) {
      rho = pow((gm1/(pgen.gamma_adi*pgen.k_adi))*delta_phi, 1.0/gm1);
      pgas = pgen.k_adi * pow(rho, pgen.gamma_adi);
    }

    w0_(m,IDN,k,j,i) = fmax(rho, rho_bg);
    w0_(m,IEN,k,j,i) = fmax(pgas, pgas_bg) / gm1;
    w0_(m,IVX,k,j,i) = 0.0;
    w0_(m,IVY,k,j,i) = 0.0;
    w0_(m,IVZ,k,j,i) = 0.0;

    if (pgen.is_gr && is_radiation_enabled) {
      Real glower[4][4], gupper[4][4];
      ComputeMetricAndInverse(x1v, x2v, x3v, coord.is_minkowski, coord.bh_spin,
                              glower, gupper);

      Real q = glower[1][1]*uu1*uu1 + 2.0*glower[1][2]*uu1*uu2 + 2.0*glower[1][3]*uu1*uu3
             + glower[2][2]*uu2*uu2 + 2.0*glower[2][3]*uu2*uu3
             + glower[3][3]*uu3*uu3;
      Real uu0 = sqrt(1.0 + q);
      Real u_tet_[4];
      u_tet_[0] = (norm_to_tet_(m,0,0,k,j,i)*uu0 + norm_to_tet_(m,0,1,k,j,i)*uu1 +
                   norm_to_tet_(m,0,2,k,j,i)*uu2 + norm_to_tet_(m,0,3,k,j,i)*uu3);
      u_tet_[1] = (norm_to_tet_(m,1,0,k,j,i)*uu0 + norm_to_tet_(m,1,1,k,j,i)*uu1 +
                   norm_to_tet_(m,1,2,k,j,i)*uu2 + norm_to_tet_(m,1,3,k,j,i)*uu3);
      u_tet_[2] = (norm_to_tet_(m,2,0,k,j,i)*uu0 + norm_to_tet_(m,2,1,k,j,i)*uu1 +
                   norm_to_tet_(m,2,2,k,j,i)*uu2 + norm_to_tet_(m,2,3,k,j,i)*uu3);
      u_tet_[3] = (norm_to_tet_(m,3,0,k,j,i)*uu0 + norm_to_tet_(m,3,1,k,j,i)*uu1 +
                   norm_to_tet_(m,3,2,k,j,i)*uu2 + norm_to_tet_(m,3,3,k,j,i)*uu3);

      for (int n=0; n<nangles_; ++n) {
        Real un_t = (u_tet_[1]*nh_c_.d_view(n,1) + u_tet_[2]*nh_c_.d_view(n,2) +
                     u_tet_[3]*nh_c_.d_view(n,3));
        Real n0_f = u_tet_[0]*nh_c_.d_view(n,0) - un_t;
        Real n0 = tet_c_(m,0,0,k,j,i); Real n_0 = 0.0;
        for (int d=0; d<4; ++d) { n_0 += tetcov_c_(m,d,0,k,j,i)*nh_c_.d_view(n,d); }
        i0_(m,n,k,j,i) = n0*n_0*(urad/(4.0*M_PI))/SQR(SQR(n0_f));
      }
    }
  });

  pmbp->phydro->peos->PrimToCons(w0_, u0_, is, ie, js, je, ks, ke);
  return;
}

namespace {

KOKKOS_INLINE_FUNCTION
static void GetBoyerLindquistCoordinates(struct xrb_pgen pgen,
                                         Real x1, Real x2, Real x3,
                                         Real *pr, Real *ptheta, Real *pphi) {
  Real rad = sqrt(SQR(x1) + SQR(x2) + SQR(x3));
  Real r = fmax((sqrt( SQR(rad) - SQR(pgen.spin) + sqrt(SQR(SQR(rad)-SQR(pgen.spin))
                      + 4.0*SQR(pgen.spin)*SQR(x3)) ) / sqrt(2.0)), 1.0);
  *pr = r;
  *ptheta = (fabs(x3/r) < 1.0) ? acos(x3/r) : acos(copysign(1.0, x3));
  *pphi = atan2(r*x2-pgen.spin*x1, pgen.spin*x2+r*x1) -
          pgen.spin*r/(SQR(r)-2.0*r+SQR(pgen.spin));
}

KOKKOS_INLINE_FUNCTION
Real CalculatePotential(struct xrb_pgen pgen, Real x1, Real x2, Real x3) {
  Real x = x1 - pgen.a_sep;
  Real y = x2;
  Real z = x3;
  Real d_accretor = sqrt(SQR(x1) + SQR(x2) + SQR(x3)); // distance to accretor
  Real d_donor = sqrt(SQR(x) + SQR(y) + SQR(z)); // distance to donor
  Real m_tot = pgen.m_accretor + pgen.m_donor;
  Real omega_sq = m_tot / (pgen.a_sep*pgen.a_sep*pgen.a_sep);
  Real x_com = pgen.m_donor * pgen.a_sep / m_tot;
  Real potential = -pgen.m_donor/d_donor - pgen.m_accretor/d_accretor;
  potential -= 0.5*omega_sq*(SQR(x1 - x_com) + SQR(x2));
  return potential;
}

KOKKOS_INLINE_FUNCTION
static void CalculateRocheAcceleration(struct xrb_pgen pgen,
                                     Real x1, Real x2, Real x3,
                                     Real vx, Real vy, Real vz,
                                     Real *ax, Real *ay, Real *az) {
  Real x_d = x1 - pgen.a_sep;
  Real y_d = x2;
  Real z_d = x3;
  Real d_d_sq = SQR(x_d) + SQR(y_d) + SQR(z_d);
  Real d_d = sqrt(d_d_sq); // distance to donor

  Real m_tot = pgen.m_accretor + pgen.m_donor;
  Real omega_sq = m_tot / (pgen.a_sep*pgen.a_sep*pgen.a_sep);
  Real omega = sqrt(omega_sq);
  Real x_com = pgen.m_donor * pgen.a_sep / m_tot;

  Real ax_val = 0.0;
  Real ay_val = 0.0;
  Real az_val = 0.0;

  // Donor point-mass gravity (skip inside the stellar surface).
  if (d_d > pgen.r_donor) {
    Real inv_d_d3 = pgen.m_donor / (d_d*d_d_sq);
    ax_val -= inv_d_d3 * x_d;
    ay_val -= inv_d_d3 * y_d;
    az_val -= inv_d_d3 * z_d;
  }

  // Accretor gravity in Newtonian mode (included in metric via CoordSrcTerms in GR).
  if (!pgen.is_gr) {
    Real d_a_sq = SQR(x1) + SQR(x2) + SQR(x3);
    Real d_a = sqrt(d_a_sq);
    Real d_a_eff = fmax(d_a, pgen.r_soft);
    Real inv_d_a3 = pgen.m_accretor / (d_a_eff*d_a_eff*d_a_eff);
    ax_val -= inv_d_a3 * x1;
    ay_val -= inv_d_a3 * x2;
    az_val -= inv_d_a3 * x3;
  }

  // Centrifugal acceleration in the corotating frame.
  ax_val += omega_sq * (x1 - x_com);
  ay_val += omega_sq * x2;

  // Coriolis acceleration: -2 Omega x v with Omega = (0, 0, omega).
  ax_val += 2.0*omega * vy;
  ay_val -= 2.0*omega * vx;

  *ax = ax_val;
  *ay = ay_val;
  *az = az_val;
}

} // namespace

void XRBSourceTerms(Mesh *pm, const Real bdt) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;
  auto &w0_ = pmbp->phydro->w0;
  auto &u0_ = pmbp->phydro->u0;
  const bool is_ideal = pmbp->phydro->peos->eos_data.is_ideal;
  auto pgen = xrb;
  int nmb1 = pmbp->nmb_thispack - 1;

  par_for("xrb_srcterm", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    Real rho = w0_(m,IDN,k,j,i);
    Real vx = w0_(m,IVX,k,j,i);
    Real vy = w0_(m,IVY,k,j,i);
    Real vz = w0_(m,IVZ,k,j,i);

    Real ax, ay, az;
    CalculateRocheAcceleration(pgen, x1v, x2v, x3v, vx, vy, vz, &ax, &ay, &az);

    Real src1 = bdt*rho*ax;
    Real src2 = bdt*rho*ay;
    Real src3 = bdt*rho*az;
    u0_(m,IM1,k,j,i) += src1;
    u0_(m,IM2,k,j,i) += src2;
    u0_(m,IM3,k,j,i) += src3;
    if (is_ideal) {
      Real work = src1*vx + src2*vy + src3*vz;
      if (pgen.is_gr) {
        u0_(m,IEN,k,j,i) -= work;
      } else {
        u0_(m,IEN,k,j,i) += work;
      }
    }
  });
}

void NoInflowXRB(Mesh *pm) {
  auto &indcs = pm->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &is = indcs.is;  int &ie  = indcs.ie;
  int &js = indcs.js;  int &je  = indcs.je;
  int &ks = indcs.ks;  int &ke  = indcs.ke;
  auto &mb_bcs = pm->pmb_pack->pmb->mb_bcs;

  auto &u0_ = pm->pmb_pack->phydro->u0;
  auto &w0_ = pm->pmb_pack->phydro->w0;
  int nmb = pm->pmb_pack->nmb_thispack;
  int nvar = u0_.extent_int(1);

  const bool is_gr = pm->pmb_pack->pcoord->is_general_relativistic;
  const bool is_radiation_enabled = (pm->pmb_pack->prad != nullptr);
  DvceArray5D<Real> i0_; int nang1;
  if (is_gr && is_radiation_enabled) {
    i0_ = pm->pmb_pack->prad->i0;
    nang1 = pm->pmb_pack->prad->prgeo->nangles - 1;
  }

  pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,is-ng,is,0,(n2-1),0,(n3-1));
  pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,ie,ie+ng,0,(n2-1),0,(n3-1));

  par_for("noinflow_hydro_x1", DevExeSpace(),0,(nmb-1),0,(nvar-1),0,(n3-1),0,(n2-1),
  KOKKOS_LAMBDA(int m, int n, int k, int j) {
    if (mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::user) {
      for (int i=0; i<ng; ++i) {
        if (n==(IVX)) {
          w0_(m,n,k,j,is-i-1) = fmin(0.0,w0_(m,n,k,j,is));
        } else {
          w0_(m,n,k,j,is-i-1) = w0_(m,n,k,j,is);
        }
      }
    }
    if (mb_bcs.d_view(m,BoundaryFace::outer_x1) == BoundaryFlag::user) {
      for (int i=0; i<ng; ++i) {
        if (n==(IVX)) {
          w0_(m,n,k,j,ie+i+1) = fmax(0.0,w0_(m,n,k,j,ie));
        } else {
          w0_(m,n,k,j,ie+i+1) = w0_(m,n,k,j,ie);
        }
      }
    }
  });
  if (is_gr && is_radiation_enabled) {
    par_for("noinflow_rad_x1", DevExeSpace(),0,(nmb-1),0,nang1,0,(n3-1),0,(n2-1),
    KOKKOS_LAMBDA(int m, int n, int k, int j) {
      if (mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::user) {
        for (int i=0; i<ng; ++i) {
          i0_(m,n,k,j,is-i-1) = i0_(m,n,k,j,is);
        }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x1) == BoundaryFlag::user) {
        for (int i=0; i<ng; ++i) {
          i0_(m,n,k,j,ie+i+1) = i0_(m,n,k,j,ie);
        }
      }
    });
  }
  pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,is-ng,is-1,0,(n2-1),0,(n3-1));
  pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,ie+1,ie+ng,0,(n2-1),0,(n3-1));

  pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,0,(n1-1),js-ng,js,0,(n3-1));
  pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,0,(n1-1),je,je+ng,0,(n3-1));

  par_for("noinflow_hydro_x2", DevExeSpace(),0,(nmb-1),0,(nvar-1),0,(n3-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int n, int k, int i) {
    if (mb_bcs.d_view(m,BoundaryFace::inner_x2) == BoundaryFlag::user) {
      for (int j=0; j<ng; ++j) {
        if (n==(IVY)) {
          w0_(m,n,k,js-j-1,i) = fmin(0.0,w0_(m,n,k,js,i));
        } else {
          w0_(m,n,k,js-j-1,i) = w0_(m,n,k,js,i);
        }
      }
    }
    if (mb_bcs.d_view(m,BoundaryFace::outer_x2) == BoundaryFlag::user) {
      for (int j=0; j<ng; ++j) {
        if (n==(IVY)) {
          w0_(m,n,k,je+j+1,i) = fmax(0.0,w0_(m,n,k,je,i));
        } else {
          w0_(m,n,k,je+j+1,i) = w0_(m,n,k,je,i);
        }
      }
    }
  });
  if (is_gr && is_radiation_enabled) {
    par_for("noinflow_rad_x2", DevExeSpace(),0,(nmb-1),0,nang1,0,(n3-1),0,(n1-1),
    KOKKOS_LAMBDA(int m, int n, int k, int i) {
      if (mb_bcs.d_view(m,BoundaryFace::inner_x2) == BoundaryFlag::user) {
        for (int j=0; j<ng; ++j) {
          i0_(m,n,k,js-j-1,i) = i0_(m,n,k,js,i);
        }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x2) == BoundaryFlag::user) {
        for (int j=0; j<ng; ++j) {
          i0_(m,n,k,je+j+1,i) = i0_(m,n,k,je,i);
        }
      }
    });
  }
  pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,0,(n1-1),js-ng,js-1,0,(n3-1));
  pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,0,(n1-1),je+1,je+ng,0,(n3-1));

  pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,0,(n1-1),0,(n2-1),ks-ng,ks);
  pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,0,(n1-1),0,(n2-1),ke,ke+ng);

  par_for("noinflow_hydro_x3", DevExeSpace(),0,(nmb-1),0,(nvar-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int n, int j, int i) {
    if (mb_bcs.d_view(m,BoundaryFace::inner_x3) == BoundaryFlag::user) {
      for (int k=0; k<ng; ++k) {
        if (n==(IVZ)) {
          w0_(m,n,ks-k-1,j,i) = fmin(0.0,w0_(m,n,ks,j,i));
        } else {
          w0_(m,n,ks-k-1,j,i) = w0_(m,n,ks,j,i);
        }
      }
    }
    if (mb_bcs.d_view(m,BoundaryFace::outer_x3) == BoundaryFlag::user) {
      for (int k=0; k<ng; ++k) {
        if (n==(IVZ)) {
          w0_(m,n,ke+k+1,j,i) = fmax(0.0,w0_(m,n,ke,j,i));
        } else {
          w0_(m,n,ke+k+1,j,i) = w0_(m,n,ke,j,i);
        }
      }
    }
  });
  if (is_gr && is_radiation_enabled) {
    par_for("noinflow_rad_x3", DevExeSpace(),0,(nmb-1),0,nang1,0,(n2-1),0,(n1-1),
    KOKKOS_LAMBDA(int m, int n, int j, int i) {
      if (mb_bcs.d_view(m,BoundaryFace::inner_x3) == BoundaryFlag::user) {
        for (int k=0; k<ng; ++k) {
          i0_(m,n,ks-k-1,j,i) = i0_(m,n,ks,j,i);
        }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x3) == BoundaryFlag::user) {
        for (int k=0; k<ng; ++k) {
          i0_(m,n,ke+k+1,j,i) = i0_(m,n,ke,j,i);
        }
      }
    });
  }
  pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,0,(n1-1),0,(n2-1),ks-ng,ks-1);
  pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,0,(n1-1),0,(n2-1),ke+1,ke+ng);
}

void AccretorFluxes(HistoryData *pdata, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  const bool is_gr = pmbp->pcoord->is_general_relativistic;

  int nvars = pmbp->phydro->nhydro + pmbp->phydro->nscalars;
  Real gamma = pmbp->phydro->peos->eos_data.gamma;
  auto &w0_ = pmbp->phydro->w0;

  auto &grids = pm->pgen->spherical_grids;
  int nradii = grids.size();
  const int nflux = 3;

  pdata->nhist = nradii*nflux;
  if (pdata->nhist > NHISTORY_VARIABLES) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "User history function specified pdata->nhist larger than"
              << " NHISTORY_VARIABLES" << std::endl;
    exit(EXIT_FAILURE);
  }
  for (int g=0; g<nradii; ++g) {
    std::stringstream stream;
    stream << std::fixed << std::setprecision(1) << grids[g]->radius;
    std::string rad_str = stream.str();
    pdata->label[nflux*g+0] = "mdot_" + rad_str;
    pdata->label[nflux*g+1] = "edot_" + rad_str;
    pdata->label[nflux*g+2] = "ldot_" + rad_str;
  }

  bool &flat = pmbp->pcoord->coord_data.is_minkowski;
  Real &spin = pmbp->pcoord->coord_data.bh_spin;

  for (int g=0; g<nradii; ++g) {
    pdata->hdata[nflux*g+0] = 0.0;
    pdata->hdata[nflux*g+1] = 0.0;
    pdata->hdata[nflux*g+2] = 0.0;

    grids[g]->InterpolateToSphere(nvars, w0_);

    for (int n=0; n<grids[g]->nangles; ++n) {
      Real r = grids[g]->radius;
      Real theta = grids[g]->polar_pos.h_view(n,0);
      Real phi = grids[g]->polar_pos.h_view(n,1);
      Real x1 = grids[g]->interp_coord.h_view(n,0);
      Real x2 = grids[g]->interp_coord.h_view(n,1);
      Real x3 = grids[g]->interp_coord.h_view(n,2);

      Real &int_dn = grids[g]->interp_vals.h_view(n,IDN);
      Real &int_vx = grids[g]->interp_vals.h_view(n,IVX);
      Real &int_vy = grids[g]->interp_vals.h_view(n,IVY);
      Real &int_vz = grids[g]->interp_vals.h_view(n,IVZ);
      Real int_ie = grids[g]->interp_vals.h_view(n,IEN);
      Real &domega = grids[g]->solid_angles.h_view(n);

      if (is_gr) {
        Real glower[4][4], gupper[4][4];
        ComputeMetricAndInverse(x1,x2,x3,flat,spin,glower,gupper);

        Real q = glower[1][1]*int_vx*int_vx + 2.0*glower[1][2]*int_vx*int_vy +
                 2.0*glower[1][3]*int_vx*int_vz + glower[2][2]*int_vy*int_vy +
                 2.0*glower[2][3]*int_vy*int_vz + glower[3][3]*int_vz*int_vz;
        Real alpha = sqrt(-1.0/gupper[0][0]);
        Real lor = sqrt(1.0 + q);
        Real u0 = lor/alpha;
        Real u1 = int_vx - alpha * lor * gupper[0][1];
        Real u2 = int_vy - alpha * lor * gupper[0][2];
        Real u3 = int_vz - alpha * lor * gupper[0][3];

        Real u_0 = glower[0][0]*u0 + glower[0][1]*u1 + glower[0][2]*u2 + glower[0][3]*u3;
        Real u_1 = glower[1][0]*u0 + glower[1][1]*u1 + glower[1][2]*u2 + glower[1][3]*u3;
        Real u_2 = glower[2][0]*u0 + glower[2][1]*u1 + glower[2][2]*u2 + glower[2][3]*u3;
        Real u_3 = glower[3][0]*u0 + glower[3][1]*u1 + glower[3][2]*u2 + glower[3][3]*u3;

        Real a2 = SQR(spin);
        Real rad2 = SQR(x1)+SQR(x2)+SQR(x3);
        Real r2 = SQR(r);
        Real sth = sin(theta);
        Real sph = sin(phi);
        Real cph = cos(phi);
        Real drdx = r*x1/(2.0*r2 - rad2 + a2);
        Real drdy = r*x2/(2.0*r2 - rad2 + a2);
        Real drdz = (r*x3 + a2*x3/r)/(2.0*r2-rad2+a2);
        Real ur  = drdx *u1 + drdy *u2 + drdz *u3;
        Real u_ph = (-r*sph-spin*cph)*sth*u_1 + (r*cph-spin*sph)*sth*u_2;
        Real sqrtmdet = (r2+SQR(spin*cos(theta)));

        pdata->hdata[nflux*g+0] += -1.0*int_dn*ur*sqrtmdet*domega;

        Real t1_0 = (int_dn + gamma*int_ie)*ur*u_0;
        pdata->hdata[nflux*g+1] += -1.0*t1_0*sqrtmdet*domega;

        Real t1_3 = (int_dn + gamma*int_ie)*ur*u_ph;
        pdata->hdata[nflux*g+2] += t1_3*sqrtmdet*domega;
      } else {
        Real sth = sin(theta);
        Real cph = cos(phi);
        Real sph = sin(phi);
        Real vr = int_vx*sth*cph + int_vy*sth*sph + int_vz*cos(theta);
        Real r2 = SQR(r);
        Real area = r2*sth;

        pdata->hdata[nflux*g+0] += -1.0*int_dn*vr*area*domega;

        Real edot_flux = int_dn*(0.5*(SQR(int_vx)+SQR(int_vy)+SQR(int_vz)) + gamma*int_ie)*vr;
        pdata->hdata[nflux*g+1] += -1.0*edot_flux*area*domega;

        Real ldot_flux = int_dn*vr*(x1*int_vy - x2*int_vx);
        pdata->hdata[nflux*g+2] += ldot_flux*area*domega;
      }
    }
  }

  for (int n=pdata->nhist; n<NHISTORY_VARIABLES; ++n) {
    pdata->hdata[n] = 0.0;
  }
}
