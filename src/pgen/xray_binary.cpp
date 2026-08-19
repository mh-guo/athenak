//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file xray_binary.cpp
//! \brief Problem generator for x-ray binary with donor on +x and accretor at origin.
//! Supports Newtonian/GR hydro or MHD (+ optional radiation in GR).
//! MHD magnetic field options (independent; both off => B=0):
//!   mean_field = poloidal : donor-envelope poloidal field from vector potential
//!     A_phi_phys = R^{r_pow} [max(rho-rho_cut,0)]^{rho_pow}  (R = cyl. radius about donor z)
//!     r_pow>=1 keeps B finite on the donor polar axis; no extra theta factor (rho is Roche)
//!     default r_pow = gamma/(2*(gamma-1)) and rho_pow=1 => mid-envelope |B|~sqrt(p)
//!       on this point-mass polytrope (1.75 when gamma=1.4)
//!     potential_beta_min = min(p/p_mag) in the live envelope
//!       (r_mask < r < r_donor and rho > rho_cut), not a volume mean over the 1/r core
//!   <turb_seed*> blocks   : one-shot seed (see pgen/turb_seed.hpp)
//! fix_donor mask resets fluid each BC. With MHD, E=0 on mask-cell edges in EFieldSrc
//!   (last step before SendE) so interior B is CT-frozen and divB stays 0.

#include <stdio.h>
#include <math.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
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
#include "eos/ideal_c2p_mhd.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "geodesic-grid/spherical_grid.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "radiation/radiation.hpp"
#include "pgen/turb_seed.hpp"

namespace {
KOKKOS_INLINE_FUNCTION
static void GetBoyerLindquistCoordinates(struct xrb_pgen pgen,
                                         Real x1, Real x2, Real x3,
                                         Real *pr, Real *ptheta, Real *pphi);

KOKKOS_INLINE_FUNCTION
Real CalculateRochePotential(struct xrb_pgen pgen, Real x1, Real x2, Real x3);

KOKKOS_INLINE_FUNCTION
static void CalculateRocheAcceleration(struct xrb_pgen pgen,
                                     Real x1, Real x2, Real x3,
                                     Real vx, Real vy, Real vz,
                                     Real *ax, Real *ay, Real *az);

KOKKOS_INLINE_FUNCTION
static void CalculateBackgroundState(struct xrb_pgen pgen,
                                   Real x1, Real x2, Real x3,
                                   Real dx1, Real dx2, Real dx3,
                                   Real *prho_bg, Real *ppgas_bg);

KOKKOS_INLINE_FUNCTION
static void CalculateDonorState(struct xrb_pgen pgen,
                                Real x1, Real x2, Real x3,
                                Real dx1, Real dx2, Real dx3,
                                Real *prho, Real *ppgas);

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
  Real r_donor_mask;                          // radius for fixed donor interior BC
  Real r_donor_flux;                          // history sphere radius about donor
  bool fix_donor;                             // enforce donor mask each BC call
  bool donor_hist;                            // enroll donor-centered history diagnostics
  Real mass_ratio;                            // mass ratio: m_accretor / m_total
  Real rho_min, rho_pow, pgas_min, pgas_pow;  // background atmosphere parameters
  bool heating;                               // Scherbak-style envelope heating (Eq. 7)
  Real t_kh;                                  // KH expansion timescale (Eq. 12)
  Real L_heat;                                // derived from t_kh via Eq. (12)
  Real heat_f_ext;                            // M_ext / M_gas input (Eq. 11)
  Real heat_M_gas;                            // integrated donor gas mass
  Real heat_m_ext;                            // heat_f_ext * heat_M_gas (Eq. 11)
  Real heat_R_h;                              // radius at Phi_h (Eq. 12)
  Real heat_phi_h;                            // derived Roche potential at heated shell
  Real heat_phi_surface;                      // Roche potential at donor surface
  Real heat_delta_phi;                        // Gaussian width in potential (Eq. 9)
  Real heat_t0;                               // ramp center time
  Real heat_dt_ramp;                          // ramp width time
  Real heating_norm;                          // L_heat / int(unnormalized profile dV)
  // MHD mean field (donor poloidal); unused when mean_field=none
  bool mean_field_poloidal;
  Real potential_beta_min;
  Real potential_rho_cut;
  Real potential_rho_pow;
  Real potential_r_pow;
};

xrb_pgen xrb;

//----------------------------------------------------------------------------------------
//! Donor-centered poloidal vector potential A = A_phi_phys * e_phi (about z through donor).
//! A_phi_phys = R^{r_pow} [max(rho - rho_cut, 0)]^{rho_pow} from the Roche polytrope.
//! Cartesian: A = f(rho) * R^{r_pow-1} * (-y, x, 0). r_pow>=1 => B finite on the axis.
//! Default r_pow = gamma/(2*(gamma-1)) with rho_pow=1 matches |B|~sqrt(p) in the
//! mid envelope (Delta Phi ~ 1/r).

KOKKOS_INLINE_FUNCTION
static Real DonorAphi(struct xrb_pgen pgen, Real x1, Real x2, Real x3) {
  Real rho, pgas;
  CalculateDonorState(pgen, x1, x2, x3, 0.0, 0.0, 0.0, &rho, &pgas);
  Real excess = rho - pgen.potential_rho_cut;
  if (excess <= 0.0) return 0.0;
  return pow(excess, pgen.potential_rho_pow);
}

KOKKOS_INLINE_FUNCTION
static void DonorVectorPotential(struct xrb_pgen pgen, Real x1, Real x2, Real x3,
                                 Real *pa1, Real *pa2, Real *pa3) {
  Real xd = x1 - pgen.a_sep;
  Real rcyl = sqrt(SQR(xd) + SQR(x2));
  Real aphi = DonorAphi(pgen, x1, x2, x3);
  // A = f(rho) R^{r_pow} e_phi  =>  (Ax,Ay) = f R^{r_pow-1} (-y, x)
  if (aphi == 0.0 || rcyl <= 1.0e-30) {
    *pa1 = 0.0;
    *pa2 = 0.0;
    *pa3 = 0.0;
    return;
  }
  Real rfac = pow(rcyl, pgen.potential_r_pow - 1.0);
  *pa1 = -aphi * rfac * x2;
  *pa2 =  aphi * rfac * xd;
  *pa3 = 0.0;
}

KOKKOS_INLINE_FUNCTION
static Real A1(struct xrb_pgen pgen, Real x1, Real x2, Real x3) {
  Real a1, a2, a3;
  DonorVectorPotential(pgen, x1, x2, x3, &a1, &a2, &a3);
  return a1;
}

KOKKOS_INLINE_FUNCTION
static Real A2(struct xrb_pgen pgen, Real x1, Real x2, Real x3) {
  Real a1, a2, a3;
  DonorVectorPotential(pgen, x1, x2, x3, &a1, &a2, &a3);
  return a2;
}

KOKKOS_INLINE_FUNCTION
static Real A3(struct xrb_pgen pgen, Real x1, Real x2, Real x3) {
  Real a1, a2, a3;
  DonorVectorPotential(pgen, x1, x2, x3, &a1, &a2, &a3);
  return a3;
}

KOKKOS_INLINE_FUNCTION
static bool CellInDonorMask(struct xrb_pgen pgen, Real x1, Real x2, Real x3) {
  Real xd = x1 - pgen.a_sep;
  return (SQR(xd) + SQR(x2) + SQR(x3)) <= SQR(pgen.r_donor_mask);
}

KOKKOS_INLINE_FUNCTION
Real HeatProfile(struct xrb_pgen pgen, Real phi,
                         Real x1, Real x2, Real x3) {
  if (phi > pgen.heat_phi_surface) return 0.0;
  Real x_d = x1 - pgen.a_sep;
  Real d_donor = sqrt(SQR(x_d) + SQR(x2) + SQR(x3));
  if (d_donor >= pgen.r_donor) return 0.0;
  if (pgen.fix_donor && d_donor <= pgen.r_donor_mask) return 0.0;
  Real dphi = pgen.heat_delta_phi;
  if (dphi <= 0.0) return 0.0;
  return exp(-SQR(phi - pgen.heat_phi_h) / (2.0*SQR(dphi)));
}

Real FindHeatRadius(struct xrb_pgen pgen) {
  Real d_lo = 0.0;
  Real d_hi = pgen.r_donor;
  for (int n = 0; n < 64; ++n) {
    Real d_mid = 0.5*(d_lo + d_hi);
    Real phi = CalculateRochePotential(pgen, pgen.a_sep - d_mid, 0.0, 0.0);
    if (phi < pgen.heat_phi_h) {
      d_lo = d_mid;
    } else {
      d_hi = d_mid;
    }
  }
  return 0.5*(d_lo + d_hi);
}

void ComputeHeatEnvelopeMass(Mesh *pm, Real phi_h_thresh, Real *pM_ext) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  auto &size = pmbp->pmb->mb_size;
  const int nmkji = pmbp->nmb_thispack * nx3 * nx2 * nx1;
  const int nkji = nx3 * nx2 * nx1;
  const int nji = nx2 * nx1;
  auto pgen = xrb;

  Real local_mass = 0.0;
  Kokkos::parallel_reduce("xrb_heat_mext",
                          Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int idx, Real &lmass) {
    int m = idx / nkji;
    int k = (idx - m*nkji) / nji;
    int j = (idx - m*nkji - k*nji) / nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;

    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
    Real &dx1 = size.d_view(m).dx1;
    Real &dx2 = size.d_view(m).dx2;
    Real &dx3 = size.d_view(m).dx3;

    Real x_d = x1v - pgen.a_sep;
    Real d_donor = sqrt(SQR(x_d) + SQR(x2v) + SQR(x3v));
    if (d_donor >= pgen.r_donor) return;

    Real phi = CalculateRochePotential(pgen, x1v, x2v, x3v);
    if (phi > phi_h_thresh) {
      Real rho, pgas;
      CalculateDonorState(pgen, x1v, x2v, x3v, dx1, dx2, dx3, &rho, &pgas);
      lmass += rho * dx1 * dx2 * dx3;
    }
  }, local_mass);

#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &local_mass, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
  *pM_ext = local_mass;
}

void ComputeDonorGasMass(Mesh *pm, Real *pM_gas) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  auto &size = pmbp->pmb->mb_size;
  const int nmkji = pmbp->nmb_thispack * nx3 * nx2 * nx1;
  const int nkji = nx3 * nx2 * nx1;
  const int nji = nx2 * nx1;
  auto pgen = xrb;

  Real local_mass = 0.0;
  Kokkos::parallel_reduce("xrb_heat_mgas",
                          Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int idx, Real &lmass) {
    int m = idx / nkji;
    int k = (idx - m*nkji) / nji;
    int j = (idx - m*nkji - k*nji) / nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;

    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
    Real &dx1 = size.d_view(m).dx1;
    Real &dx2 = size.d_view(m).dx2;
    Real &dx3 = size.d_view(m).dx3;

    Real x_d = x1v - pgen.a_sep;
    Real d_donor = sqrt(SQR(x_d) + SQR(x2v) + SQR(x3v));
    if (d_donor >= pgen.r_donor) return;

    Real rho, pgas;
    CalculateDonorState(pgen, x1v, x2v, x3v, dx1, dx2, dx3, &rho, &pgas);
    lmass += rho * dx1 * dx2 * dx3;
  }, local_mass);

#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &local_mass, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
  *pM_gas = local_mass;
}

Real FindPhiHFromMext(Mesh *pm, Real target_m_ext) {
  Real phi_hi = xrb.heat_phi_surface;
  Real phi_lo =
    CalculateRochePotential(xrb, xrb.a_sep - 0.01*xrb.r_donor, 0.0, 0.0);

  Real m_lo = 0.0;
  ComputeHeatEnvelopeMass(pm, phi_lo, &m_lo);

  if (target_m_ext <= 0.0 || target_m_ext > m_lo) {
    if (global_variable::my_rank == 0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "heat_m_ext out of range: target=" << target_m_ext
                << " max envelope mass=" << m_lo << std::endl;
    }
    exit(EXIT_FAILURE);
  }

  for (int n = 0; n < 64; ++n) {
    Real phi_mid = 0.5*(phi_lo + phi_hi);
    Real m_mid = 0.0;
    ComputeHeatEnvelopeMass(pm, phi_mid, &m_mid);
    if (m_mid > target_m_ext) {
      phi_lo = phi_mid;
    } else {
      phi_hi = phi_mid;
    }
  }
  return 0.5*(phi_lo + phi_hi);
}

void SetupHeating(Mesh *pm) {
  if (!xrb.heating) return;

  xrb.heat_delta_phi = (xrb.heat_phi_surface - xrb.heat_phi_h) * (xrb.gamma_adi - 1.0);
  if (xrb.heat_delta_phi <= 0.0) {
    if (global_variable::my_rank == 0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "heat_phi_h must be deeper than heat_phi_surface (more negative Phi)"
                << std::endl;
    }
    exit(EXIT_FAILURE);
  }

  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  auto &size = pmbp->pmb->mb_size;
  const int nmkji = pmbp->nmb_thispack * nx3 * nx2 * nx1;
  const int nkji = nx3 * nx2 * nx1;
  const int nji = nx2 * nx1;
  auto pgen = xrb;

  Real local_sum = 0.0;
  Kokkos::parallel_reduce("xrb_heat_norm",
                          Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int idx, Real &lsum) {
    int m = idx / nkji;
    int k = (idx - m*nkji) / nji;
    int j = (idx - m*nkji - k*nji) / nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;

    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

    Real phi = CalculateRochePotential(pgen, x1v, x2v, x3v);
    Real profile = HeatProfile(pgen, phi, x1v, x2v, x3v);
    if (profile > 0.0) {
      lsum += profile * size.d_view(m).dx1 * size.d_view(m).dx2 * size.d_view(m).dx3;
    }
  }, local_sum);

#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &local_sum, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif

  if (local_sum <= 0.0) {
    if (global_variable::my_rank == 0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "Heating normalization integral is zero." << std::endl
                << "Check heat_phi_h vs fix_donor/r_donor_mask: heating is disabled"
                << " inside r_donor_mask." << std::endl;
    }
    exit(EXIT_FAILURE);
  }
  xrb.heating_norm = xrb.L_heat / local_sum;
}

void InitializeHeating(ParameterInput *pin, Mesh *pm) {
  xrb.heating = pin->GetOrAddBoolean("problem", "heating", false);
  if (!xrb.heating) return;

  Real m_tot = xrb.m_accretor + xrb.m_donor;
  Real omega = sqrt(m_tot / (xrb.a_sep*xrb.a_sep*xrb.a_sep));
  Real phi_surface =
    CalculateRochePotential(xrb, xrb.a_sep - xrb.r_donor, 0.0, 0.0);
  xrb.heat_phi_surface = pin->GetOrAddReal("problem", "heat_phi_surface", phi_surface);
  xrb.heat_f_ext = pin->GetOrAddReal("problem", "heat_f_ext", 1.0e-3);
  if (xrb.heat_f_ext <= 0.0) {
    if (global_variable::my_rank == 0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "heat_f_ext must be positive" << std::endl;
    }
    exit(EXIT_FAILURE);
  }

  ComputeDonorGasMass(pm, &xrb.heat_M_gas);
  if (xrb.heat_M_gas <= 0.0) {
    if (global_variable::my_rank == 0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "Donor gas mass is zero; check k_adi and r_donor" << std::endl;
    }
    exit(EXIT_FAILURE);
  }
  xrb.heat_m_ext = xrb.heat_f_ext * xrb.heat_M_gas;
  xrb.heat_phi_h = FindPhiHFromMext(pm, xrb.heat_m_ext);
  xrb.heat_t0 = pin->GetOrAddReal("problem", "heat_t0", 15.0/omega);
  xrb.heat_dt_ramp = pin->GetOrAddReal("problem", "heat_dt_ramp", 5.0/omega);

  xrb.heat_R_h = FindHeatRadius(xrb);
  if (xrb.heat_R_h <= 0.0) {
    if (global_variable::my_rank == 0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "Invalid R_h for t_kh heating setup" << std::endl;
    }
    exit(EXIT_FAILURE);
  }

  xrb.t_kh = pin->GetOrAddReal("problem", "t_kh", 525.0/omega);
  if (xrb.t_kh <= 0.0) {
    if (global_variable::my_rank == 0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "t_kh must be positive" << std::endl;
    }
    exit(EXIT_FAILURE);
  }
  // Scherbak Eq. (12): t_kh = M_1 M_ext / (R_h L_heat)
  xrb.L_heat = xrb.m_donor * xrb.heat_m_ext / (xrb.heat_R_h * xrb.t_kh);
  SetupHeating(pm);
}

} // namespace

void NoInflowXRB(Mesh *pm);
void ApplyDonorMask(Mesh *pm);
void ZeroDonorMaskEMF(Mesh *pm);
void UserBcsXRB(Mesh *pm);
void XRBSourceTerms(Mesh *pm, const Real bdt);
void AccretorFluxes(HistoryData *pdata, Mesh *pm);
void FillXRBHeatDerived(Mesh *pm, DvceArray5D<Real> dv, int i_dv);

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if ((pmbp->phydro == nullptr && pmbp->pmhd == nullptr) ||
      (pmbp->phydro != nullptr && pmbp->pmhd != nullptr)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "xray_binary requires exactly one of <hydro> or <mhd>" << std::endl;
    exit(EXIT_FAILURE);
  }
  const bool use_mhd = (pmbp->pmhd != nullptr);

  const bool is_gr = pmbp->pcoord->is_general_relativistic;
  const bool is_radiation_enabled = (pmbp->prad != nullptr);
  if (!is_gr && is_radiation_enabled) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Newtonian xray_binary does not support <radiation>" << std::endl;
    exit(EXIT_FAILURE);
  }

  user_bcs_func = UserBcsXRB;
  user_srcs_func = XRBSourceTerms;
  user_hist_func = AccretorFluxes;
  user_derived_func = FillXRBHeatDerived;

  auto &indcs = pmy_mesh_->mb_indcs;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int ie = indcs.ie, je = indcs.je, ke = indcs.ke;
  auto &coord = pmbp->pcoord->coord_data;
  int nmb = pmbp->nmb_thispack;

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
  grids.push_back(std::make_unique<SphericalGrid>(pmbp, 20, rflux_inner));
  // Near-horizon diagnostic spheres (gr_torus heritage); useless for Newtonian
  // XRB where r_soft ~ 1e4 and a_sep ~ 1e6.
  if (is_gr) {
    grids.push_back(std::make_unique<SphericalGrid>(pmbp, 20, 12.0));
    grids.push_back(std::make_unique<SphericalGrid>(pmbp, 20, 24.0));
  }

  if (use_mhd) {
    xrb.gamma_adi = pmbp->pmhd->peos->eos_data.gamma;
  } else {
    xrb.gamma_adi = pmbp->phydro->peos->eos_data.gamma;
  }
  xrb.k_adi = pin->GetReal("problem", "k_adi");
  xrb.a_sep = pin->GetReal("problem", "a_sep");
  xrb.m_donor = pin->GetReal("problem", "m_donor");
  xrb.m_accretor = pin->GetOrAddReal("problem", "m_accretor", 1.0);
  xrb.r_donor = pin->GetReal("problem", "r_donor");
  xrb.fix_donor = pin->GetOrAddBoolean("problem", "fix_donor", false);
  xrb.r_donor_mask = pin->GetOrAddReal("problem", "r_donor_mask", xrb.r_donor);
  if (use_mhd && xrb.fix_donor) {
    user_efield_func = ZeroDonorMaskEMF;
  }
  xrb.r_donor_flux = pin->GetOrAddReal("problem", "flux_radius_donor", 1.01*xrb.r_donor);
  xrb.donor_hist = false;
  xrb.mass_ratio = xrb.m_accretor / (xrb.m_donor + xrb.m_accretor);
  xrb.rho_min = pin->GetReal("problem", "rho_min");
  xrb.rho_pow = pin->GetReal("problem", "rho_pow");
  xrb.pgas_min = pin->GetReal("problem", "pgas_min");
  xrb.pgas_pow = pin->GetReal("problem", "pgas_pow");

  // Donor-centered flux sphere (Newtonian only): must lie outside fix_donor mask
  if (!is_gr) {
    if (xrb.r_donor_flux <= xrb.r_donor_mask) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "flux_radius_donor (" << xrb.r_donor_flux
                << ") must be > r_donor_mask (" << xrb.r_donor_mask << ")" << std::endl;
      exit(EXIT_FAILURE);
    }
    grids.push_back(std::make_unique<SphericalGrid>(
        pmbp, 20, xrb.r_donor_flux, -1, xrb.a_sep, 0.0, 0.0));
    xrb.donor_hist = true;
  }
  // MHD mean-field options
  xrb.mean_field_poloidal = false;
  xrb.potential_beta_min = 100.0;
  xrb.potential_rho_cut = 0.0;
  xrb.potential_rho_pow = 1.0;
  xrb.potential_r_pow = 1.0;
  if (use_mhd) {
    std::string mean_field = pin->GetOrAddString("problem", "mean_field", "none");
    if (mean_field == "poloidal") {
      xrb.mean_field_poloidal = true;
      xrb.potential_beta_min = pin->GetOrAddReal("problem", "potential_beta_min", 100.0);
      xrb.potential_rho_cut = pin->GetOrAddReal("problem", "potential_rho_cut",
                                                2.0*xrb.rho_min);
      xrb.potential_rho_pow = pin->GetOrAddReal("problem", "potential_rho_pow", 1.0);
      // Mid-envelope |B|~sqrt(p) on a point-mass polytrope: r_pow = gamma/(2*(gamma-1))
      Real r_pow_def = 1.0;
      if (xrb.gamma_adi > 1.0) {
        r_pow_def = xrb.gamma_adi / (2.0*(xrb.gamma_adi - 1.0));
      }
      xrb.potential_r_pow = pin->GetOrAddReal("problem", "potential_r_pow", r_pow_def);
      if (xrb.potential_beta_min <= 0.0) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl
                  << "problem/potential_beta_min must be positive" << std::endl;
        exit(EXIT_FAILURE);
      }
    } else if (mean_field != "none") {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "problem/mean_field must be 'none' or 'poloidal'" << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  InitializeHeating(pin, pmy_mesh_);

  if (global_variable::my_rank == 0) {
    std::cout << "xrb.gamma_adi = " << xrb.gamma_adi << std::endl;
    std::cout << "xrb.k_adi = " << xrb.k_adi << std::endl;
    std::cout << "xrb.a_sep = " << xrb.a_sep << std::endl;
    std::cout << "xrb.m_donor = " << xrb.m_donor << std::endl;
    std::cout << "xrb.m_accretor = " << xrb.m_accretor << std::endl;
    std::cout << "xrb.r_donor = " << xrb.r_donor << std::endl;
    std::cout << "xrb.fix_donor = " << xrb.fix_donor << std::endl;
    std::cout << "xrb.r_donor_mask = " << xrb.r_donor_mask << std::endl;
    if (use_mhd && xrb.fix_donor) {
      std::cout << "xrb.fix_donor_emf = 1 (E=0 on mask edges; CT-frozen B)"
                << std::endl;
    }
    if (xrb.donor_hist) {
      std::cout << "xrb.r_donor_flux = " << xrb.r_donor_flux << std::endl;
    }
    std::cout << "xrb.mass_ratio = " << xrb.mass_ratio << std::endl;
    std::cout << "xrb.rho_min = " << xrb.rho_min << std::endl;
    std::cout << "xrb.rho_pow = " << xrb.rho_pow << std::endl;
    std::cout << "xrb.pgas_min = " << xrb.pgas_min << std::endl;
    std::cout << "xrb.pgas_pow = " << xrb.pgas_pow << std::endl;
    std::cout << "xrb.heating = " << xrb.heating << std::endl;
    if (xrb.heating) {
      std::cout << "xrb.t_kh = " << xrb.t_kh << std::endl;
      std::cout << "xrb.L_heat = " << xrb.L_heat << std::endl;
      std::cout << "xrb.heat_f_ext = " << xrb.heat_f_ext << std::endl;
      std::cout << "xrb.heat_M_gas = " << xrb.heat_M_gas << std::endl;
      std::cout << "xrb.heat_m_ext = " << xrb.heat_m_ext << std::endl;
      std::cout << "xrb.heat_R_h = " << xrb.heat_R_h << std::endl;
      std::cout << "xrb.heat_phi_h = " << xrb.heat_phi_h << std::endl;
      std::cout << "xrb.heat_phi_surface = " << xrb.heat_phi_surface << std::endl;
      std::cout << "xrb.heat_delta_phi = " << xrb.heat_delta_phi << std::endl;
      std::cout << "xrb.heat_t0 = " << xrb.heat_t0 << std::endl;
      std::cout << "xrb.heat_dt_ramp = " << xrb.heat_dt_ramp << std::endl;
      std::cout << "xrb.heating_norm = " << xrb.heating_norm << std::endl;
    }
    if (use_mhd) {
      std::cout << "xrb.mean_field_poloidal = " << xrb.mean_field_poloidal << std::endl;
      if (xrb.mean_field_poloidal) {
        std::cout << "xrb.potential_beta_min = " << xrb.potential_beta_min << std::endl;
        std::cout << "xrb.potential_rho_cut = " << xrb.potential_rho_cut << std::endl;
        std::cout << "xrb.potential_rho_pow = " << xrb.potential_rho_pow << std::endl;
        std::cout << "xrb.potential_r_pow = " << xrb.potential_r_pow << std::endl;
      }
    }
  }

  if (restart) return;

  DvceArray5D<Real> u0_, w0_;
  if (use_mhd) {
    u0_ = pmbp->pmhd->u0;
    w0_ = pmbp->pmhd->w0;
  } else {
    u0_ = pmbp->phydro->u0;
    w0_ = pmbp->phydro->w0;
  }

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

    Real rho, pgas;
    CalculateDonorState(pgen, x1v, x2v, x3v, dx1, dx2, dx3, &rho, &pgas);
    Real uu1 = 0.0;
    Real uu2 = 0.0;
    Real uu3 = 0.0;
    const Real urad = 0.0;

    w0_(m,IDN,k,j,i) = rho;
    w0_(m,IEN,k,j,i) = pgas / gm1;
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

  //--------------------------------------------------------------------------
  // Magnetic field (MHD only): start from B=0, then optional mean + turb_seed
  //--------------------------------------------------------------------------
  if (use_mhd) {
    auto &b0_z = pmbp->pmhd->b0;
    auto &bcc_z = pmbp->pmhd->bcc0;
    par_for("xrb_zero_b", DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      b0_z.x1f(m,k,j,i) = 0.0;
      b0_z.x2f(m,k,j,i) = 0.0;
      b0_z.x3f(m,k,j,i) = 0.0;
      if (i==ie) {b0_z.x1f(m,k,j,i+1) = 0.0;}
      if (j==je) {b0_z.x2f(m,k,j+1,i) = 0.0;}
      if (k==ke) {b0_z.x3f(m,k+1,j,i) = 0.0;}
      bcc_z(m,IBX,k,j,i) = 0.0;
      bcc_z(m,IBY,k,j,i) = 0.0;
      bcc_z(m,IBZ,k,j,i) = 0.0;
    });
  }

  if (use_mhd && xrb.mean_field_poloidal) {
    int ncells1 = indcs.nx1 + 2*(indcs.ng);
    int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
    int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
    DvceArray4D<Real> a1, a2, a3;
    Kokkos::realloc(a1, nmb, ncells3, ncells2, ncells1);
    Kokkos::realloc(a2, nmb, ncells3, ncells2, ncells1);
    Kokkos::realloc(a3, nmb, ncells3, ncells2, ncells1);

    auto &nghbr = pmbp->pmb->nghbr;
    auto &mblev = pmbp->pmb->mb_lev;
    auto trs = xrb;

    par_for("xrb_vector_potential", DevExeSpace(), 0,nmb-1,ks,ke+1,js,je+1,is,ie+1,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      int nx1 = indcs.nx1;
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
      Real x1f = LeftEdgeX(i-is, nx1, x1min, x1max);

      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      int nx2 = indcs.nx2;
      Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
      Real x2f = LeftEdgeX(j-js, nx2, x2min, x2max);

      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      int nx3 = indcs.nx3;
      Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
      Real x3f = LeftEdgeX(k-ks, nx3, x3min, x3max);

      Real dx1 = size.d_view(m).dx1;
      Real dx2 = size.d_view(m).dx2;
      Real dx3 = size.d_view(m).dx3;

      a1(m,k,j,i) = A1(trs, x1v, x2f, x3f);
      a2(m,k,j,i) = A2(trs, x1f, x2v, x3f);
      a3(m,k,j,i) = A3(trs, x1f, x2f, x3v);

      // Fine-neighbor edge averaging (torus / turb_seed scheme)
      if ((nghbr.d_view(m,8 ).lev > mblev.d_view(m) && j==js) ||
          (nghbr.d_view(m,9 ).lev > mblev.d_view(m) && j==js) ||
          (nghbr.d_view(m,10).lev > mblev.d_view(m) && j==js) ||
          (nghbr.d_view(m,11).lev > mblev.d_view(m) && j==js) ||
          (nghbr.d_view(m,12).lev > mblev.d_view(m) && j==je+1) ||
          (nghbr.d_view(m,13).lev > mblev.d_view(m) && j==je+1) ||
          (nghbr.d_view(m,14).lev > mblev.d_view(m) && j==je+1) ||
          (nghbr.d_view(m,15).lev > mblev.d_view(m) && j==je+1) ||
          (nghbr.d_view(m,24).lev > mblev.d_view(m) && k==ks) ||
          (nghbr.d_view(m,25).lev > mblev.d_view(m) && k==ks) ||
          (nghbr.d_view(m,26).lev > mblev.d_view(m) && k==ks) ||
          (nghbr.d_view(m,27).lev > mblev.d_view(m) && k==ks) ||
          (nghbr.d_view(m,28).lev > mblev.d_view(m) && k==ke+1) ||
          (nghbr.d_view(m,29).lev > mblev.d_view(m) && k==ke+1) ||
          (nghbr.d_view(m,30).lev > mblev.d_view(m) && k==ke+1) ||
          (nghbr.d_view(m,31).lev > mblev.d_view(m) && k==ke+1) ||
          (nghbr.d_view(m,40).lev > mblev.d_view(m) && j==js && k==ks) ||
          (nghbr.d_view(m,41).lev > mblev.d_view(m) && j==js && k==ks) ||
          (nghbr.d_view(m,42).lev > mblev.d_view(m) && j==je+1 && k==ks) ||
          (nghbr.d_view(m,43).lev > mblev.d_view(m) && j==je+1 && k==ks) ||
          (nghbr.d_view(m,44).lev > mblev.d_view(m) && j==js && k==ke+1) ||
          (nghbr.d_view(m,45).lev > mblev.d_view(m) && j==js && k==ke+1) ||
          (nghbr.d_view(m,46).lev > mblev.d_view(m) && j==je+1 && k==ke+1) ||
          (nghbr.d_view(m,47).lev > mblev.d_view(m) && j==je+1 && k==ke+1)) {
        Real xl = x1v + 0.25*dx1;
        Real xr = x1v - 0.25*dx1;
        a1(m,k,j,i) = 0.5*(A1(trs, xl,x2f,x3f) + A1(trs, xr,x2f,x3f));
      }

      if ((nghbr.d_view(m,0 ).lev > mblev.d_view(m) && i==is) ||
          (nghbr.d_view(m,1 ).lev > mblev.d_view(m) && i==is) ||
          (nghbr.d_view(m,2 ).lev > mblev.d_view(m) && i==is) ||
          (nghbr.d_view(m,3 ).lev > mblev.d_view(m) && i==is) ||
          (nghbr.d_view(m,4 ).lev > mblev.d_view(m) && i==ie+1) ||
          (nghbr.d_view(m,5 ).lev > mblev.d_view(m) && i==ie+1) ||
          (nghbr.d_view(m,6 ).lev > mblev.d_view(m) && i==ie+1) ||
          (nghbr.d_view(m,7 ).lev > mblev.d_view(m) && i==ie+1) ||
          (nghbr.d_view(m,24).lev > mblev.d_view(m) && k==ks) ||
          (nghbr.d_view(m,25).lev > mblev.d_view(m) && k==ks) ||
          (nghbr.d_view(m,26).lev > mblev.d_view(m) && k==ks) ||
          (nghbr.d_view(m,27).lev > mblev.d_view(m) && k==ks) ||
          (nghbr.d_view(m,28).lev > mblev.d_view(m) && k==ke+1) ||
          (nghbr.d_view(m,29).lev > mblev.d_view(m) && k==ke+1) ||
          (nghbr.d_view(m,30).lev > mblev.d_view(m) && k==ke+1) ||
          (nghbr.d_view(m,31).lev > mblev.d_view(m) && k==ke+1) ||
          (nghbr.d_view(m,32).lev > mblev.d_view(m) && i==is && k==ks) ||
          (nghbr.d_view(m,33).lev > mblev.d_view(m) && i==is && k==ks) ||
          (nghbr.d_view(m,34).lev > mblev.d_view(m) && i==ie+1 && k==ks) ||
          (nghbr.d_view(m,35).lev > mblev.d_view(m) && i==ie+1 && k==ks) ||
          (nghbr.d_view(m,36).lev > mblev.d_view(m) && i==is && k==ke+1) ||
          (nghbr.d_view(m,37).lev > mblev.d_view(m) && i==is && k==ke+1) ||
          (nghbr.d_view(m,38).lev > mblev.d_view(m) && i==ie+1 && k==ke+1) ||
          (nghbr.d_view(m,39).lev > mblev.d_view(m) && i==ie+1 && k==ke+1)) {
        Real xl = x2v + 0.25*dx2;
        Real xr = x2v - 0.25*dx2;
        a2(m,k,j,i) = 0.5*(A2(trs, x1f,xl,x3f) + A2(trs, x1f,xr,x3f));
      }

      if ((nghbr.d_view(m,0 ).lev > mblev.d_view(m) && i==is) ||
          (nghbr.d_view(m,1 ).lev > mblev.d_view(m) && i==is) ||
          (nghbr.d_view(m,2 ).lev > mblev.d_view(m) && i==is) ||
          (nghbr.d_view(m,3 ).lev > mblev.d_view(m) && i==is) ||
          (nghbr.d_view(m,4 ).lev > mblev.d_view(m) && i==ie+1) ||
          (nghbr.d_view(m,5 ).lev > mblev.d_view(m) && i==ie+1) ||
          (nghbr.d_view(m,6 ).lev > mblev.d_view(m) && i==ie+1) ||
          (nghbr.d_view(m,7 ).lev > mblev.d_view(m) && i==ie+1) ||
          (nghbr.d_view(m,8 ).lev > mblev.d_view(m) && j==js) ||
          (nghbr.d_view(m,9 ).lev > mblev.d_view(m) && j==js) ||
          (nghbr.d_view(m,10).lev > mblev.d_view(m) && j==js) ||
          (nghbr.d_view(m,11).lev > mblev.d_view(m) && j==js) ||
          (nghbr.d_view(m,12).lev > mblev.d_view(m) && j==je+1) ||
          (nghbr.d_view(m,13).lev > mblev.d_view(m) && j==je+1) ||
          (nghbr.d_view(m,14).lev > mblev.d_view(m) && j==je+1) ||
          (nghbr.d_view(m,15).lev > mblev.d_view(m) && j==je+1) ||
          (nghbr.d_view(m,16).lev > mblev.d_view(m) && i==is && j==js) ||
          (nghbr.d_view(m,17).lev > mblev.d_view(m) && i==is && j==js) ||
          (nghbr.d_view(m,18).lev > mblev.d_view(m) && i==ie+1 && j==js) ||
          (nghbr.d_view(m,19).lev > mblev.d_view(m) && i==ie+1 && j==js) ||
          (nghbr.d_view(m,20).lev > mblev.d_view(m) && i==is && j==je+1) ||
          (nghbr.d_view(m,21).lev > mblev.d_view(m) && i==is && j==je+1) ||
          (nghbr.d_view(m,22).lev > mblev.d_view(m) && i==ie+1 && j==je+1) ||
          (nghbr.d_view(m,23).lev > mblev.d_view(m) && i==ie+1 && j==je+1)) {
        Real xl = x3v + 0.25*dx3;
        Real xr = x3v - 0.25*dx3;
        a3(m,k,j,i) = 0.5*(A3(trs, x1f,x2f,xl) + A3(trs, x1f,x2f,xr));
      }
    });

    auto &b0 = pmbp->pmhd->b0;
    par_for("xrb_b0", DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real dx1 = size.d_view(m).dx1;
      Real dx2 = size.d_view(m).dx2;
      Real dx3 = size.d_view(m).dx3;

      b0.x1f(m,k,j,i) = ((a3(m,k,j+1,i) - a3(m,k,j,i))/dx2 -
                         (a2(m,k+1,j,i) - a2(m,k,j,i))/dx3);
      b0.x2f(m,k,j,i) = ((a1(m,k+1,j,i) - a1(m,k,j,i))/dx3 -
                         (a3(m,k,j,i+1) - a3(m,k,j,i))/dx1);
      b0.x3f(m,k,j,i) = ((a2(m,k,j,i+1) - a2(m,k,j,i))/dx1 -
                         (a1(m,k,j+1,i) - a1(m,k,j,i))/dx2);

      if (i==ie) {
        b0.x1f(m,k,j,i+1) = ((a3(m,k,j+1,i+1) - a3(m,k,j,i+1))/dx2 -
                             (a2(m,k+1,j,i+1) - a2(m,k,j,i+1))/dx3);
      }
      if (j==je) {
        b0.x2f(m,k,j+1,i) = ((a1(m,k+1,j+1,i) - a1(m,k,j+1,i))/dx3 -
                             (a3(m,k,j+1,i+1) - a3(m,k,j+1,i))/dx1);
      }
      if (k==ke) {
        b0.x3f(m,k+1,j,i) = ((a2(m,k+1,j,i+1) - a2(m,k+1,j,i))/dx1 -
                             (a1(m,k+1,j+1,i) - a1(m,k+1,j,i))/dx2);
      }
    });

    auto &bcc_ = pmbp->pmhd->bcc0;
    par_for("xrb_bcc", DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      bcc_(m,IBX,k,j,i) = 0.5*(b0.x1f(m,k,j,i) + b0.x1f(m,k,j,i+1));
      bcc_(m,IBY,k,j,i) = 0.5*(b0.x2f(m,k,j,i) + b0.x2f(m,k,j+1,i));
      bcc_(m,IBZ,k,j,i) = 0.5*(b0.x3f(m,k,j,i) + b0.x3f(m,k+1,j,i));
    });

    // Normalize so min(p/p_mag) = potential_beta_min in the live envelope
    // (outside the donor mask, inside r_donor, and rho > cut). The 1/r core is
    // excluded: volume means / peaks over all rho>cut are dominated by it.
    const Real arad_ = is_radiation_enabled ? pmbp->prad->arad : 0.0;
    const bool is_rad = is_radiation_enabled;
    Real beta_min = std::numeric_limits<Real>::max();
    Real pmag_sum = 0.0, pgas_sum = 0.0, vol_sum = 0.0;
    Real pgas_max = 0.0, pmag_max = 0.0;
    Kokkos::parallel_reduce("xrb_beta_norm",
                            Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int idx, Real &lbmin, Real &lpmag, Real &lpgas, Real &lvol,
                  Real &lpmax, Real &lpmagmax) {
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/indcs.nx1;
      int i = (idx - m*nkji - k*nji - j*indcs.nx1) + is;
      k += ks;
      j += js;

      Real rho = w0_(m,IDN,k,j,i);
      if (rho <= trs.potential_rho_cut) return;

      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);
      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);
      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);
      Real d_donor = sqrt(SQR(x1v - trs.a_sep) + SQR(x2v) + SQR(x3v));
      if (d_donor <= trs.r_donor_mask || d_donor >= trs.r_donor) return;

      Real vol = size.d_view(m).dx1 * size.d_view(m).dx2 * size.d_view(m).dx3;
      Real pgas = gm1 * w0_(m,IEN,k,j,i);
      if (is_rad) {
        Real tgas = pgas / rho;
        pgas += arad_ * SQR(SQR(tgas)) / 3.0;
      }
      Real bx = bcc_(m,IBX,k,j,i);
      Real by = bcc_(m,IBY,k,j,i);
      Real bz = bcc_(m,IBZ,k,j,i);
      Real pmag = 0.5*(SQR(bx) + SQR(by) + SQR(bz));
      lpmag += pmag * vol;
      lpgas += pgas * vol;
      lvol  += vol;
      lpmax = fmax(lpmax, pgas);
      lpmagmax = fmax(lpmagmax, pmag);
      if (pmag > 0.0) {
        lbmin = fmin(lbmin, pgas / pmag);
      }
    }, Kokkos::Min<Real>(beta_min), pmag_sum, pgas_sum, vol_sum,
       Kokkos::Max<Real>(pgas_max), Kokkos::Max<Real>(pmag_max));

#if MPI_PARALLEL_ENABLED
    Real red_sum[3] = {pmag_sum, pgas_sum, vol_sum};
    MPI_Allreduce(MPI_IN_PLACE, red_sum, 3, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
    pmag_sum = red_sum[0];
    pgas_sum = red_sum[1];
    vol_sum = red_sum[2];
    MPI_Allreduce(MPI_IN_PLACE, &beta_min, 1, MPI_ATHENA_REAL, MPI_MIN, MPI_COMM_WORLD);
    Real red_max[2] = {pgas_max, pmag_max};
    MPI_Allreduce(MPI_IN_PLACE, red_max, 2, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
    pgas_max = red_max[0];
    pmag_max = red_max[1];
#endif

    Real bnorm = 0.0;
    if (vol_sum > 0.0 && pmag_sum > 0.0 && std::isfinite(beta_min) &&
        beta_min > 0.0 && beta_min < std::numeric_limits<Real>::max()) {
      bnorm = sqrt(beta_min / xrb.potential_beta_min);
    }
    if (!std::isfinite(bnorm) || bnorm <= 0.0) {
      bnorm = 0.0;
      if (global_variable::my_rank == 0) {
        std::cout << "### WARNING in " << __FILE__ << " at line " << __LINE__ << std::endl
                  << "donor vector potential is zero or degenerate in the envelope; "
                  << "skipping magnetic field normalization (b0=0)" << std::endl;
      }
    }

    Real bnorm_ = bnorm;
    par_for("xrb_normb0", DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      b0.x1f(m,k,j,i) *= bnorm_;
      b0.x2f(m,k,j,i) *= bnorm_;
      b0.x3f(m,k,j,i) *= bnorm_;
      if (i==ie) { b0.x1f(m,k,j,i+1) *= bnorm_; }
      if (j==je) { b0.x2f(m,k,j+1,i) *= bnorm_; }
      if (k==ke) { b0.x3f(m,k+1,j,i) *= bnorm_; }
    });

    par_for("xrb_normbcc", DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      bcc_(m,IBX,k,j,i) = 0.5*(b0.x1f(m,k,j,i) + b0.x1f(m,k,j,i+1));
      bcc_(m,IBY,k,j,i) = 0.5*(b0.x2f(m,k,j,i) + b0.x2f(m,k,j+1,i));
      bcc_(m,IBZ,k,j,i) = 0.5*(b0.x3f(m,k,j,i) + b0.x3f(m,k+1,j,i));
    });

    if (global_variable::my_rank == 0) {
      Real beta_mean = (vol_sum > 0.0 && pmag_sum > 0.0) ? (pgas_sum / pmag_sum)
                                                        : std::numeric_limits<Real>::infinity();
      Real beta_peak = (pmag_max > 0.0) ? (pgas_max / pmag_max)
                                        : std::numeric_limits<Real>::infinity();
      Real scale = (bnorm > 0.0) ? (bnorm * bnorm) : 0.0;
      std::cout << "xrb poloidal mean field: bnorm = " << bnorm
                << " (target envelope min beta = " << xrb.potential_beta_min << ")"
                << std::endl;
      std::cout << "  envelope unnorm: beta_min = " << beta_min
                << "  <p>/<p_mag> = " << beta_mean
                << "  p_max/p_mag_max = " << beta_peak << std::endl;
      if (scale > 0.0) {
        std::cout << "  envelope after norm: beta_min = " << (beta_min / scale)
                  << "  <p>/<p_mag> = " << (beta_mean / scale)
                  << "  p_max/p_mag_max = " << (beta_peak / scale) << std::endl;
      }
    }
  }

  // One-shot turbulence / B seeds (operate on primitives / face B; before PrimToCons)
  for (auto it = pin->block.begin(); it != pin->block.end(); ++it) {
    if (it->block_name.compare(0, 9, "turb_seed") == 0) {
      TurbSeed tseed(it->block_name, pmbp, pin);
      tseed.Apply();
    }
  }

  // Convert primitives to conserved
  if (use_mhd) {
    auto &bcc0_ = pmbp->pmhd->bcc0;
    pmbp->pmhd->peos->PrimToCons(w0_, bcc0_, u0_, is, ie, js, je, ks, ke);
  } else {
    pmbp->phydro->peos->PrimToCons(w0_, u0_, is, ie, js, je, ks, ke);
  }
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
Real CalculateRochePotential(struct xrb_pgen pgen, Real x1, Real x2, Real x3) {
  Real d_accretor = sqrt(SQR(x1) + SQR(x2) + SQR(x3)); // distance to accretor
  Real d_donor = sqrt(SQR(x1-pgen.a_sep) + SQR(x2) + SQR(x3)); // distance to donor
  Real m_tot = pgen.m_accretor + pgen.m_donor;
  Real omega_sq = m_tot / (pgen.a_sep*pgen.a_sep*pgen.a_sep);
  Real x_com = pgen.m_donor * pgen.a_sep / m_tot;
  Real potential = -pgen.m_donor/d_donor - pgen.m_accretor/d_accretor;
  potential -= 0.5*omega_sq*(SQR(x1 - x_com) + SQR(x2));
  return potential;
}

KOKKOS_INLINE_FUNCTION
static void CalculateBackgroundState(struct xrb_pgen pgen,
                                   Real x1, Real x2, Real x3,
                                   Real dx1, Real dx2, Real dx3,
                                   Real *prho_bg, Real *ppgas_bg) {
  if (pgen.is_gr) {
    Real r, theta, phi;
    GetBoyerLindquistCoordinates(pgen, x1, x2, x3, &r, &theta, &phi);

    Real r_excise_cell, theta_excise, phi_excise;
    GetBoyerLindquistCoordinates(pgen, x1 + copysign(0.5*dx1, x1),
                                      x2 + copysign(0.5*dx2, x2),
                                      x3 + copysign(0.5*dx3, x3),
                                      &r_excise_cell, &theta_excise, &phi_excise);
    if (r_excise_cell > 1.0) {
      *prho_bg = pgen.rho_min * pow(r, pgen.rho_pow);
      *ppgas_bg = pgen.pgas_min * pow(r, pgen.pgas_pow);
    } else {
      *prho_bg = pgen.dexcise;
      *ppgas_bg = pgen.pexcise;
    }
  } else {
    Real r = sqrt(SQR(x1) + SQR(x2) + SQR(x3));
    Real r_eff = fmax(r, pgen.r_soft);
    *prho_bg = pgen.rho_min * pow(r_eff, pgen.rho_pow);
    *ppgas_bg = pgen.pgas_min * pow(r_eff, pgen.pgas_pow);
  }
}

KOKKOS_INLINE_FUNCTION
static void CalculateDonorState(struct xrb_pgen pgen,
                                Real x1, Real x2, Real x3,
                                Real dx1, Real dx2, Real dx3,
                                Real *prho, Real *ppgas) {
  Real rho_bg, pgas_bg;
  CalculateBackgroundState(pgen, x1, x2, x3, dx1, dx2, dx3, &rho_bg, &pgas_bg);

  Real rho = rho_bg;
  Real pgas = pgas_bg;
  Real gm1 = pgen.gamma_adi - 1.0;

  Real potential_surface =
    CalculateRochePotential(pgen, pgen.a_sep - pgen.r_donor, 0.0, 0.0);
  Real potential = CalculateRochePotential(pgen, x1, x2, x3);
  Real x_d = x1 - pgen.a_sep;
  Real d_donor = sqrt(SQR(x_d) + SQR(x2) + SQR(x3));
  Real delta_phi = potential_surface - potential;
  if (d_donor < pgen.r_donor && delta_phi > 0.0) {
    rho = pow((gm1/(pgen.gamma_adi*pgen.k_adi))*delta_phi, 1.0/gm1);
    pgas = pgen.k_adi * pow(rho, pgen.gamma_adi);
  }

  *prho = fmax(rho, rho_bg);
  *ppgas = fmax(pgas, pgas_bg);
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

  // Donor point-mass gravity (skip inside the stellar inner mask).
  if (d_d > 0.9*pgen.r_donor_mask) {
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

  DvceArray5D<Real> w0_, u0_;
  bool is_ideal;
  if (pmbp->pmhd != nullptr) {
    w0_ = pmbp->pmhd->w0;
    u0_ = pmbp->pmhd->u0;
    is_ideal = pmbp->pmhd->peos->eos_data.is_ideal;
  } else {
    w0_ = pmbp->phydro->w0;
    u0_ = pmbp->phydro->u0;
    is_ideal = pmbp->phydro->peos->eos_data.is_ideal;
  }

  auto pgen = xrb;
  int nmb1 = pmbp->nmb_thispack - 1;
  Real time = pm->time;

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

      // Scherbak et al. 2025 Eq. (7): Gaussian in Roche potential, tanh time ramp.
      if (pgen.heating) {
        Real phi = CalculateRochePotential(pgen, x1v, x2v, x3v);
        Real profile = HeatProfile(pgen, phi, x1v, x2v, x3v);
        if (profile > 0.0) {
          Real ramp = 0.5*(tanh((time - pgen.heat_t0)/pgen.heat_dt_ramp) + 1.0);
          Real eps_heat = ramp * pgen.heating_norm * profile;
          if (pgen.is_gr) {
            u0_(m,IEN,k,j,i) -= bdt * eps_heat;
          } else {
            u0_(m,IEN,k,j,i) += bdt * eps_heat;
          }
        }
      }
    }
  });
}

//----------------------------------------------------------------------------------------
//! \brief Fill derived output var with the cell-by-cell heating rate density (Scherbak
//! et al. 2025 Eq. 7): eps_heat = ramp(t) * heating_norm * HeatProfile(Phi,x).
//! Enrolled as pm->pgen->user_derived_func for output variable "xrb_heat". Value is the
//! physical (positive) heating rate density regardless of the GR IEN sign convention.

void FillXRBHeatDerived(Mesh *pm, DvceArray5D<Real> dv, int i_dv) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;

  auto pgen = xrb;
  int nmb1 = pmbp->nmb_thispack - 1;
  Real time = pm->time;
  bool heating = xrb.heating;

  par_for("xrb_heat_dv", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
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

    Real eps_heat = 0.0;
    if (heating) {
      Real phi = CalculateRochePotential(pgen, x1v, x2v, x3v);
      Real profile = HeatProfile(pgen, phi, x1v, x2v, x3v);
      if (profile > 0.0) {
        Real ramp = 0.5*(tanh((time - pgen.heat_t0)/pgen.heat_dt_ramp) + 1.0);
        eps_heat = ramp * pgen.heating_norm * profile;
      }
    }
    dv(m,i_dv,k,j,i) = eps_heat;
  });
}

void ZeroDonorMaskEMF(Mesh *pm) {
  // Enrolled as user_efield_func; runs at the end of EFieldSrc, before SendE.
  if (!xrb.fix_donor) return;
  MeshBlockPack *pmbp = pm->pmb_pack;
  if (pmbp->pmhd == nullptr) return;

  auto &indcs = pm->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  auto &size = pmbp->pmb->mb_size;
  auto e1 = pmbp->pmhd->efld.x1e;
  auto e2 = pmbp->pmhd->efld.x2e;
  auto e3 = pmbp->pmhd->efld.x3e;
  auto pgen = xrb;
  int nmb1 = pmbp->nmb_thispack - 1;

  // e1[is:ie, js:je+1, ks:ke+1]: x1-edge touches cells (i, j-1..j, k-1..k)
  par_for("xrb_mask_e1", DevExeSpace(), 0, nmb1, ks, ke+1, js, je+1, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real x1min = size.d_view(m).x1min;
    Real x1max = size.d_view(m).x1max;
    Real x2min = size.d_view(m).x2min;
    Real x2max = size.d_view(m).x2max;
    Real x3min = size.d_view(m).x3min;
    Real x3max = size.d_view(m).x3max;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
    bool touch = false;
    for (int kk = k-1; kk <= k; ++kk) {
      Real x3v = CellCenterX(kk-ks, nx3, x3min, x3max);
      for (int jj = j-1; jj <= j; ++jj) {
        Real x2v = CellCenterX(jj-js, nx2, x2min, x2max);
        if (CellInDonorMask(pgen, x1v, x2v, x3v)) touch = true;
      }
    }
    if (touch) e1(m,k,j,i) = 0.0;
  });

  // e2[is:ie+1, js:je, ks:ke+1]: x2-edge touches cells (i-1..i, j, k-1..k)
  par_for("xrb_mask_e2", DevExeSpace(), 0, nmb1, ks, ke+1, js, je, is, ie+1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real x1min = size.d_view(m).x1min;
    Real x1max = size.d_view(m).x1max;
    Real x2min = size.d_view(m).x2min;
    Real x2max = size.d_view(m).x2max;
    Real x3min = size.d_view(m).x3min;
    Real x3max = size.d_view(m).x3max;
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
    bool touch = false;
    for (int kk = k-1; kk <= k; ++kk) {
      Real x3v = CellCenterX(kk-ks, nx3, x3min, x3max);
      for (int ii = i-1; ii <= i; ++ii) {
        Real x1v = CellCenterX(ii-is, nx1, x1min, x1max);
        if (CellInDonorMask(pgen, x1v, x2v, x3v)) touch = true;
      }
    }
    if (touch) e2(m,k,j,i) = 0.0;
  });

  // e3[is:ie+1, js:je+1, ks:ke]: x3-edge touches cells (i-1..i, j-1..j, k)
  par_for("xrb_mask_e3", DevExeSpace(), 0, nmb1, ks, ke, js, je+1, is, ie+1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real x1min = size.d_view(m).x1min;
    Real x1max = size.d_view(m).x1max;
    Real x2min = size.d_view(m).x2min;
    Real x2max = size.d_view(m).x2max;
    Real x3min = size.d_view(m).x3min;
    Real x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
    bool touch = false;
    for (int jj = j-1; jj <= j; ++jj) {
      Real x2v = CellCenterX(jj-js, nx2, x2min, x2max);
      for (int ii = i-1; ii <= i; ++ii) {
        Real x1v = CellCenterX(ii-is, nx1, x1min, x1max);
        if (CellInDonorMask(pgen, x1v, x2v, x3v)) touch = true;
      }
    }
    if (touch) e3(m,k,j,i) = 0.0;
  });
}

void ApplyDonorMask(Mesh *pm) {
  if (!xrb.fix_donor) return;

  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;
  const bool use_mhd = (pmbp->pmhd != nullptr);

  DvceArray5D<Real> w0_, u0_;
  if (use_mhd) {
    w0_ = pmbp->pmhd->w0;
    u0_ = pmbp->pmhd->u0;
  } else {
    w0_ = pmbp->phydro->w0;
    u0_ = pmbp->phydro->u0;
  }

  auto pgen = xrb;
  Real gm1 = pgen.gamma_adi - 1.0;
  int nmb1 = pmbp->nmb_thispack - 1;
  auto &coord = pmbp->pcoord->coord_data;

  if (use_mhd) {
    // Fluid reset inside mask; face B is not overwritten. Interior B is frozen
    // by E=0 on mask-cell edges in ZeroDonorMaskEMF (end of MHD::EFieldSrc, before SendE).
    // Refresh bcc from faces and set conserved via MHD PrimToCons algebra.
    auto &b0 = pmbp->pmhd->b0;
    auto &bcc_ = pmbp->pmhd->bcc0;
    Real gamma = pgen.gamma_adi;

    par_for("xrb_donor_mask_mhd", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
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

      Real &dx1 = size.d_view(m).dx1;
      Real &dx2 = size.d_view(m).dx2;
      Real &dx3 = size.d_view(m).dx3;

      Real x_d = x1v - pgen.a_sep;
      Real d_donor = sqrt(SQR(x_d) + SQR(x2v) + SQR(x3v));
      if (d_donor > pgen.r_donor_mask) return;

      Real rho, pgas;
      CalculateDonorState(pgen, x1v, x2v, x3v, dx1, dx2, dx3, &rho, &pgas);
      Real eint = pgas / gm1;

      Real bx = 0.5*(b0.x1f(m,k,j,i) + b0.x1f(m,k,j,i+1));
      Real by = 0.5*(b0.x2f(m,k,j,i) + b0.x2f(m,k,j+1,i));
      Real bz = 0.5*(b0.x3f(m,k,j,i) + b0.x3f(m,k+1,j,i));
      bcc_(m,IBX,k,j,i) = bx;
      bcc_(m,IBY,k,j,i) = by;
      bcc_(m,IBZ,k,j,i) = bz;

      w0_(m,IDN,k,j,i) = rho;
      w0_(m,IVX,k,j,i) = 0.0;
      w0_(m,IVY,k,j,i) = 0.0;
      w0_(m,IVZ,k,j,i) = 0.0;
      w0_(m,IEN,k,j,i) = eint;

      MHDPrim1D w;
      w.d = rho; w.vx = 0.0; w.vy = 0.0; w.vz = 0.0; w.e = eint;
      w.bx = bx; w.by = by; w.bz = bz;
      HydCons1D u;
      if (pgen.is_gr) {
        Real glower[4][4], gupper[4][4];
        ComputeMetricAndInverse(x1v, x2v, x3v, coord.is_minkowski, coord.bh_spin,
                                glower, gupper);
        SingleP2C_IdealGRMHD(glower, gupper, w, gamma, u);
      } else {
        SingleP2C_IdealMHD(w, u);
      }
      u0_(m,IDN,k,j,i) = u.d;
      u0_(m,IM1,k,j,i) = u.mx;
      u0_(m,IM2,k,j,i) = u.my;
      u0_(m,IM3,k,j,i) = u.mz;
      u0_(m,IEN,k,j,i) = u.e;
    });
  } else {
    par_for("xrb_donor_mask", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
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

      Real &dx1 = size.d_view(m).dx1;
      Real &dx2 = size.d_view(m).dx2;
      Real &dx3 = size.d_view(m).dx3;

      Real x_d = x1v - pgen.a_sep;
      Real d_donor = sqrt(SQR(x_d) + SQR(x2v) + SQR(x3v));
      if (d_donor <= pgen.r_donor_mask) {
        Real rho, pgas;
        CalculateDonorState(pgen, x1v, x2v, x3v, dx1, dx2, dx3, &rho, &pgas);
        Real eint = pgas / gm1;
        w0_(m,IDN,k,j,i) = rho;
        w0_(m,IVX,k,j,i) = 0.0;
        w0_(m,IVY,k,j,i) = 0.0;
        w0_(m,IVZ,k,j,i) = 0.0;
        u0_(m,IDN,k,j,i) = rho;
        u0_(m,IM1,k,j,i) = 0.0;
        u0_(m,IM2,k,j,i) = 0.0;
        u0_(m,IM3,k,j,i) = 0.0;
        if (pgen.is_gr) {
          w0_(m,IEN,k,j,i) = -eint;
          u0_(m,IEN,k,j,i) = -eint;
        } else {
          w0_(m,IEN,k,j,i) = eint;
          u0_(m,IEN,k,j,i) = eint;
        }
      }
    });
  }
}

void UserBcsXRB(Mesh *pm) {
  ApplyDonorMask(pm);
  NoInflowXRB(pm);
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
  MeshBlockPack *pmbp = pm->pmb_pack;
  const bool use_mhd = (pmbp->pmhd != nullptr);

  DvceArray5D<Real> u0_, w0_, bcc_;
  if (use_mhd) {
    u0_ = pmbp->pmhd->u0;
    w0_ = pmbp->pmhd->w0;
    bcc_ = pmbp->pmhd->bcc0;
  } else {
    u0_ = pmbp->phydro->u0;
    w0_ = pmbp->phydro->w0;
  }
  int nmb = pmbp->nmb_thispack;
  int nvar = u0_.extent_int(1);

  const bool is_gr = pmbp->pcoord->is_general_relativistic;
  const bool is_radiation_enabled = (pmbp->prad != nullptr);
  DvceArray5D<Real> i0_; int nang1 = 0;
  if (is_gr && is_radiation_enabled) {
    i0_ = pmbp->prad->i0;
    nang1 = pmbp->prad->prgeo->nangles - 1;
  }

  // Face B ghosts come from MHD::BFieldBCs; user BC only adjusts fluid (+ rad).
  if (use_mhd) {
    auto &b0 = pmbp->pmhd->b0;
    pmbp->pmhd->peos->ConsToPrim(u0_,b0,w0_,bcc_,false,is-ng,is,0,(n2-1),0,(n3-1));
    pmbp->pmhd->peos->ConsToPrim(u0_,b0,w0_,bcc_,false,ie,ie+ng,0,(n2-1),0,(n3-1));
  } else {
    pmbp->phydro->peos->ConsToPrim(u0_,w0_,false,is-ng,is,0,(n2-1),0,(n3-1));
    pmbp->phydro->peos->ConsToPrim(u0_,w0_,false,ie,ie+ng,0,(n2-1),0,(n3-1));
  }

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
  if (use_mhd) {
    auto &b0 = pmbp->pmhd->b0;
    pmbp->pmhd->peos->PrimToCons(w0_,bcc_,u0_,is-ng,is-1,0,(n2-1),0,(n3-1));
    pmbp->pmhd->peos->PrimToCons(w0_,bcc_,u0_,ie+1,ie+ng,0,(n2-1),0,(n3-1));
    pmbp->pmhd->peos->ConsToPrim(u0_,b0,w0_,bcc_,false,0,(n1-1),js-ng,js,0,(n3-1));
    pmbp->pmhd->peos->ConsToPrim(u0_,b0,w0_,bcc_,false,0,(n1-1),je,je+ng,0,(n3-1));
  } else {
    pmbp->phydro->peos->PrimToCons(w0_,u0_,is-ng,is-1,0,(n2-1),0,(n3-1));
    pmbp->phydro->peos->PrimToCons(w0_,u0_,ie+1,ie+ng,0,(n2-1),0,(n3-1));
    pmbp->phydro->peos->ConsToPrim(u0_,w0_,false,0,(n1-1),js-ng,js,0,(n3-1));
    pmbp->phydro->peos->ConsToPrim(u0_,w0_,false,0,(n1-1),je,je+ng,0,(n3-1));
  }

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
  if (use_mhd) {
    auto &b0 = pmbp->pmhd->b0;
    pmbp->pmhd->peos->PrimToCons(w0_,bcc_,u0_,0,(n1-1),js-ng,js-1,0,(n3-1));
    pmbp->pmhd->peos->PrimToCons(w0_,bcc_,u0_,0,(n1-1),je+1,je+ng,0,(n3-1));
    pmbp->pmhd->peos->ConsToPrim(u0_,b0,w0_,bcc_,false,0,(n1-1),0,(n2-1),ks-ng,ks);
    pmbp->pmhd->peos->ConsToPrim(u0_,b0,w0_,bcc_,false,0,(n1-1),0,(n2-1),ke,ke+ng);
  } else {
    pmbp->phydro->peos->PrimToCons(w0_,u0_,0,(n1-1),js-ng,js-1,0,(n3-1));
    pmbp->phydro->peos->PrimToCons(w0_,u0_,0,(n1-1),je+1,je+ng,0,(n3-1));
    pmbp->phydro->peos->ConsToPrim(u0_,w0_,false,0,(n1-1),0,(n2-1),ks-ng,ks);
    pmbp->phydro->peos->ConsToPrim(u0_,w0_,false,0,(n1-1),0,(n2-1),ke,ke+ng);
  }

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
  if (use_mhd) {
    pmbp->pmhd->peos->PrimToCons(w0_,bcc_,u0_,0,(n1-1),0,(n2-1),ks-ng,ks-1);
    pmbp->pmhd->peos->PrimToCons(w0_,bcc_,u0_,0,(n1-1),0,(n2-1),ke+1,ke+ng);
  } else {
    pmbp->phydro->peos->PrimToCons(w0_,u0_,0,(n1-1),0,(n2-1),ks-ng,ks-1);
    pmbp->phydro->peos->PrimToCons(w0_,u0_,0,(n1-1),0,(n2-1),ke+1,ke+ng);
  }
}

void AccretorFluxes(HistoryData *pdata, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  const bool is_gr = pmbp->pcoord->is_general_relativistic;

  int nvars;
  Real gamma;
  DvceArray5D<Real> w0_;
  if (pmbp->pmhd != nullptr) {
    nvars = pmbp->pmhd->nmhd + pmbp->pmhd->nscalars;
    gamma = pmbp->pmhd->peos->eos_data.gamma;
    w0_ = pmbp->pmhd->w0;
  } else {
    nvars = pmbp->phydro->nhydro + pmbp->phydro->nscalars;
    gamma = pmbp->phydro->peos->eos_data.gamma;
    w0_ = pmbp->phydro->w0;
  }
  const Real gm1 = gamma - 1.0;

  auto &grids = pm->pgen->spherical_grids;
  int nradii = grids.size();
  const int nflux = 3;
  const bool donor_hist = xrb.donor_hist;
  const int n_acc = donor_hist ? (nradii - 1) : nradii;
  const int n_donor = donor_hist ? 3 : 0;  // mdot_donor, rho_donor, pgas_donor

  pdata->nhist = n_acc*nflux + n_donor;
  if (pdata->nhist > NHISTORY_VARIABLES) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "User history function specified pdata->nhist larger than"
              << " NHISTORY_VARIABLES" << std::endl;
    exit(EXIT_FAILURE);
  }
  for (int g=0; g<n_acc; ++g) {
    std::stringstream stream;
    stream << std::fixed << std::setprecision(1) << grids[g]->radius;
    std::string rad_str = stream.str();
    pdata->label[nflux*g+0] = "mdot_" + rad_str;
    pdata->label[nflux*g+1] = "edot_" + rad_str;
    pdata->label[nflux*g+2] = "ldot_" + rad_str;
  }
  if (donor_hist) {
    const int i0 = n_acc*nflux;
    // labels <= 10 chars (history header truncates with %.10s)
    pdata->label[i0+0] = "mdot_donor";  // outflow from donor > 0
    pdata->label[i0+1] = "rho_donor";   // area-weighted <rho> on donor sphere
    pdata->label[i0+2] = "pgas_donor";  // area-weighted <P> on donor sphere
  }

  bool &flat = pmbp->pcoord->coord_data.is_minkowski;
  Real &spin = pmbp->pcoord->coord_data.bh_spin;

  for (int g=0; g<n_acc; ++g) {
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
        // solid_angles are already dΩ; area element is r^2 dΩ (no extra sinθ).
        Real sth = sin(theta);
        Real cph = cos(phi);
        Real sph = sin(phi);
        Real vr = int_vx*sth*cph + int_vy*sth*sph + int_vz*cos(theta);
        Real dA = SQR(r)*domega;

        pdata->hdata[nflux*g+0] += -1.0*int_dn*vr*dA;

        Real edot_flux = int_dn*(0.5*(SQR(int_vx)+SQR(int_vy)+SQR(int_vz)) + gamma*int_ie)*vr;
        pdata->hdata[nflux*g+1] += -1.0*edot_flux*dA;

        Real ldot_flux = int_dn*vr*(x1*int_vy - x2*int_vx);
        pdata->hdata[nflux*g+2] += ldot_flux*dA;
      }
    }
  }

  // Donor-centered sphere: mdot_donor > 0 is mass leaving the donor.
  // Area averages use the same dA weight as the Newtonian accretor fluxes.
  if (donor_hist) {
    const int g = n_acc;  // last grid
    const int i0 = n_acc*nflux;
    grids[g]->InterpolateToSphere(nvars, w0_);

    Real mdot_loc = 0.0;
    Real sum_rho_A = 0.0;
    Real sum_pgas_A = 0.0;
    Real sum_A = 0.0;
    for (int n=0; n<grids[g]->nangles; ++n) {
      if (!grids[g]->AngleIsLocal(n)) continue;

      Real r = grids[g]->radius;
      Real theta = grids[g]->polar_pos.h_view(n,0);
      Real phi = grids[g]->polar_pos.h_view(n,1);
      Real &int_dn = grids[g]->interp_vals.h_view(n,IDN);
      Real &int_vx = grids[g]->interp_vals.h_view(n,IVX);
      Real &int_vy = grids[g]->interp_vals.h_view(n,IVY);
      Real &int_vz = grids[g]->interp_vals.h_view(n,IVZ);
      Real int_ie = grids[g]->interp_vals.h_view(n,IEN);
      Real &domega = grids[g]->solid_angles.h_view(n);

      Real sth = sin(theta);
      Real cph = cos(phi);
      Real sph = sin(phi);
      // outward radial velocity relative to donor center
      Real vr = int_vx*sth*cph + int_vy*sth*sph + int_vz*cos(theta);
      // solid_angles are already dΩ; area element is r^2 dΩ (no extra sinθ).
      Real dA = SQR(r)*domega;
      Real pgas = gm1*int_ie;

      mdot_loc += int_dn*vr*dA;  // outflow positive
      sum_rho_A += int_dn*dA;
      sum_pgas_A += pgas*dA;
      sum_A += dA;
    }

#if MPI_PARALLEL_ENABLED
    Real reduce_buf[4] = {mdot_loc, sum_rho_A, sum_pgas_A, sum_A};
    MPI_Allreduce(MPI_IN_PLACE, reduce_buf, 4, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
    mdot_loc = reduce_buf[0];
    sum_rho_A = reduce_buf[1];
    sum_pgas_A = reduce_buf[2];
    sum_A = reduce_buf[3];
#endif

    // Rank 0 stores finals; others store 0 so history.cpp MPI_SUM is a no-op.
    if (global_variable::my_rank == 0) {
      pdata->hdata[i0+0] = mdot_loc;
      pdata->hdata[i0+1] = (sum_A > 0.0) ? (sum_rho_A/sum_A) : 0.0;
      pdata->hdata[i0+2] = (sum_A > 0.0) ? (sum_pgas_A/sum_A) : 0.0;
    } else {
      pdata->hdata[i0+0] = 0.0;
      pdata->hdata[i0+1] = 0.0;
      pdata->hdata[i0+2] = 0.0;
    }
  }

  for (int n=pdata->nhist; n<NHISTORY_VARIABLES; ++n) {
    pdata->hdata[n] = 0.0;
  }
}
