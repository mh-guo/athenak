//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file bondi.cpp
//! \brief Problem generator for spherically symmetric black hole accretion.

#include <cmath>   // abs(), NAN, pow(), sqrt()
#include <cstring> // strcmp()
#include <iostream>
#include <sstream>
#include <iomanip>
#include <memory>
#include <cstdio>
#include <string>
#include <algorithm>
#include <limits>

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

#include "pgen/turb_init.hpp"
#include "pgen/turb_mhd.hpp"

namespace {

KOKKOS_INLINE_FUNCTION
static void ComputePrimitiveSingle(Real x1v, Real x2v, Real x3v, CoordData coord,
                                   struct bondi_pgen pgen,
                                   Real& rho, Real& pgas,
                                   Real& uu1, Real& uu2, Real& uu3);

KOKKOS_INLINE_FUNCTION
static void GetBoyerLindquistCoordinates(struct bondi_pgen pgen,
                                         Real x1, Real x2, Real x3,
                                         Real *pr, Real *ptheta, Real *pphi);

KOKKOS_INLINE_FUNCTION
static void TransformVector(struct bondi_pgen pgen,
                            Real a1_bl, Real a2_bl, Real a3_bl,
                            Real x1, Real x2, Real x3,
                            Real *pa1, Real *pa2, Real *pa3);

KOKKOS_INLINE_FUNCTION
static void CalculatePrimitives(struct bondi_pgen pgen, Real r,
                                Real *prho, Real *ppgas, Real *pur);

KOKKOS_INLINE_FUNCTION
static Real TemperatureMin(struct bondi_pgen pgen, Real r, Real t_min, Real t_max);

KOKKOS_INLINE_FUNCTION
static Real TemperatureBisect(struct bondi_pgen pgen, Real r, Real t_min, Real t_max);

KOKKOS_INLINE_FUNCTION
static Real TemperatureResidual(struct bondi_pgen pgen, Real t, Real r);

KOKKOS_INLINE_FUNCTION
static void CalculatePrimitivesNewtonian(struct bondi_pgen pgen, Real r,
                                         Real *prho, Real *ppgas, Real *pur);

struct bondi_pgen {
  bool is_gr = false;       // true if GR
  Real r_in = 1.0;          // inner radius
  Real spin;                // black hole spin
  Real dexcise, pexcise;    // excision parameters
  Real n_adi, k_adi, gm;    // hydro EOS parameters
  Real r_crit;              // sonic point radius
  Real temp_inf, c_s_inf;   // asymptotic temperature and sound speed
  Real rho_inf, pgas_inf;   // asymptotic density and pressure
  Real r_bondi;             // Bondi radius
  Real c1, c2;              // useful constants
  Real temp_min, temp_max;  // bounds for temperature root find
  Real b_ini;               // initial magnetic field
  bool reset_ic = false;    // reset initial conditions after run
  int  ndiag = 0;           // number of cycles between diagnostics
  bool multi_zone = false;  // true if multi-zone
  bool fixed_zone = false;  // true if fixed-zone
  int  mz_type = 0;         // type of multi-zone
  int  vcycle_n = 0;        // number of cycles in a V-cycle
  int  mz_beg_level = 0;    // beg level of multi-zone
  int  mz_end_level = 0;    // end level of multi-zone
  int  mz_max_level = 0;    // max level of multi-zone
  int  mz_level = 0;        // current level in multi-zone
  int  mz_dir = 0;          // direction to move level, -1 for down, 1 for up
  Real mz_tf = 0.0;         // fraction of dynamical time spent at current level
  Real mz_tpow = 0.0;       // power law for duration of level
  Real mz_t_beg = 0.0;      // beg time of current level in multi-zone
  Real mz_t_end = 0.0;      // end time of current level in multi-zone
  Real mz_end_duration = 0.0; // duration of end level in multi-zone
  Real rv_in = 0.0;         // inner radius of V-cycle
  Real rv_out = 0.0;        // outer radius of V-cycle
  bool is_amr = false;      // true if AMR
  int  ncycle_amr = 0;      // number cycles between AMR
  int  beg_level = 0;       // beginning level for AMR
  int  end_level = 0;       // ending level for AMR
  Real r_refine = 0.0;      // mesh refinement radius
  Real slope = 0.0;         // slope of density and pressure
};

  bondi_pgen bondi;

KOKKOS_INLINE_FUNCTION
Real VcycleRadius(struct bondi_pgen pgen, int i, int n, Real rmin, Real rmax) {
  Real x = static_cast<Real>(i%n)/static_cast<Real>(n-1);
  Real y=0;
  if (pgen.mz_type==0) {
    y = fabs(1.0-2.0*x);
  } else {
    y = fmin(fmax(10.0*(fabs(2.0*x-1.0)-1.0)+1.0,0.0),1.0);
  }
  Real r = rmin*std::pow(rmax/rmin,y);
  int level = static_cast<int>(std::log2(r/rmin));
  r = rmin*std::pow(2.0,static_cast<Real>(level));
  r = (r<(10.0*pgen.r_in)) ? 0.0 : r;
  return r;
}

Real MultiZoneRadius(Real time) {
  Real r = std::pow(2.0,static_cast<Real>(bondi.mz_max_level-bondi.mz_level));
  r = (r<(12.0*bondi.r_in)) ? 0.0 : r;
  if (time>=bondi.mz_t_end) {
    std::cout << "MultiZoneRadius: old level = " << bondi.mz_level;
    bondi.mz_level += bondi.mz_dir;
    bondi.mz_dir = (bondi.mz_level==bondi.mz_beg_level) ? 1 : bondi.mz_dir;
    bondi.mz_dir = (bondi.mz_level==bondi.mz_end_level) ? -1 : bondi.mz_dir;
    bondi.mz_t_beg = time;
    r = bondi.r_in*std::pow(2.0,static_cast<Real>(bondi.mz_max_level-bondi.mz_level));
    r = (r<(12.0*bondi.r_in)) ? 0.0 : r;
    Real duration = bondi.mz_tf*pow(r/bondi.r_in,bondi.mz_tpow);
    duration = (bondi.mz_level==bondi.mz_beg_level)? 1.5*duration : duration;
    duration = (bondi.mz_level==bondi.mz_end_level)? bondi.mz_end_duration : duration;
    bondi.mz_t_end = bondi.mz_t_beg + duration;
    std::cout << ", new level = " << bondi.mz_level << ", radius = " << r
              << ", duration = " << duration << std::endl;
  }
  return r;
}
KOKKOS_INLINE_FUNCTION
int AMRLevel(int i, int n, int lbeg, int lend) {
  int nlevel = abs(lend-lbeg)+1;
  int j = static_cast<int>(i/n)%(2*(nlevel));
  int sign = (lend>lbeg) ? 1 : -1;
  int level = (j<(nlevel)) ? lbeg+sign*j : lend-sign*(j-nlevel);
  return level;
}

// prototypes for user-defined BCs and error functions
void FixedBondiInflow(Mesh *pm);
void AddUserSrcs(Mesh *pm, const Real bdt);
void BondiErrors(ParameterInput *pin, Mesh *pm);
void BondiFluxes(HistoryData *pdata, Mesh *pm);
Real BondiTimeStep(Mesh *pm);
void BondiRefine(MeshBlockPack* pmbp);
void Diagnostic(Mesh *pm);

} // namespace

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem()
//! \brief set initial conditions for Bondi accretion test
//  Compile with '-D PROBLEM=bondi' to enroll as user-specific problem generator
//    reference: Hawley, Smarr, & Wilson 1984, ApJ 277 296 (HSW)

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  bool is_gr = pmbp->pcoord->is_general_relativistic;
  bool is_mhd = (pmbp->pmhd != nullptr);
  auto peos = (is_mhd) ? pmbp->pmhd->peos : pmbp->phydro->peos;
  bool use_e = peos->eos_data.use_e;

  if (!(use_e)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "bondi test requires use_e=true" << std::endl;
    exit(EXIT_FAILURE);
  }

  // set user-defined BCs and error function pointers
  pgen_final_func = BondiErrors;
  user_bcs_func = FixedBondiInflow;
  user_srcs_func = AddUserSrcs;
  user_hist_func = BondiFluxes;
  user_dt_func = BondiTimeStep;
  user_ref_func = BondiRefine;

  // Reset time
  if (pin->GetOrAddBoolean("problem","reset",false)) {
    pmy_mesh_->ncycle = 0;
    pmy_mesh_->time = 0.0;
    for (auto it = pin->block.begin(); it != pin->block.end(); ++it) {
      if (it->block_name.compare(0, 6, "output") == 0) {
        pin->SetInteger(it->block_name,"file_number", 0);
        pin->SetReal(it->block_name,"last_time", -1.0e+100);
      }
    }
  }

  // Spherical Grid for user-defined history
  auto &grids = spherical_grids;
  int hist_nr = pin->GetOrAddInteger("problem","hist_nr",4);
  for (int i=0; i<hist_nr; i++) {
    Real rmin = is_gr ? 1.0+sqrt(1.0-SQR(pmbp->pcoord->coord_data.bh_spin)) : 1.0;
    Real rmax = pin->GetReal("mesh","x1max");
    Real r_i = std::pow(rmax/rmin,static_cast<Real>(i)/static_cast<Real>(hist_nr-1))*rmin;
    grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, r_i));
  }

  // Read problem-specific parameters from input file
  // global parameters
  bondi.is_gr = is_gr;
  bondi.k_adi = pin->GetReal("problem", "k_adi");
  Real b_ini = bondi.b_ini = pin->GetOrAddReal("problem", "b_ini", 0.0);

  bondi.ndiag = pin->GetOrAddInteger("problem", "ndiag", 0);

  bondi.multi_zone = pin->GetOrAddBoolean("coord", "multi_zone", false);
  bondi.fixed_zone = pin->GetOrAddBoolean("coord", "fixed_zone", false);
  if (bondi.multi_zone) {
    bondi.mz_type = pin->GetOrAddInteger("problem", "mz_type", 0);
    if (bondi.mz_type>1) {
      bondi.mz_beg_level = pin->GetInteger("problem", "beg_level");
      bondi.mz_end_level = pin->GetInteger("problem", "end_level");
      bondi.mz_max_level = pmy_mesh_->max_level;
      bondi.mz_dir = (bondi.mz_beg_level<bondi.mz_end_level) ? 1 : -1;
      bondi.mz_level = bondi.mz_beg_level - bondi.mz_dir;
      bondi.mz_tf = pin->GetReal("problem", "mz_tf");
      bondi.mz_tpow = pin->GetReal("problem", "mz_tpow");
      bondi.mz_end_duration = pin->GetReal("problem", "end_duration");
      bondi.mz_t_beg = 0.0;
      bondi.mz_t_end = 0.0;
    } else {
      bondi.vcycle_n = pin->GetInteger("problem", "vcycle_n");
      bondi.rv_in = pin->GetReal("problem", "rv_in");
      bondi.rv_out = pin->GetReal("problem", "rv_out");
    }
  }

  bondi.is_amr = pmy_mesh_->adaptive;
  if (bondi.is_amr) {
    bondi.ncycle_amr = pin->GetInteger("problem","ncycle_amr");
    bondi.beg_level = pin->GetInteger("problem","beg_level");
    bondi.end_level = pin->GetInteger("problem","end_level");
    bondi.r_refine = pin->GetOrAddReal("problem","r_refine",0.0);
  }

  // Get ideal gas EOS data
  bondi.gm = peos->eos_data.gamma;
  Real gm1 = bondi.gm - 1.0;
  // Get ratio of specific heats
  bondi.n_adi = 1.0/(bondi.gm - 1.0);

  // Parameters
  bondi.temp_min = 1.0e-7;  // lesser temperature root must be greater than this
  bondi.temp_max = 1.0e0;   // greater temperature root must be less than this

  if (bondi.is_gr) {
    // Get spin of black hole
    bondi.spin = pmbp->pcoord->coord_data.bh_spin;

    // Get excision parameters
    bondi.dexcise = pmbp->pcoord->coord_data.dexcise;
    bondi.pexcise = pmbp->pcoord->coord_data.pexcise;

    bondi.r_crit = pin->GetReal("problem", "r_crit");

    // Prepare various constants for determining primitives
    Real u_crit_sq = 1.0/(2.0*bondi.r_crit);                           // (HSW 71)
    Real u_crit = -sqrt(u_crit_sq);
    Real t_crit = (bondi.n_adi/(bondi.n_adi+1.0)
                  * u_crit_sq/(1.0-(bondi.n_adi+3.0)*u_crit_sq));      // (HSW 74)
    bondi.c1 = pow(t_crit, bondi.n_adi) * u_crit * SQR(bondi.r_crit);  // (HSW 68)
    bondi.c2 = (SQR(1.0 + (bondi.n_adi+1.0) * t_crit)
                * (1.0 - 3.0/(2.0*bondi.r_crit)));                     // (HSW 69)
    bondi.temp_inf = (sqrt(bondi.c2)-1.0)/(1.0+bondi.n_adi);           // (HSW 69)
    bondi.c_s_inf = sqrt(bondi.gm * bondi.temp_inf);
    bondi.r_bondi = 1.0/SQR(bondi.c_s_inf);
  } else {
    bondi.dexcise = pin->GetOrAddReal("coord", "dexcise", 1e-5);
    bondi.pexcise = pin->GetOrAddReal("coord", "pexcise", 1e-10);

    // Prepare various constants for determining primitives
    bondi.temp_inf = pin->GetReal("problem", "temp_inf");
    bondi.c_s_inf = sqrt(bondi.gm * bondi.temp_inf);
    bondi.rho_inf = pow(bondi.temp_inf/bondi.k_adi, bondi.n_adi);
    bondi.pgas_inf = bondi.rho_inf * bondi.temp_inf;
    bondi.r_bondi = 1.0/SQR(bondi.c_s_inf);
    bondi.r_crit = (5.0-3.0*bondi.gm)/4.0;
    bondi.c1 = 0.25*pow(2.0/(5.0-3.0*bondi.gm), (5.0-3.0*bondi.gm)*0.5*bondi.n_adi);
    bondi.c2 = -bondi.n_adi; // useless in Newtonian case
    std::cout << " Bondi radius = " << bondi.r_bondi << std::endl;
    std::cout << " Critical radius = " << bondi.r_crit << std::endl;
    std::cout << " c1 = " << bondi.c1 << std::endl;
  }
  bondi.slope = pin->GetOrAddReal("problem", "slope", 0.0); // slope change of density

  if (restart) return;

  // capture variables for the kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  auto &coord = pmbp->pcoord->coord_data;
  auto bondi_ = bondi;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int n1m1 = n1 - 1, n2m1 = n2 - 1, n3m1 = n3 - 1;
  int is = indcs.is;
  int js = indcs.js;
  int ks = indcs.ks;
  int nmb = pmbp->nmb_thispack;
  auto w0_ = is_mhd ? pmbp->pmhd->w0 : pmbp->phydro->w0;

  // Initialize primitive values (HYDRO ONLY)
  par_for("pgen_bondi", DevExeSpace(), 0,(nmb-1),0,n3m1,0,n2m1,0,n1m1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real rho, pgas, uu1, uu2, uu3;
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    ComputePrimitiveSingle(x1v,x2v,x3v,coord,bondi_,rho,pgas,uu1,uu2,uu3);
    w0_(m,IDN,k,j,i) = rho;
    w0_(m,IEN,k,j,i) = pgas/gm1;
    w0_(m,IM1,k,j,i) = uu1;
    w0_(m,IM2,k,j,i) = uu2;
    w0_(m,IM3,k,j,i) = uu3;
  });

  // Add magnetic field
  if (is_mhd && b_ini>0.0) {
    auto &bcc0_ = pmbp->pmhd->bcc0;
    auto &b0_ = pmbp->pmhd->b0;
    par_for("pgen_bondi_bfield", DevExeSpace(), 0,(nmb-1),0,n3m1,0,n2m1,0,n1m1,
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
  auto &u0_ = is_mhd ? pmbp->pmhd->u0 : pmbp->phydro->u0;
  auto &u1_ = is_mhd ? pmbp->pmhd->u1 : pmbp->phydro->u1;
  if (bondi.reset_ic) {
    if (pmbp->phydro != nullptr) {
      pmbp->phydro->peos->PrimToCons(w0_, u1_, 0, n1m1, 0, n2m1, 0, n3m1);
    } else if (pmbp->pmhd != nullptr) {
      auto &bcc0_ = pmbp->pmhd->bcc0;
      pmbp->pmhd->peos->PrimToCons(w0_, bcc0_, u1_, 0, n1m1, 0, n2m1, 0, n3m1);
    }
    return;
  }

  // Convert primitives to conserved
  if (pmbp->phydro != nullptr) {
    pmbp->phydro->peos->PrimToCons(w0_, u0_, 0, n1m1, 0, n2m1, 0, n3m1);
  } else if (pmbp->pmhd != nullptr) {
    auto &bcc0_ = pmbp->pmhd->bcc0;
    pmbp->pmhd->peos->PrimToCons(w0_, bcc0_, u0_, 0, n1m1, 0, n2m1, 0, n3m1);
  }
  // Add turbulence
  for (auto it = pin->block.begin(); it != pin->block.end(); ++it) {
    if (it->block_name.compare(0, 9, "turb_init") == 0) {
      TurbulenceInit *pturb;
      pturb = new TurbulenceInit(it->block_name,pmbp, pin);
      pturb->InitializeModes(1);
      pturb->AddForcing(1);
      delete pturb;
    }
    if (it->block_name.compare(0, 8, "turb_mhd") == 0) {
      TurbulenceMhd *pturb;
      pturb = new TurbulenceMhd(it->block_name,pmbp, pin);
      pturb->InitializeModes(1);
      pturb->AddForcing(1);
      delete pturb;
    }
  }
  // Convert conserved to primitives
  if (pmbp->phydro != nullptr) {
    pmbp->phydro->peos->ConsToPrim(u0_, w0_, false, 0, n1m1, 0, n2m1, 0, n3m1);
    pmbp->phydro->CopyCons(nullptr,1);
  } else if (pmbp->pmhd != nullptr) {
    auto &bcc0_ = pmbp->pmhd->bcc0;
    auto &b0_ = pmbp->pmhd->b0;
    pmbp->pmhd->peos->ConsToPrim(u0_, b0_, w0_, bcc0_, false, 0, n1m1, 0, n2m1, 0, n3m1);
    pmbp->pmhd->CopyCons(nullptr,1);
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
static void ComputePrimitiveSingle(Real x1v, Real x2v, Real x3v, CoordData coord,
                                   struct bondi_pgen pgen,
                                   Real& rho, Real& pgas,
                                   Real& uu1, Real& uu2, Real& uu3) {
  if (!pgen.is_gr) {
    // Calculate primitives
    Real rad = sqrt(SQR(x1v) + SQR(x2v) + SQR(x3v));
    Real my_rho, my_pgas, my_ur;
    CalculatePrimitivesNewtonian(pgen, rad, &my_rho, &my_pgas, &my_ur);
    if (rad>pgen.r_in) {
      rho = my_rho;
      pgas = my_pgas;
      uu1 = -my_ur * x1v / rad;
      uu2 = -my_ur * x2v / rad;
      uu3 = -my_ur * x3v / rad;
    } else {
      rho = pgen.dexcise;
      pgas = pgen.pexcise;
      uu1 = 0.0;
      uu2 = 0.0;
      uu3 = 0.0;
    }
    return;
  }
  // Calculate Boyer-Lindquist coordinates of cell
  Real r, theta, phi;
  GetBoyerLindquistCoordinates(pgen, x1v, x2v, x3v, &r, &theta, &phi);

  // Compute primitive in BL coordinates, transform to Cartesian KS
  Real my_rho, my_pgas, my_ur;
  CalculatePrimitives(pgen, r, &my_rho, &my_pgas, &my_ur);
  Real u0(0.0), u1(0.0), u2(0.0), u3(0.0);
  TransformVector(pgen, my_ur, 0.0, 0.0, x1v, x2v, x3v, &u1, &u2, &u3);

  Real glower[4][4], gupper[4][4];
  ComputeMetricAndInverse(x1v,x2v,x3v, coord.is_minkowski, coord.bh_spin, glower, gupper);

  Real tmp = glower[1][1]*u1*u1 + 2.0*glower[1][2]*u1*u2 + 2.0*glower[1][3]*u1*u3
           + glower[2][2]*u2*u2 + 2.0*glower[2][3]*u2*u3
           + glower[3][3]*u3*u3;
  Real gammasq = 1.0 + tmp;
  Real b = glower[0][1]*u1 + glower[0][2]*u2 + glower[0][3]*u3;
  u0 = (-b - sqrt(fmax(SQR(b) - glower[0][0]*gammasq, 0.0)))/glower[0][0];

  if (r > 1.0) {
    rho = my_rho;
    pgas = my_pgas;
    if (pgen.slope>0.0) {
      rho = my_rho/pow(1.0+pgen.r_bondi/r,pgen.slope);
      pgas = my_pgas/pow(1.0+pgen.r_bondi/r,pgen.slope);
    }
    uu1 = u1 - gupper[0][1]/gupper[0][0] * u0;
    uu2 = u2 - gupper[0][2]/gupper[0][0] * u0;
    uu3 = u3 - gupper[0][3]/gupper[0][0] * u0;
  } else {
    rho = pgen.dexcise;
    pgas = pgen.pexcise;
    uu1 = 0.0;
    uu2 = 0.0;
    uu3 = 0.0;
  }

  return;
}


//----------------------------------------------------------------------------------------
// Function for returning corresponding Boyer-Lindquist coordinates of point
// Inputs:
//   x1,x2,x3: global coordinates to be converted
// Outputs:
//   pr,ptheta,pphi: variables pointed to set to Boyer-Lindquist coordinates

KOKKOS_INLINE_FUNCTION
static void GetBoyerLindquistCoordinates(struct bondi_pgen pgen,
                                         Real x1, Real x2, Real x3,
                                         Real *pr, Real *ptheta, Real *pphi) {
    Real rad = sqrt(SQR(x1) + SQR(x2) + SQR(x3));
    Real r = fmax((sqrt( SQR(rad) - SQR(pgen.spin) + sqrt(SQR(SQR(rad)-SQR(pgen.spin))
                        + 4.0*SQR(pgen.spin)*SQR(x3)) ) / sqrt(2.0)), 1.0);
    *pr = r;
    *ptheta = acos(x3/r);
    *pphi = atan2(r*x2-pgen.spin*x1, pgen.spin*x2+r*x1) -
            pgen.spin*r/(SQR(r)-2.0*r+SQR(pgen.spin));
  return;
}

//----------------------------------------------------------------------------------------
// Function for transforming 4-vector from Boyer-Lindquist to desired coordinates
// Inputs:
//   a0_bl,a1_bl,a2_bl,a3_bl: upper 4-vector components in Boyer-Lindquist coordinates
//   x1,x2,x3: Cartesian Kerr-Schild coordinates of point
// Outputs:
//   pa0,pa1,pa2,pa3: pointers to upper 4-vector components in desired coordinates
// Notes:
//   Schwarzschild coordinates match Boyer-Lindquist when a = 0

KOKKOS_INLINE_FUNCTION
static void TransformVector(struct bondi_pgen pgen,
                            Real a1_bl, Real a2_bl, Real a3_bl,
                            Real x1, Real x2, Real x3,
                            Real *pa1, Real *pa2, Real *pa3) {
  Real rad = sqrt( SQR(x1) + SQR(x2) + SQR(x3) );
  Real r = fmax((sqrt( SQR(rad) - SQR(pgen.spin) + sqrt(SQR(SQR(rad)-SQR(pgen.spin))
                      + 4.0*SQR(pgen.spin)*SQR(x3)) ) / sqrt(2.0)), 1.0);
  Real delta = SQR(r) - 2.0*r + SQR(pgen.spin);
  *pa1 = a1_bl * ( (r*x1+pgen.spin*x2)/(SQR(r) + SQR(pgen.spin)) - x2*pgen.spin/delta) +
         a2_bl * x1*x3/r * sqrt((SQR(r) + SQR(pgen.spin))/(SQR(x1) + SQR(x2))) -
         a3_bl * x2;
  *pa2 = a1_bl * ( (r*x2-pgen.spin*x1)/(SQR(r) + SQR(pgen.spin)) + x1*pgen.spin/delta) +
         a2_bl * x2*x3/r * sqrt((SQR(r) + SQR(pgen.spin))/(SQR(x1) + SQR(x2))) +
         a3_bl * x1;
  *pa3 = a1_bl * x3/r -
         a2_bl * r * sqrt((SQR(x1) + SQR(x2))/(SQR(r) + SQR(pgen.spin)));
  return;
}

//----------------------------------------------------------------------------------------
// Function for calculating primitives given radius
// Inputs:
//   r: Schwarzschild radius
//   temp_min,temp_max: bounds on temperature
// Outputs:
//   prho: value set to density
//   ppgas: value set to gas pressure
//   put: value set to u^t in Schwarzschild coordinates
//   pur: value set to u^r in Schwarzschild coordinates
// Notes:
//   references Hawley, Smarr, & Wilson 1984, ApJ 277 296 (HSW)

KOKKOS_INLINE_FUNCTION
static void CalculatePrimitives(struct bondi_pgen pgen, Real r,
                                Real *prho, Real *ppgas, Real *pur) {
  // Calculate solution to (HSW 76)
  Real temp_neg_res = TemperatureMin(pgen, r, pgen.temp_min, pgen.temp_max);
  Real temp;
  if (r <= pgen.r_crit) {  // use lesser of two roots
    temp = TemperatureBisect(pgen, r, pgen.temp_min, temp_neg_res);
  } else {  // user greater of two roots
    temp = TemperatureBisect(pgen, r, temp_neg_res, pgen.temp_max);
  }

  // Calculate primitives
  Real rho = pow(temp/pgen.k_adi, pgen.n_adi);             // not same K as HSW
  Real pgas = temp * rho;
  Real ur = pgen.c1 / (SQR(r) * pow(temp, pgen.n_adi));    // (HSW 75)

  // Set primitives
  *prho = rho;
  *ppgas = pgas;
  *pur = ur;
  return;
}

//----------------------------------------------------------------------------------------
// Function for finding temperature at which residual is minimized
// Inputs:
//   r: Schwarzschild radius
//   t_min,t_max: bounds between which minimum must occur
// Outputs:
//   returned value: some temperature for which residual of (HSW 76) is negative
// Notes:
//   references Hawley, Smarr, & Wilson 1984, ApJ 277 296 (HSW)
//   performs golden section search (cf. Numerical Recipes, 3rd ed., 10.2)

KOKKOS_INLINE_FUNCTION
static Real TemperatureMin(struct bondi_pgen pgen, Real r, Real t_min, Real t_max) {
  // Parameters
  const Real ratio = 0.3819660112501051;  // (3+\sqrt{5})/2
  const int max_iterations = 40;          // maximum number of iterations

  // Initialize values
  Real t_mid = t_min + ratio * (t_max - t_min);
  Real res_mid = TemperatureResidual(pgen, t_mid, r);

  // Apply golden section method
  bool larger_to_right = true;  // flag indicating larger subinterval is on right
  for (int n = 0; n < max_iterations; ++n) {
    if (res_mid < 0.0) {
      return t_mid;
    }
    Real t_new;
    if (larger_to_right) {
      t_new = t_mid + ratio * (t_max - t_mid);
      Real res_new = TemperatureResidual(pgen, t_new, r);
      if (res_new < res_mid) {
        t_min = t_mid;
        t_mid = t_new;
        res_mid = res_new;
      } else {
        t_max = t_new;
        larger_to_right = false;
      }
    } else {
      t_new = t_mid - ratio * (t_mid - t_min);
      Real res_new = TemperatureResidual(pgen, t_new, r);
      if (res_new < res_mid) {
        t_max = t_mid;
        t_mid = t_new;
        res_mid = res_new;
      } else {
        t_min = t_new;
        larger_to_right = true;
      }
    }
  }
  return NAN;
}

//----------------------------------------------------------------------------------------
// Bisection root finder
// Inputs:
//   r: Schwarzschild radius
//   t_min,t_max: bounds between which root must occur
// Outputs:
//   returned value: temperature that satisfies (HSW 76)
// Notes:
//   references Hawley, Smarr, & Wilson 1984, ApJ 277 296 (HSW)
//   performs bisection search

KOKKOS_INLINE_FUNCTION
static Real TemperatureBisect(struct bondi_pgen pgen, Real r, Real t_min, Real t_max) {
  // Parameters
  const int max_iterations = 40;
  const Real tol_residual = 1.0e-12;
  const Real tol_temperature = 1.0e-12;

  // Find initial residuals
  Real res_min = TemperatureResidual(pgen, t_min, r);
  Real res_max = TemperatureResidual(pgen, t_max, r);
  if (std::abs(res_min) < tol_residual) {
    return t_min;
  }
  if (std::abs(res_max) < tol_residual) {
    return t_max;
  }
  if ((res_min < 0.0 && res_max < 0.0) || (res_min > 0.0 && res_max > 0.0)) {
    return NAN;
  }

  // Iterate to find root
  Real t_mid;
  for (int i = 0; i < max_iterations; ++i) {
    t_mid = (t_min + t_max) / 2.0;
    if (t_max - t_min < tol_temperature) {
      return t_mid;
    }
    Real res_mid = TemperatureResidual(pgen, t_mid, r);
    if (std::abs(res_mid) < tol_residual) {
      return t_mid;
    }
    if ((res_mid < 0.0 && res_min < 0.0) || (res_mid > 0.0 && res_min > 0.0)) {
      t_min = t_mid;
      res_min = res_mid;
    } else {
      t_max = t_mid;
      res_max = res_mid;
    }
  }
  return t_mid;
}

//----------------------------------------------------------------------------------------
// Function whose value vanishes for correct temperature
// Inputs:
//   t: temperature
//   r: Schwarzschild radius
// Outputs:
//   returned value: residual that should vanish for correct temperature
// Notes:
//   implements (76) from Hawley, Smarr, & Wilson 1984, ApJ 277 296

KOKKOS_INLINE_FUNCTION
static Real TemperatureResidual(struct bondi_pgen pgen, Real t, Real r) {
  if (!pgen.is_gr) {
    Real rho = pow(t/pgen.k_adi, pgen.n_adi);
    Real alpha = rho/pgen.rho_inf;
    Real x = r/pgen.r_bondi;
    Real v = pgen.c1/alpha/x/x;
    return 0.5*v*v+pgen.n_adi*(pgen.k_adi*pow(alpha,pgen.gm-1.0)-1.0)-1.0/x;
  }
  return SQR(1.0 + (pgen.n_adi+1.0) * t)
      * (1.0 - 2.0/r + SQR(pgen.c1)
         / (SQR(SQR(r)) * pow(t, 2.0*pgen.n_adi))) - pgen.c2;
}

KOKKOS_INLINE_FUNCTION
static void CalculatePrimitivesNewtonian(struct bondi_pgen pgen, Real r,
                                         Real *prho, Real *ppgas, Real *pur) {
  Real temp_neg_res = TemperatureMin(pgen, r, pgen.temp_min, pgen.temp_max);
  Real temp;
  if (r <= pgen.r_crit) {  // use lesser of two roots
    temp = TemperatureBisect(pgen, r, pgen.temp_min, temp_neg_res);
  } else {  // user greater of two roots
    temp = TemperatureBisect(pgen, r, temp_neg_res, pgen.temp_max);
  }
  // Calculate primitives
  Real rho = pow(temp/pgen.k_adi, pgen.n_adi);
  Real pgas = temp * rho;
  Real ur = pgen.c1 * pgen.c_s_inf * pgen.rho_inf/rho * SQR(pgen.r_bondi/r);

  // Set primitives
  *prho = rho;
  *ppgas = pgas;
  *pur = ur;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn FixedBondiInflow
//  \brief Sets boundary condition on surfaces of computational domain
// Note quantities at boundaryies are held fixed to initial condition values

void FixedBondiInflow(Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  auto &coord = pmbp->pcoord->coord_data;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int n1m1 = n1 - 1, n2m1 = n2 - 1, n3m1 = n3 - 1;
  int &is = indcs.is;  int &ie  = indcs.ie;
  int &js = indcs.js;  int &je  = indcs.je;
  int &ks = indcs.ks;  int &ke  = indcs.ke;
  auto &mb_bcs = pmbp->pmb->mb_bcs;
  int nmb = pmbp->nmb_thispack;

  auto bondi_ = bondi;
  bool is_mhd = (pmbp->pmhd != nullptr);
  auto u0_ = is_mhd ? pmbp->pmhd->u0 : pmbp->phydro->u0;
  auto w0_ = is_mhd ? pmbp->pmhd->w0 : pmbp->phydro->w0;

  if (!bondi.is_gr) {
    int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
    Real gm1 = bondi.gm - 1.0;
    Real rin = bondi.r_in, dexcise = bondi.dexcise, pexcise = bondi.pexcise;
    par_for("fixed_rin", DevExeSpace(),0,nmb-1,ks-ng,ke+ng,js-ng,je+ng,is-ng,ie+ng,
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

      if (rad < rin) {
        u0_(m,IDN,k,j,i) = dexcise;
        u0_(m,IEN,k,j,i) = pexcise/gm1;
        u0_(m,IM1,k,j,i) = 0.0;
        u0_(m,IM2,k,j,i) = 0.0;
        u0_(m,IM3,k,j,i) = 0.0;
      }
    });
  }

  // B-field boundary conditions
  if (is_mhd) {
    auto &b0 = pm->pmb_pack->pmhd->b0;
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
  par_for("fixed_x1", DevExeSpace(),0,(nmb-1),0,n3m1,0,n2m1,0,(ng-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // inner x1 boundary
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    Real rho, pgas, uu1, uu2, uu3;
    if (mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::user) {
      ComputePrimitiveSingle(x1v,x2v,x3v,coord,bondi_,rho,pgas,uu1,uu2,uu3);
      w0_(m,IDN,k,j,i) = rho;
      w0_(m,IEN,k,j,i) = pgas/(bondi_.gm - 1.0);
      w0_(m,IM1,k,j,i) = uu1;
      w0_(m,IM2,k,j,i) = uu2;
      w0_(m,IM3,k,j,i) = uu3;
    }

    // outer x1 boundary
    x1v = CellCenterX((ie+i+1)-is, indcs.nx1, x1min, x1max);

    if (mb_bcs.d_view(m,BoundaryFace::outer_x1) == BoundaryFlag::user) {
      ComputePrimitiveSingle(x1v,x2v,x3v,coord,bondi_, rho,pgas,uu1,uu2,uu3);
      w0_(m,IDN,k,j,(ie+i+1)) = rho;
      w0_(m,IEN,k,j,(ie+i+1)) = pgas/(bondi_.gm - 1.0);
      w0_(m,IM1,k,j,(ie+i+1)) = uu1;
      w0_(m,IM2,k,j,(ie+i+1)) = uu2;
      w0_(m,IM3,k,j,(ie+i+1)) = uu3;
    }
  });
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
  par_for("fixed_x2", DevExeSpace(),0,(nmb-1),0,n3m1,0,(ng-1),0,n1m1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // inner x2 boundary
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    Real rho, pgas, uu1, uu2, uu3;
    if (mb_bcs.d_view(m,BoundaryFace::inner_x2) == BoundaryFlag::user) {
      ComputePrimitiveSingle(x1v,x2v,x3v,coord,bondi_,rho,pgas,uu1,uu2,uu3);
      w0_(m,IDN,k,j,i) = rho;
      w0_(m,IEN,k,j,i) = pgas/(bondi_.gm - 1.0);
      w0_(m,IM1,k,j,i) = uu1;
      w0_(m,IM2,k,j,i) = uu2;
      w0_(m,IM3,k,j,i) = uu3;
    }

    // outer x2 boundary
    x2v = CellCenterX((je+j+1)-js, indcs.nx2, x2min, x2max);

    if (mb_bcs.d_view(m,BoundaryFace::outer_x2) == BoundaryFlag::user) {
      ComputePrimitiveSingle(x1v,x2v,x3v,coord,bondi_,rho,pgas,uu1,uu2,uu3);
      w0_(m,IDN,k,(je+j+1),i) = rho;
      w0_(m,IEN,k,(je+j+1),i) = pgas/(bondi_.gm - 1.0);
      w0_(m,IM1,k,(je+j+1),i) = uu1;
      w0_(m,IM2,k,(je+j+1),i) = uu2;
      w0_(m,IM3,k,(je+j+1),i) = uu3;
    }
  });
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

    Real rho, pgas, uu1, uu2, uu3;
    if (mb_bcs.d_view(m,BoundaryFace::inner_x3) == BoundaryFlag::user) {
      ComputePrimitiveSingle(x1v,x2v,x3v,coord,bondi_,rho,pgas,uu1,uu2,uu3);
      w0_(m,IDN,k,j,i) = rho;
      w0_(m,IEN,k,j,i) = pgas/(bondi_.gm - 1.0);
      w0_(m,IM1,k,j,i) = uu1;
      w0_(m,IM2,k,j,i) = uu2;
      w0_(m,IM3,k,j,i) = uu3;
    }

    // outer x3 boundary
    x3v = CellCenterX((ke+k+1)-ks, indcs.nx3, x3min, x3max);

    if (mb_bcs.d_view(m,BoundaryFace::outer_x3) == BoundaryFlag::user) {
      ComputePrimitiveSingle(x1v,x2v,x3v,coord,bondi_,rho,pgas,uu1,uu2,uu3);
      w0_(m,IDN,(ke+k+1),j,i) = rho;
      w0_(m,IEN,(ke+k+1),j,i) = pgas/(bondi_.gm - 1.0);
      w0_(m,IM1,(ke+k+1),j,i) = uu1;
      w0_(m,IM2,(ke+k+1),j,i) = uu2;
      w0_(m,IM3,(ke+k+1),j,i) = uu3;
    }
  });
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
  DvceArray5D<Real> &u0 = (pmbp->pmhd != nullptr) ? pmbp->pmhd->u0 : pmbp->phydro->u0;
  DvceArray5D<Real> &w0 = (pmbp->pmhd != nullptr) ? pmbp->pmhd->w0 : pmbp->phydro->w0;
  if (!bondi.is_gr) {
    //std::cout << "AddAccel" << std::endl;
    auto &indcs = pm->mb_indcs;
    auto &size = pmbp->pmb->mb_size;
    int is = indcs.is, ie = indcs.ie, nx1 = indcs.nx1;
    int js = indcs.js, je = indcs.je, nx2 = indcs.nx2;
    int ks = indcs.ks, ke = indcs.ke, nx3 = indcs.nx3;
    int nmb1 = pmbp->nmb_thispack - 1;
    Real rin = bondi.r_in;
    par_for("accel", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
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

      Real accel = -1.0/rad/rad;
      if (rad < rin) {
        // Potential = -GM/(r^2+r_a^2*exp(-r^2/r_a^2))^0.5
        Real r_a = 0.5*rin;
        Real fac = exp(-rad*rad/r_a/r_a);
        accel *= rad*rad*rad*(1-fac)/pow(rad*rad+r_a*r_a*fac,1.5);
      }
      Real dmomr = bdt*w0(m,IDN,k,j,i)*accel;
      Real dmomx1 = dmomr*x1v/rad;
      Real dmomx2 = dmomr*x2v/rad;
      Real dmomx3 = dmomr*x3v/rad;
      Real denergy = bdt*w0(m,IDN,k,j,i)*accel/rad*
                    (w0(m,IVX,k,j,i)*x1v+w0(m,IVY,k,j,i)*x2v+w0(m,IVZ,k,j,i)*x3v);

      u0(m,IM1,k,j,i) += dmomx1;
      u0(m,IM2,k,j,i) += dmomx2;
      u0(m,IM3,k,j,i) += dmomx3;
      u0(m,IEN,k,j,i) += denergy;
    });
  }
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::BondiErrors()
//  \brief Computes errors in linear wave solution and outputs to file.

void BondiErrors(ParameterInput *pin, Mesh *pm) {
  // calculate reference solution by calling pgen again.  Solution stored in second
  // register u1/b1 when flag is false.
  bondi.reset_ic=true;
  pm->pgen->UserProblem(pin, false);

  Real l1_err[8];
  int nvars=0;

  // capture class variables for kernel
  auto &indcs = pm->mb_indcs;
  int &nx1 = indcs.nx1;
  int &nx2 = indcs.nx2;
  int &nx3 = indcs.nx3;
  int &is = indcs.is;
  int &js = indcs.js;
  int &ks = indcs.ks;
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &size = pmbp->pmb->mb_size;

  // TODO(@mhguo): should we add for mhd?
  // compute errors for Hydro  -----------------------------------------------------------
  if (pmbp->phydro != nullptr) {
    nvars = pmbp->phydro->nhydro;

    EOS_Data &eos = pmbp->phydro->peos->eos_data;
    auto &u0_ = pmbp->phydro->u0;
    auto &u1_ = pmbp->phydro->u1;

    const int nmkji = (pmbp->nmb_thispack)*nx3*nx2*nx1;
    const int nkji = nx3*nx2*nx1;
    const int nji  = nx2*nx1;
    array_sum::GlobalSum sum_this_mb;
    Kokkos::parallel_reduce("Bondi-err-Sums",
                            Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, array_sum::GlobalSum &mb_sum) {
      // compute n,k,j,i indices of thread
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks;
      j += js;

      Real vol = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;

      // Hydro conserved variables:
      array_sum::GlobalSum evars;
      evars.the_array[IDN] = vol*fabs(u0_(m,IDN,k,j,i) - u1_(m,IDN,k,j,i));
      evars.the_array[IM1] = vol*fabs(u0_(m,IM1,k,j,i) - u1_(m,IM1,k,j,i));
      evars.the_array[IM2] = vol*fabs(u0_(m,IM2,k,j,i) - u1_(m,IM2,k,j,i));
      evars.the_array[IM3] = vol*fabs(u0_(m,IM3,k,j,i) - u1_(m,IM3,k,j,i));
      if (eos.is_ideal) {
        evars.the_array[IEN] = vol*fabs(u0_(m,IEN,k,j,i) - u1_(m,IEN,k,j,i));
      }

      // fill rest of the_array with zeros, if narray < NREDUCTION_VARIABLES
      for (int n=nvars; n<NREDUCTION_VARIABLES; ++n) {
        evars.the_array[n] = 0.0;
      }

      // sum into parallel reduce
      mb_sum += evars;
    }, Kokkos::Sum<array_sum::GlobalSum>(sum_this_mb));

    // store data into l1_err array
    for (int n=0; n<nvars; ++n) {
      l1_err[n] = sum_this_mb.the_array[n];
    }
  }

#if MPI_PARALLEL_ENABLED
  // sum over all ranks
  if (global_variable::my_rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, l1_err, 8, MPI_DOUBLE, MPI_SUM, 0,
               MPI_COMM_WORLD);
  } else {
    MPI_Reduce(l1_err, l1_err, 8, MPI_DOUBLE, MPI_SUM, 0,
               MPI_COMM_WORLD);
  }
#endif

  // normalize errors by number of cells
  Real vol=  (pmbp->pmesh->mesh_size.x1max - pmbp->pmesh->mesh_size.x1min)
            *(pmbp->pmesh->mesh_size.x2max - pmbp->pmesh->mesh_size.x2min)
            *(pmbp->pmesh->mesh_size.x3max - pmbp->pmesh->mesh_size.x3min);
  for (int i=0; i<nvars; ++i) l1_err[i] = l1_err[i]/vol;

  // compute rms error
  Real rms_err = 0.0;
  for (int i=0; i<nvars; ++i) {
    rms_err += SQR(l1_err[i]);
  }
  rms_err = std::sqrt(rms_err);

  // open output file and write out errors
  if (global_variable::my_rank==0) {
    // open output file and write out errors
    std::string fname;
    fname.assign(pin->GetString("job","basename"));
    fname.append("-errs.dat");
    FILE *pfile;

    // The file exists -- reopen the file in append mode
    if ((pfile = std::fopen(fname.c_str(), "r")) != nullptr) {
      if ((pfile = std::freopen(fname.c_str(), "a", pfile)) == nullptr) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Error output file could not be opened" <<std::endl;
        std::exit(EXIT_FAILURE);
      }

    // The file es not exist -- open the file in write mode and add headers
    } else {
      if ((pfile = std::fopen(fname.c_str(), "w")) == nullptr) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Error output file could not be opened" <<std::endl;
        std::exit(EXIT_FAILURE);
      }
      std::fprintf(pfile, "# Nx1  Nx2  Nx3   Ncycle  RMS-L1-err       ");
      if (pmbp->phydro != nullptr) {
        std::fprintf(pfile, "d_L1         M1_L1         M2_L1");
        std::fprintf(pfile, "         M3_L1         E_L1 ");
      }
      std::fprintf(pfile, "\n");
    }

    // write errors
    std::fprintf(pfile, "%04d", pmbp->pmesh->mesh_indcs.nx1);
    std::fprintf(pfile, "  %04d", pmbp->pmesh->mesh_indcs.nx2);
    std::fprintf(pfile, "  %04d", pmbp->pmesh->mesh_indcs.nx3);
    std::fprintf(pfile, "  %05d  %e", pmbp->pmesh->ncycle, rms_err);
    for (int i=0; i<nvars; ++i) {
      std::fprintf(pfile, "  %e", l1_err[i]);
    }
    std::fprintf(pfile, "\n");
    std::fclose(pfile);
  }
  return;
}

//----------------------------------------------------------------------------------------
// Function for computing accretion fluxes through constant spherical KS radius surfaces

void BondiFluxes(HistoryData *pdata, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  // set nvars, adiabatic index, primitive array w0, and field array bcc0 if is_mhd
  int nvars; Real gamma; bool is_mhd = false;
  DvceArray5D<Real> w0_, bcc0_;
  if (pmbp->phydro != nullptr) {
    nvars = pmbp->phydro->nhydro + pmbp->phydro->nscalars;
    gamma = pmbp->phydro->peos->eos_data.gamma;
    w0_ = pmbp->phydro->w0;
  } else if (pmbp->pmhd != nullptr) {
    is_mhd = true;
    nvars = pmbp->pmhd->nmhd + pmbp->pmhd->nscalars;
    gamma = pmbp->pmhd->peos->eos_data.gamma;
    w0_ = pmbp->pmhd->w0;
    bcc0_ = pmbp->pmhd->bcc0;
  }

  // extract grids, number of radii, number of fluxes, and history appending index
  auto &grids = pm->pgen->spherical_grids;
  int nradii = grids.size();
  //int nflux = (is_mhd) ? 4 : 3;
  const int nflux = 15;
  // set number of and names of history variables for hydro or mhd
  //  (0) mass
  //  (1) mass accretion rate
  //  (2) energy flux
  //  (3) angular momentum flux * 3
  //  (4) magnetic flux (iff MHD)
  pdata->nhist = nradii * nflux;
  if (pdata->nhist > NHISTORY_VARIABLES) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "User history function specified pdata->nhist larger than"
              << " NHISTORY_VARIABLES" << std::endl;
    exit(EXIT_FAILURE);
  }
  for (int g=0; g<nradii; ++g) {
    std::string rstr = std::to_string(g);
    pdata->label[nflux*g+0] = "r_" + rstr;
    pdata->label[nflux*g+1] = "m_" + rstr;
    pdata->label[nflux*g+2] = "mdot" + rstr;
    pdata->label[nflux*g+3] = "mdout" + rstr;
    pdata->label[nflux*g+4] = "mdh" + rstr;
    pdata->label[nflux*g+5] = "edot" + rstr;
    pdata->label[nflux*g+6] = "edout" + rstr;
    pdata->label[nflux*g+7] = "lx" + rstr;
    pdata->label[nflux*g+8] = "ly" + rstr;
    pdata->label[nflux*g+9] = "lz" + rstr;
    pdata->label[nflux*g+10] = "phi" + rstr;
    pdata->label[nflux*g+11] = "edot_hyd" + rstr;
    pdata->label[nflux*g+12] = "edotout_hyd" + rstr;
    pdata->label[nflux*g+13] = "edot_adv" + rstr;
    pdata->label[nflux*g+14] = "edotout_adv" + rstr;
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
    Real r = grids[g]->radius;
    pdata->hdata[nflux*g+0] = r;
    if (is_gr) {
      // extract BH parameters
      bool &flat = pmbp->pcoord->coord_data.is_minkowski;
      Real &spin = pmbp->pcoord->coord_data.bh_spin;
      for (int n=0; n<grids[g]->nangles; ++n) {
        // extract coordinate data at this angle
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
        Real &int_ie = grids[g]->interp_vals.h_view(n,IEN);

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

        Real is_out = (ur>0.0)? 1.0 : 0.0;
        Real int_temp = (gamma-1.0)*int_ie/int_dn;
        Real is_hot = (int_temp>0.1/r)? 1.0 : 0.0;

        // compute mass density
        pdata->hdata[nflux*g+1] += int_dn*sqrtmdet*domega;

        // compute mass flux
        pdata->hdata[nflux*g+2] += 1.0*int_dn*ur*sqrtmdet*domega;
        pdata->hdata[nflux*g+3] += is_out*int_dn*ur*sqrtmdet*domega;
        pdata->hdata[nflux*g+4] += is_hot*int_dn*ur*sqrtmdet*domega;

        // compute energy flux
        Real t1_0 = (int_dn + gamma*int_ie + b_sq)*ur*u_0 - br*b_0;
        pdata->hdata[nflux*g+5] += 1.0*t1_0*sqrtmdet*domega;
        pdata->hdata[nflux*g+6] += is_out*t1_0*sqrtmdet*domega;

        // compute angular momentum flux
        // TODO(@mhguo): write a correct function to compute x,y angular momentum flux
        Real t1_1 = 0.0;
        Real t1_2 = 0.0;
        Real t1_3 = (int_dn + gamma*int_ie + b_sq)*ur*u_ph - br*b_ph;
        pdata->hdata[nflux*g+7] += t1_1*sqrtmdet*domega;
        pdata->hdata[nflux*g+8] += t1_2*sqrtmdet*domega;
        pdata->hdata[nflux*g+9] += t1_3*sqrtmdet*domega;

        // compute magnetic flux
        if (is_mhd) {
          pdata->hdata[nflux*g+10] += 0.5*fabs(br*u0 - b0*ur)*sqrtmdet*domega;
        }

        Real t1_0_hyd = (int_dn + gamma*int_ie)*ur*u_0;
        Real bernl_hyd = -(1.0 + gamma*int_ie/int_dn)*u_0-1.0;
        pdata->hdata[nflux*g+11] += 1.0*t1_0_hyd*sqrtmdet*domega;
        pdata->hdata[nflux*g+12] += is_out*t1_0_hyd*sqrtmdet*domega;
        pdata->hdata[nflux*g+13] += 1.0*bernl_hyd*sqrtmdet*domega;
        pdata->hdata[nflux*g+14] += is_out*bernl_hyd*sqrtmdet*domega;
      }
    } else {
      for (int n=0; n<grids[g]->nangles; ++n) {
        // extract coordinate data at this angle
        Real x1 = grids[g]->interp_coord.h_view(n,0);
        Real x2 = grids[g]->interp_coord.h_view(n,1);
        Real x3 = grids[g]->interp_coord.h_view(n,2);
        // extract interpolated primitives
        Real &int_dn = grids[g]->interp_vals.h_view(n,IDN);
        Real &int_vx = grids[g]->interp_vals.h_view(n,IVX);
        Real &int_vy = grids[g]->interp_vals.h_view(n,IVY);
        Real &int_vz = grids[g]->interp_vals.h_view(n,IVZ);
        Real &int_ie = grids[g]->interp_vals.h_view(n,IEN);
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
        Real int_temp = (gamma-1.0)*int_ie/int_dn;
        Real is_hot = (int_temp>0.1/r)? 1.0 : 0.0;

        // compute mass density
        pdata->hdata[nflux*g+1] += int_dn*r_sq*domega;

        // compute mass flux
        pdata->hdata[nflux*g+2] += 1.0*int_dn*vr*r_sq*domega;
        pdata->hdata[nflux*g+3] += is_out*int_dn*vr*r_sq*domega;
        pdata->hdata[nflux*g+4] += is_hot*int_dn*vr*r_sq*domega;

        // compute energy flux
        // TODO(@mhguo): this should be correct now but need to check more carefully
        // total enthalpy
        Real t1_0 = (gamma*int_ie + e_k + 0.5*b_sq)*vr;
        pdata->hdata[nflux*g+5] += 1.0*t1_0*r_sq*domega;
        pdata->hdata[nflux*g+6] += is_out*t1_0*r_sq*domega;

        // compute angular momentum flux
        // TODO(@mhguo): check whether this is correct!
        pdata->hdata[nflux*g+7] += int_dn*(x2*v3-x3*v2)*r_sq*domega;
        pdata->hdata[nflux*g+8] += int_dn*(x3*v1-x1*v3)*r_sq*domega;
        pdata->hdata[nflux*g+9] += int_dn*(x1*v2-x2*v1)*r_sq*domega;

        // compute magnetic flux
        if (is_mhd) {
          pdata->hdata[nflux*g+10] += 0.5*fabs(br)*r_sq*domega;
        }

        // TODO(@mhguo): this needs to be checked, assuming GM=1, adding potential term?
        Real t1_0_hyd = (gamma*int_ie + e_k)*vr;
        Real bernl_hyd = (gamma*int_ie + e_k)/int_dn - 1.0/r;
        pdata->hdata[nflux*g+11] += 1.0*t1_0_hyd*r_sq*domega;
        pdata->hdata[nflux*g+12] += is_out*t1_0_hyd*r_sq*domega;
        pdata->hdata[nflux*g+13] += 1.0*bernl_hyd*r_sq*domega;
        pdata->hdata[nflux*g+14] += is_out*bernl_hyd*r_sq*domega;
      }
    }
  }
  // fill rest of the_array with zeros, if nhist < NHISTORY_VARIABLES
  for (int n=pdata->nhist; n<NHISTORY_VARIABLES; ++n) {
    pdata->hdata[n] = 0.0;
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn Real BondiTimeStep()
//! \brief User-defined time step function

Real BondiTimeStep(Mesh *pm) {
  if (bondi.ndiag>0 && pm->ncycle % bondi.ndiag == 0) {
    Diagnostic(pm);
  }
  Real dt = pm->dt/pm->cfl_no;
  if (!bondi.multi_zone && !bondi.fixed_zone && !bondi.is_amr) {
    return dt;
  }

  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  int is = indcs.is, nx1 = indcs.nx1;
  int js = indcs.js, nx2 = indcs.nx2;
  int ks = indcs.ks, nx3 = indcs.nx3;

  Real dt1 = std::numeric_limits<float>::max();
  Real dt2 = std::numeric_limits<float>::max();
  Real dt3 = std::numeric_limits<float>::max();

  // capture class variables for kernel
  bool is_mhd = (pmbp->pmhd != nullptr);
  auto &w0_ = (is_mhd)? pmbp->pmhd->w0 : pmbp->phydro->w0;
  auto &eos = (is_mhd)? pmbp->pmhd->peos->eos_data : pmbp->phydro->peos->eos_data;
  auto &size = pmbp->pmb->mb_size;

  auto &is_gr = pmbp->pcoord->is_general_relativistic;
  const int nmkji = (pmbp->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;

  Real r_v = 0.0;
  if (bondi.multi_zone) {
    if (bondi.mz_type<2) {
      r_v = VcycleRadius(bondi, pm->ncycle, bondi.vcycle_n, bondi.rv_in, bondi.rv_out);
    } else {
      r_v = MultiZoneRadius(pm->time);
    }
    if (pm->ncycle%10 == 0) {
      std::cout << "Vcycle radius = " << r_v << std::endl;
    }
    pmbp->pcoord->zone_r = r_v;
    pmbp->pcoord->SetZoneMasks(pmbp->pcoord->zone_mask, r_v,
                               std::numeric_limits<Real>::max());
  }

  Real r_f = 0.0;
  if (bondi.fixed_zone) {
    int ncycle = pm->ncycle;
    int nc_amr = bondi.ncycle_amr;
    int old_level = AMRLevel(ncycle-nc_amr, nc_amr, bondi.beg_level, bondi.end_level);
    int new_level = AMRLevel(ncycle, nc_amr, bondi.beg_level, bondi.end_level);
    Real dx = (pm->mesh_size.x1max - pm->mesh_size.x1min)/pm->mesh_indcs.nx1
              /pow(2.0, new_level);
    r_f = (new_level <= old_level) ? 2.0*dx : 0.0;
    pmbp->pcoord->SetZoneMasks(pmbp->pcoord->zone_mask, r_f,
                               std::numeric_limits<Real>::max());
  }

  if (is_gr && bondi.is_amr && pmbp->pcoord->coord_data.bh_excise) {
    pmbp->pcoord->SetExcisionMasks(pmbp->pcoord->excision_floor,
                                   pmbp->pcoord->excision_flux);
  }

  if (is_mhd) {
    // find smallest dx/(v +/- Cf) in each direction for mhd problems
    auto &bcc0_ = pmbp->pmhd->bcc0;
    Kokkos::parallel_reduce("AccMHDNudt",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, Real &min_dt1, Real &min_dt2, Real &min_dt3) {
      // compute m,k,j,i indices of thread and call function
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks;
      j += js;
      Real max_dv1 = 0.0, max_dv2 = 0.0, max_dv3 = 0.0;

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

      if (is_gr && r_v <= 1.0 && rad <=1.0) {
        max_dv1 = 1.0;
        max_dv2 = 1.0;
        max_dv3 = 1.0;
      } else if (rad >= 0.88*r_v) {
        Real &w_d = w0_(m,IDN,k,j,i);
        Real &w_bx = bcc0_(m,IBX,k,j,i);
        Real &w_by = bcc0_(m,IBY,k,j,i);
        Real &w_bz = bcc0_(m,IBZ,k,j,i);
        Real cf;
        Real p = eos.IdealGasPressure(w0_(m,IEN,k,j,i));
        if (eos.is_ideal) {
          cf = eos.IdealMHDFastSpeed(w_d, p, w_bx, w_by, w_bz);
        } else {
          cf = eos.IdealMHDFastSpeed(w_d, w_bx, w_by, w_bz);
        }
        max_dv1 = fabs(w0_(m,IVX,k,j,i)) + cf;

        if (eos.is_ideal) {
          cf = eos.IdealMHDFastSpeed(w_d, p, w_by, w_bz, w_bx);
        } else {
          cf = eos.IdealMHDFastSpeed(w_d, w_by, w_bz, w_bx);
        }
        max_dv2 = fabs(w0_(m,IVY,k,j,i)) + cf;

        if (eos.is_ideal) {
          cf = eos.IdealMHDFastSpeed(w_d, p, w_bz, w_bx, w_by);
        } else {
          cf = eos.IdealMHDFastSpeed(w_d, w_bz, w_bx, w_by);
        }
        max_dv3 = fabs(w0_(m,IVZ,k,j,i)) + cf;
      }
      if (is_gr) {
        max_dv1 = fmin(max_dv1, 1.0);
        max_dv2 = fmin(max_dv2, 1.0);
        max_dv3 = fmin(max_dv3, 1.0);
      }

      min_dt1 = fmin((size.d_view(m).dx1/max_dv1), min_dt1);
      min_dt2 = fmin((size.d_view(m).dx2/max_dv2), min_dt2);
      min_dt3 = fmin((size.d_view(m).dx3/max_dv3), min_dt3);
    }, Kokkos::Min<Real>(dt1), Kokkos::Min<Real>(dt2),Kokkos::Min<Real>(dt3));
  } else {
    // find smallest dx/(v +/- Cs) in each direction for hydrodynamic problems
    Kokkos::parallel_reduce("AccHydroNudt",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, Real &min_dt1, Real &min_dt2, Real &min_dt3) {
      // compute m,k,j,i indices of thread and call function
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks;
      j += js;

      Real max_dv1 = 0.0, max_dv2 = 0.0, max_dv3 = 0.0;

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

      if (is_gr && r_v <= 1.0 && rad <=1.0) {
        max_dv1 = 1.0;
        max_dv2 = 1.0;
        max_dv3 = 1.0;
      } else if (rad >= 0.88*r_v) {
        Real cs;
        if (eos.is_ideal) {
          Real p = eos.IdealGasPressure(w0_(m,IEN,k,j,i));
          cs = eos.IdealHydroSoundSpeed(w0_(m,IDN,k,j,i), p);
        } else {
          cs = eos.iso_cs;
        }
        max_dv1 = fabs(w0_(m,IVX,k,j,i)) + cs;
        max_dv2 = fabs(w0_(m,IVY,k,j,i)) + cs;
        max_dv3 = fabs(w0_(m,IVZ,k,j,i)) + cs;
      }
      if (is_gr) {
        max_dv1 = fmin(max_dv1, 1.0);
        max_dv2 = fmin(max_dv2, 1.0);
        max_dv3 = fmin(max_dv3, 1.0);
      }

      min_dt1 = fmin((size.d_view(m).dx1/max_dv1), min_dt1);
      min_dt2 = fmin((size.d_view(m).dx2/max_dv2), min_dt2);
      min_dt3 = fmin((size.d_view(m).dx3/max_dv3), min_dt3);
    }, Kokkos::Min<Real>(dt1), Kokkos::Min<Real>(dt2),Kokkos::Min<Real>(dt3));
  }

  // compute minimum of dt1/dt2/dt3 for 1D/2D/3D problems
  Real dtnew = dt1;
  if (pm->multi_d) { dtnew = std::min(dtnew, dt2); }
  if (pm->three_d) { dtnew = std::min(dtnew, dt3); }

  return dtnew;
}

//----------------------------------------------------------------------------------------
//! \fn BondiRefine
//! \brief User-defined refinement condition(s)

// TODO(@mhguo): current AMR method still not working for horizon, need to fix it
void BondiRefine(MeshBlockPack* pmbp) {
  // capture variables for kernels
  Mesh *pm = pmbp->pmesh;
  auto &size = pmbp->pmb->mb_size;

  // check (on device) Hydro/MHD refinement conditions over all MeshBlocks
  auto refine_flag_ = pm->pmr->refine_flag;
  int nmb = pmbp->nmb_thispack;
  int mbs = pm->gids_eachrank[global_variable::my_rank];

  if (bondi.ncycle_amr>0 && pm->ncycle % bondi.ncycle_amr == 0) {
    int root_level = pm->root_level;
    int ncycle=pm->ncycle, ncycle_amr=bondi.ncycle_amr;
    int old_level = AMRLevel(ncycle-1, ncycle_amr, bondi.beg_level, bondi.end_level);
    int new_level = AMRLevel(ncycle, ncycle_amr, bondi.beg_level, bondi.end_level);
    Real &rad_thresh  = bondi.r_refine;
    DualArray1D<int> levels_thisrank("levels_thisrank", nmb);
    if (global_variable::my_rank == 0) {
      std::cout << "BondiRefine: ncycle= " << ncycle << " old_level= " << old_level
                << " new_level= " << new_level << std::endl;
    }
    for (int m=0; m<nmb; ++m) {
      levels_thisrank.h_view(m) = pm->lloc_eachmb[m+mbs].level;
    }
    levels_thisrank.template modify<HostMemSpace>();
    levels_thisrank.template sync<DevExeSpace>();
    par_for_outer("BondiRefineLevel",DevExeSpace(), 0, 0, 0, (nmb-1),
    KOKKOS_LAMBDA(TeamMember_t tmember, const int m) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      Real ax1min = x1min*x1max>0.0? fmin(fabs(x1min), fabs(x1max)) : 0.0;
      Real ax2min = x2min*x2max>0.0? fmin(fabs(x2min), fabs(x2max)) : 0.0;
      Real ax3min = x3min*x3max>0.0? fmin(fabs(x3min), fabs(x3max)) : 0.0;
      Real rad_min = sqrt(SQR(ax1min)+SQR(ax2min)+SQR(ax3min));
      if (levels_thisrank.d_view(m+mbs) == old_level+root_level) {
        if (new_level > old_level) {
          if (rad_min < rad_thresh) {
            refine_flag_.d_view(m+mbs) = 1;
          }
        } else if (new_level < old_level) {
          refine_flag_.d_view(m+mbs) = -1;
        }
      }
    });
  }
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
