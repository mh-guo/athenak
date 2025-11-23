//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file zoom_gr_acc.cpp
//! \brief (@mhguo) Problem generator for black hole accretion from galactic scale

#include <cmath>   // abs(), NAN, pow(), sqrt()
#include <cstring> // strcmp()
#include <iostream>
#include <sstream>
#include <string> // string
#include <cstdio> // fclose

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
#include "pgen/zoom.hpp"

#include "eos/ideal_c2p_hyd.hpp"
#include "eos/ideal_c2p_mhd.hpp"
#include "srcterms/ismcooling.hpp"
#include "units/units.hpp"

#include <Kokkos_Random.hpp>

namespace {

KOKKOS_INLINE_FUNCTION
static void ComputePrimitiveSingle(Real x1v, Real x2v, Real x3v, CoordData coord,
                                   struct acc_pgen pgen,
                                   Real& rho, Real& pgas,
                                   Real& uu1, Real& uu2, Real& uu3);

KOKKOS_INLINE_FUNCTION
static void GetBoyerLindquistCoordinates(struct acc_pgen pgen,
                                         Real x1, Real x2, Real x3,
                                         Real *pr, Real *ptheta, Real *pphi);

KOKKOS_INLINE_FUNCTION
static void TransformVector(struct acc_pgen pgen,
                            Real a1_bl, Real a2_bl, Real a3_bl,
                            Real x1, Real x2, Real x3,
                            Real *pa1, Real *pa2, Real *pa3);

KOKKOS_INLINE_FUNCTION
static void CalculatePrimitives(struct acc_pgen pgen, Real r,
                                Real *prho, Real *ppgas, Real *pur);

KOKKOS_INLINE_FUNCTION
static Real TemperatureMin(struct acc_pgen pgen, Real r, Real t_min, Real t_max);

KOKKOS_INLINE_FUNCTION
static Real TemperatureBisect(struct acc_pgen pgen, Real r, Real t_min, Real t_max);

KOKKOS_INLINE_FUNCTION
static Real TemperatureResidual(struct acc_pgen pgen, Real t, Real r);

KOKKOS_INLINE_FUNCTION
static Real Acceleration(struct acc_pgen pgen, const Real r);

struct acc_pgen {
  Real spin;                // black hole spin
  Real dexcise, pexcise;    // excision parameters
  int  ic_type;             // initial condition type
  Real n_adi, k_adi, gm;    // hydro EOS parameters
  Real r_crit;              // sonic point radius
  Real c1, c2;              // useful constants
  Real temp_min, temp_max;  // bounds for temperature root find
  Real temp_norm, c_s_norm; // normalization of temperature and sound speed
  Real rho_norm, pgas_norm; // normalization of density and pressure
  Real r_bondi;             // Bondi radius 2GM/c_s^2
  bool reset_ic = false;    // reset initial conditions after run
  int  bc_type = 0;         // boundary condition type
  Real rb_out;              // outer boundary radius
  bool potential = false;   // add potential term
  Real sigma2 = 0.0;        // potential term coefficient
  Real r_iso = 0.0;         // radius of isothermal core
  bool cooling = false;     // add ISM cooling
  Real mu_h = 0.6;          // mean molecular weight of hydrogen
  Real tau_cool = -1.0;     // cooling time in dynamical time
  Real r_entropy;              // radius of entropy core
  int  ndiag = -1;          // diagnostic interval
};

  acc_pgen acc;

struct acc_arr {
  DualArray1D<Real> temp_arr;
};

acc_arr* accarr = new acc_arr();

// prototypes for user-defined BCs and error functions
void FixedBondiInflow(Mesh *pm);
void BondiErrors(ParameterInput *pin, Mesh *pm);
void BondiFluxes(HistoryData *pdata, Mesh *pm);
void AddUserSrcs(Mesh *pm, const Real bdt);
void AddAccel(Mesh *pm, const Real bdt, DvceArray5D<Real> &u0,
              const DvceArray5D<Real> &w0);
void AddISMCooling(Mesh *pm, const Real bdt, DvceArray5D<Real> &u0,
                   const DvceArray5D<Real> &w0, const EOS_Data &eos_data);
void ZoomAMR(MeshBlockPack* pmbp) {pmbp->pzoom->AMR();}
Real ZoomNewTimeStep(Mesh* pm) {return pm->pmb_pack->pzoom->NewTimeStep(pm);}
void AccFinalWork(ParameterInput *pin, Mesh *pm);

} // namespace

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem()
//! \brief set initial conditions for Bondi accretion test
//  Compile with '-D PROBLEM=gr_acc' to enroll as user-specific problem generator
//    reference: Hawley, Smarr, & Wilson 1984, ApJ 277 296 (HSW)

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  bool is_mhd = (pmbp->pmhd != nullptr);
  auto peos = (is_mhd) ? pmbp->pmhd->peos : pmbp->phydro->peos;
  auto &coord = pmbp->pcoord->coord_data;

  // set user-defined BCs and error function pointers
  // pgen_final_func = BondiErrors;
  user_bcs_func = FixedBondiInflow;
  user_srcs_func = AddUserSrcs;
  user_hist_func = BondiFluxes;
  if (pmbp->pzoom != nullptr && pmbp->pzoom->is_set) {
    pmbp->pzoom->Update(restart);
    pmbp->pzoom->PrintInfo();
    user_ref_func = ZoomAMR;
    if (pmbp->pzoom->zoom_dt) user_dt_func = ZoomNewTimeStep;
  }
  pgen_final_func = AccFinalWork;

  // Read problem-specific parameters from input file
  // global parameters
  acc.k_adi = pin->GetReal("problem", "k_adi");
  acc.r_crit = pin->GetReal("problem", "r_crit");

  // Get ideal gas EOS data
  acc.gm = peos->eos_data.gamma;
  Real gm1 = acc.gm - 1.0;

  // Parameters
  acc.temp_min = 1.0e-7;  // lesser temperature root must be greater than this
  acc.temp_max = 1.0e0;   // greater temperature root must be less than this

  // Get spin of black hole
  acc.spin = pmbp->pcoord->coord_data.bh_spin;

  // Get excision parameters
  acc.dexcise = pmbp->pcoord->coord_data.dexcise;
  acc.pexcise = pmbp->pcoord->coord_data.pexcise;

  acc.potential = pin->GetOrAddBoolean("problem", "potential", false);
  acc.sigma2 = pin->GetOrAddReal("problem", "sigma2", 0.0);
  acc.r_iso = pin->GetOrAddReal("problem", "r_iso", 0.0);

  // Get initial condition type
  std::string ic_type = pin->GetOrAddString("problem","ic_type","gr_bondi");
  if (ic_type == "gr_bondi") { acc.ic_type = 0;
  } else if (ic_type == "uniform") { acc.ic_type = 1;
  } else if (ic_type == "entropy") { acc.ic_type = 2;
  } else {
    std::cout << "### ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Unknown boundary condition type: " << ic_type << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Get ratio of specific heats
  acc.n_adi = 1.0/(acc.gm - 1.0);

  // Prepare various constants for determining primitives
  Real u_crit_sq = 1.0/(2.0*acc.r_crit);                           // (HSW 71)
  Real u_crit = -sqrt(u_crit_sq);
  Real t_crit = (acc.n_adi/(acc.n_adi+1.0)
                 * u_crit_sq/(1.0-(acc.n_adi+3.0)*u_crit_sq));     // (HSW 74)
  acc.c1 = pow(t_crit, acc.n_adi) * u_crit * SQR(acc.r_crit);  // (HSW 68)
  acc.c2 = (SQR(1.0 + (acc.n_adi+1.0) * t_crit)
              * (1.0 - 3.0/(2.0*acc.r_crit)));                     // (HSW 69)
  acc.temp_norm = (sqrt(acc.c2)-1.0)/(1.0+acc.n_adi);           // (HSW 69)
  acc.c_s_norm = sqrt(acc.gm * acc.temp_norm);
  acc.r_bondi = 2.0/SQR(acc.c_s_norm);

  acc.cooling = pin->GetOrAddBoolean("problem", "cooling", false);
  acc.mu_h = pin->GetOrAddReal("problem", "mu_h", 1.0);
  acc.tau_cool = pin->GetOrAddReal("problem", "tau_cool", -1.0);
  acc.r_entropy = pin->GetOrAddReal("problem", "r_entropy", std::numeric_limits<Real>::max());

  if (acc.ic_type > 0) {
    // Prepare various constants for determining primitives
    acc.temp_norm = pin->GetReal("problem", "temp_norm");
    acc.c_s_norm = sqrt(acc.gm * acc.temp_norm);
    acc.r_bondi = 2.0/SQR(acc.c_s_norm);
    acc.rho_norm = pow(acc.temp_norm/acc.k_adi, acc.n_adi);
    acc.rho_norm = pin->GetOrAddReal("problem", "dens_norm", acc.rho_norm);
    if (acc.cooling) {
      Real t_char = acc.r_bondi / acc.c_s_norm;
      auto punit = pmbp->punit;
      Real temp_unit = punit->temperature_cgs();
      Real n_h_unit = punit->density_cgs()/acc.mu_h/punit->atomic_mass_unit_cgs;
      Real cooling_unit = punit->pressure_cgs()/punit->time_cgs()/SQR(n_h_unit);
      Real lambda_cooling = ISMCoolFn(acc.temp_norm*temp_unit)/cooling_unit;
      if (acc.tau_cool > 0.0) {
        Real t_cool = acc.tau_cool * t_char;
        acc.rho_norm = acc.temp_norm / (t_cool * gm1 * lambda_cooling);  
      } else {
        Real t_cool = acc.temp_norm / (acc.rho_norm * gm1 * lambda_cooling);
        acc.tau_cool = t_cool / t_char;
      }
    }
    acc.k_adi = acc.temp_norm / pow(acc.rho_norm, gm1); // update k_adi
    acc.pgas_norm = acc.rho_norm * acc.temp_norm;
    // acc.r_crit = (5.0-3.0*acc.gm)/4.0;
    // acc.c1 = 0.25*pow(2.0/(5.0-3.0*acc.gm), (5.0-3.0*acc.gm)*0.5*acc.n_adi);
    // acc.c2 = -acc.n_adi; // useless in Newtonian case
  }

  std::string bc_type = pin->GetOrAddString("problem","bc_type","none");
  if (bc_type == "none") { acc.bc_type = 0;
  } else if (bc_type == "fixed") { acc.bc_type = 1;
  } else {
    std::cout << "### ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Unknown boundary condition type: " << bc_type << std::endl;
    std::exit(EXIT_FAILURE);
  }
  acc.rb_out = pin->GetOrAddReal("problem","rb_out", std::numeric_limits<Real>::max());

  acc.ndiag = pin->GetOrAddInteger("problem", "ndiag", -1);

  if (global_variable::my_rank == 0) {
    std::cout << " Bondi radius = " << acc.r_bondi << std::endl;
    std::cout << " Critical radius = " << acc.r_crit << std::endl;
    std::cout << " c1 = " << acc.c1 << std::endl;
    std::cout << " rho_norm = " << acc.rho_norm << std::endl;
    if (acc.potential) {
      std::cout << " Potential: sigma2 = " << acc.sigma2 << " r_iso = " << acc.r_iso
                << std::endl;
    }
    if (acc.cooling) {
      std::cout << " t_cool / t_bondi = " << acc.tau_cool << std::endl;
    }
    if (acc.ic_type == 2) {
      std::cout << " Radius of entropy core: r_entropy = " << acc.r_entropy << std::endl;
    }
    std::cout << " Initial condition type  = " << ic_type << std::endl;
    std::cout << " Boundary condition type = " << bc_type << std::endl;
    std::cout << " Outer boundary radius   = " << acc.rb_out << std::endl;
  }

  // Extract BH parameters
  const Real r_excise = coord.rexcise;
  const bool is_radiation_enabled = (pmbp->prad != nullptr);
  // Spherical Grid for user-defined history
  auto &grids = spherical_grids;
  const Real rflux =
    (is_radiation_enabled) ? ceil(r_excise + 1.0) : 1.0 + sqrt(1.0 - SQR(acc.spin));
  int nintp = pin->GetOrAddInteger("problem", "hist_nintp", 2);
  grids.push_back(std::make_unique<SphericalGrid>(pmbp, 10, rflux, nintp));
  grids.push_back(std::make_unique<SphericalGrid>(pmbp, 10, 1.5*std::pow(2.0,0.5), nintp));
  int hist_nr = pin->GetOrAddInteger("problem", "hist_nr", 4);
  Real rmin = pin->GetOrAddReal("problem", "hist_rmin", 3.0);
  Real rmax = pin->GetOrAddReal("problem", "hist_rmax", 0.75*pmy_mesh_->mesh_size.x1max);
  for (int i=0; i<hist_nr-2; i++) {
    Real r_i = std::pow(rmax/rmin,static_cast<Real>(i)/static_cast<Real>(hist_nr-3))*rmin;
    grids.push_back(std::make_unique<SphericalGrid>(pmbp, 10, r_i, nintp));
  }
  if (global_variable::my_rank == 0) {
    std::cout << "Spherical grids for user-defined history:" << std::endl;
    std::cout << "  rmin = " << rmin << " rmax = " << rmax << std::endl;
    for (auto &grid : grids) {
      std::cout << "  r = " << grid->radius << std::endl;
    }
  }
  // return if restart
  if (restart) return;

  // capture variables for the kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  auto acc_ = acc;
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
  // local parameters
  Real pert_amp = pin->GetOrAddReal("problem", "pert_amp", 0.0);
  Kokkos::Random_XorShift64_Pool<> rand_pool64(pmbp->gids);
  par_for("pgen_acc", DevExeSpace(), 0,(nmb-1),0,n3m1,0,n2m1,0,n1m1,
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

    // TODO: add flat IC
    ComputePrimitiveSingle(x1v,x2v,x3v,coord,acc_,rho,pgas,uu1,uu2,uu3);
    // Calculate perturbation
    auto rand_gen = rand_pool64.get_state(); // get random number state this thread
    Real perturbation = 2.0*pert_amp*(rand_gen.frand() - 0.5);
    rand_pool64.free_state(rand_gen);        // free state for use by other threads
    w0_(m,IDN,k,j,i) = rho;
    w0_(m,IEN,k,j,i) = pgas/gm1 * (1.0 + perturbation);
    w0_(m,IM1,k,j,i) = uu1;
    w0_(m,IM2,k,j,i) = uu2;
    w0_(m,IM3,k,j,i) = uu3;
  });

  // Add magnetic field
  Real b_ini = pin->GetOrAddReal("problem", "b_ini", 0.0);
  if (is_mhd && b_ini>0.0) {
    auto &bcc0_ = pmbp->pmhd->bcc0;
    auto &b0_ = pmbp->pmhd->b0;
    par_for("pgen_acc_bfield", DevExeSpace(), 0,(nmb-1),0,n3m1,0,n2m1,0,n1m1,
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
  if (acc.reset_ic) {
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
//! \fn void ProblemGenerator::BondiErrors()
//  \brief Computes errors in linear wave solution and outputs to file.

void BondiErrors(ParameterInput *pin, Mesh *pm) {
  // calculate reference solution by calling pgen again.  Solution stored in second
  // register u1/b1 when flag is false.
  acc.reset_ic=true;
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
//! \fn static Real Acceleration()
//  \brief Computes the gravitational acceleration at a given radius

KOKKOS_INLINE_FUNCTION
static Real Acceleration(struct acc_pgen pgen, const Real r) {
  if (pgen.r_iso > 0.0) {
    return - pgen.sigma2 * (1.0-pgen.r_iso/r*log(1.0+r/pgen.r_iso)) / r;
  }
  return - pgen.sigma2 / r;
};

//----------------------------------------------------------------------------------------
// Function for returning corresponding Boyer-Lindquist coordinates of point
// Inputs:
//   x1,x2,x3: global coordinates to be converted
// Outputs:
//   pr,ptheta,pphi: variables pointed to set to Boyer-Lindquist coordinates

KOKKOS_INLINE_FUNCTION
static void ComputePrimitiveSingle(Real x1v, Real x2v, Real x3v, CoordData coord,
                                   struct acc_pgen pgen,
                                   Real& rho, Real& pgas,
                                   Real& uu1, Real& uu2, Real& uu3) {
  if (pgen.ic_type == 1) {
    rho = pgen.rho_norm;
    pgas = pgen.pgas_norm;
    uu1 = 0.0;
    uu2 = 0.0;
    uu3 = 0.0;
    return;
  }
  // Calculate Boyer-Lindquist coordinates of cell
  Real r, theta, phi;
  GetBoyerLindquistCoordinates(pgen, x1v, x2v, x3v, &r, &theta, &phi);

  // TODO(@mhguo): may add power-law index for entropy
  // entropy core profile
  if (pgen.ic_type == 2) {
    // rho = pgen.rho_norm/(1.0 + (fmax(r-pgen.r_bondi,0.0)/pgen.r_bondi));
    // rho = pgen.rho_norm/((r>pgen.r_bondi) ? r/pgen.r_bondi : 1.0);
    Real temp = pgen.temp_norm; // isothermal
    Real entropy = pgen.k_adi * (1.0 + r/pgen.r_entropy);
    rho = pow(temp/entropy, pgen.n_adi);
    pgas = rho * pgen.temp_norm;
    uu1 = 0.0;
    uu2 = 0.0;
    uu3 = 0.0;
    return;
  }

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
static void GetBoyerLindquistCoordinates(struct acc_pgen pgen,
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
static void TransformVector(struct acc_pgen pgen,
                            Real a1_bl, Real a2_bl, Real a3_bl,
                            Real x1, Real x2, Real x3,
                            Real *pa1, Real *pa2, Real *pa3) {
  Real rad = sqrt( SQR(x1) + SQR(x2) + SQR(x3) );
  Real spin = pgen.spin;
  Real r = fmax((sqrt( SQR(rad) - SQR(spin) + sqrt(SQR(SQR(rad)-SQR(spin))
                      + 4.0*SQR(spin)*SQR(x3)) ) / sqrt(2.0)), 1.0);
  Real delta = SQR(r) - 2.0*r + SQR(spin);
  *pa1 = a1_bl * ( (r*x1+spin*x2)/(SQR(r) + SQR(spin)) - x2*spin/delta) +
         a2_bl * x1*x3/r * sqrt((SQR(r) + SQR(spin))/(SQR(x1) + SQR(x2)))
         - a3_bl * x2;
  *pa2 = a1_bl * ( (r*x2-spin*x1)/(SQR(r) + SQR(spin)) + x1*spin/delta) +
         a2_bl * x2*x3/r * sqrt((SQR(r) + SQR(spin))/(SQR(x1) + SQR(x2)))
         + a3_bl * x1;
  *pa3 = a1_bl * x3/r - a2_bl * r * sqrt((SQR(x1) + SQR(x2))/(SQR(r) + SQR(spin)));
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
static void CalculatePrimitives(struct acc_pgen pgen, Real r,
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
static Real TemperatureMin(struct acc_pgen pgen, Real r, Real t_min, Real t_max) {
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
static Real TemperatureBisect(struct acc_pgen pgen, Real r, Real t_min, Real t_max) {
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
static Real TemperatureResidual(struct acc_pgen pgen, Real t, Real r) {
  return SQR(1.0 + (pgen.n_adi+1.0) * t)
      * (1.0 - 2.0/r + SQR(pgen.c1)
         / (SQR(SQR(r)) * pow(t, 2.0*pgen.n_adi))) - pgen.c2;
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
  int nx1 = indcs.nx1;
  int nx2 = indcs.nx2;
  int nx3 = indcs.nx3;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int n1m1 = n1 - 1, n2m1 = n2 - 1, n3m1 = n3 - 1;
  int &is = indcs.is;  int &ie  = indcs.ie;
  int &js = indcs.js;  int &je  = indcs.je;
  int &ks = indcs.ks;  int &ke  = indcs.ke;
  auto &mb_bcs = pmbp->pmb->mb_bcs;
  int nmb = pmbp->nmb_thispack;

  auto acc_ = acc;
  bool is_mhd = (pmbp->pmhd != nullptr);
  auto u0_ = is_mhd ? pmbp->pmhd->u0 : pmbp->phydro->u0;
  auto w0_ = is_mhd ? pmbp->pmhd->w0 : pmbp->phydro->w0;

  // extract BH parameters
  bool &flat = coord.is_minkowski;
  Real &spin = coord.bh_spin;

  const EOS_Data &eos_data = (pmbp->pmhd != nullptr) ?
                              pmbp->pmhd->peos->eos_data : pmbp->phydro->peos->eos_data;
  Real gamma = eos_data.gamma;

  // Primitive boundary conditions
  // X1-Boundary
  // Set X1-BCs on b0 if Meshblock face is at the edge of computational domain
  if (is_mhd) {
    auto &b0 = pmbp->pmhd->b0;
    par_for("noinflow_field_x1", DevExeSpace(),0,(nmb-1),0,n3m1,0,n2m1,
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
  }
  // TODO (@mhguo): check whether it should be is or is-1, also in gr_torus problem
  // ConsToPrim over all X1 ghost zones *and* at the innermost/outermost X1-active zones
  // of Meshblocks, even if Meshblock face is not at the edge of computational domain
  if (!is_mhd) {
    pmbp->phydro->peos->ConsToPrim(u0_,w0_,false,is-ng,is,0,n2m1,0,n3m1);
    pmbp->phydro->peos->ConsToPrim(u0_,w0_,false,ie,ie+ng,0,n2m1,0,n3m1);
  } else {
    auto &bcc0_ = pmbp->pmhd->bcc0;
    auto &b0_ = pmbp->pmhd->b0;
    pmbp->pmhd->peos->ConsToPrim(u0_,b0_,w0_,bcc0_,false,is-ng,is,0,n2m1,0,n3m1);
    pmbp->pmhd->peos->ConsToPrim(u0_,b0_,w0_,bcc0_,false,ie,ie+ng,0,n2m1,0,n3m1);
  }
  // Set X1-BCs on w0 if Meshblock face is at the edge of computational domain
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
      ComputePrimitiveSingle(x1v,x2v,x3v,coord,acc_,rho,pgas,uu1,uu2,uu3);
      w0_(m,IDN,k,j,i) = rho;
      w0_(m,IEN,k,j,i) = pgas/(acc_.gm - 1.0);
      w0_(m,IM1,k,j,i) = uu1;
      w0_(m,IM2,k,j,i) = uu2;
      w0_(m,IM3,k,j,i) = uu3;
    }

    // outer x1 boundary
    x1v = CellCenterX((ie+i+1)-is, indcs.nx1, x1min, x1max);

    if (mb_bcs.d_view(m,BoundaryFace::outer_x1) == BoundaryFlag::user) {
      ComputePrimitiveSingle(x1v,x2v,x3v,coord,acc_, rho,pgas,uu1,uu2,uu3);
      w0_(m,IDN,k,j,(ie+i+1)) = rho;
      w0_(m,IEN,k,j,(ie+i+1)) = pgas/(acc_.gm - 1.0);
      w0_(m,IM1,k,j,(ie+i+1)) = uu1;
      w0_(m,IM2,k,j,(ie+i+1)) = uu2;
      w0_(m,IM3,k,j,(ie+i+1)) = uu3;
    }
  });
  // PrimToCons on X1 physical boundary ghost zones
  if (!is_mhd) {
    pmbp->phydro->peos->PrimToCons(w0_,u0_,is-ng,is-1,0,n2m1,0,n3m1);
    pmbp->phydro->peos->PrimToCons(w0_,u0_,ie+1,ie+ng,0,n2m1,0,n3m1);
  } else {
    auto &bcc0_ = pmbp->pmhd->bcc0;
    pmbp->pmhd->peos->PrimToCons(w0_,bcc0_,u0_,is-ng,is-1,0,n2m1,0,n3m1);
    pmbp->pmhd->peos->PrimToCons(w0_,bcc0_,u0_,ie+1,ie+ng,0,n2m1,0,n3m1);
  }

  // X2-Boundary
  // Set X2-BCs on b0 if Meshblock face is at the edge of computational domain
  if (is_mhd) {
    auto &b0 = pmbp->pmhd->b0;
    par_for("noinflow_field_x2", DevExeSpace(),0,(nmb-1),0,n3m1,0,n1m1,
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
  }
  // ConsToPrim over all X2 ghost zones *and* at the innermost/outermost X2-active zones
  // of Meshblocks, even if Meshblock face is not at the edge of computational domain
  if (!is_mhd) {
    pmbp->phydro->peos->ConsToPrim(u0_,w0_,false,0,n1m1,js-ng,js,0,n3m1);
    pmbp->phydro->peos->ConsToPrim(u0_,w0_,false,0,n1m1,je,je+ng,0,n3m1);
  } else {
    auto &bcc0_ = pmbp->pmhd->bcc0;
    auto &b0_ = pmbp->pmhd->b0;
    pmbp->pmhd->peos->ConsToPrim(u0_,b0_,w0_,bcc0_,false,0,n1m1,js-ng,js,0,n3m1);
    pmbp->pmhd->peos->ConsToPrim(u0_,b0_,w0_,bcc0_,false,0,n1m1,je,je+ng,0,n3m1);
  }
  // Set X2-BCs on w0 if Meshblock face is at the edge of computational domain
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
      ComputePrimitiveSingle(x1v,x2v,x3v,coord,acc_,rho,pgas,uu1,uu2,uu3);
      w0_(m,IDN,k,j,i) = rho;
      w0_(m,IEN,k,j,i) = pgas/(acc_.gm - 1.0);
      w0_(m,IM1,k,j,i) = uu1;
      w0_(m,IM2,k,j,i) = uu2;
      w0_(m,IM3,k,j,i) = uu3;
    }

    // outer x2 boundary
    x2v = CellCenterX((je+j+1)-js, indcs.nx2, x2min, x2max);

    if (mb_bcs.d_view(m,BoundaryFace::outer_x2) == BoundaryFlag::user) {
      ComputePrimitiveSingle(x1v,x2v,x3v,coord,acc_,rho,pgas,uu1,uu2,uu3);
      w0_(m,IDN,k,(je+j+1),i) = rho;
      w0_(m,IEN,k,(je+j+1),i) = pgas/(acc_.gm - 1.0);
      w0_(m,IM1,k,(je+j+1),i) = uu1;
      w0_(m,IM2,k,(je+j+1),i) = uu2;
      w0_(m,IM3,k,(je+j+1),i) = uu3;
    }
  });
  // PrimToCons on X2 physical boundary ghost zones
  if (!is_mhd) {
    pmbp->phydro->peos->PrimToCons(w0_,u0_,0,n1m1,js-ng,js-1,0,n3m1);
    pmbp->phydro->peos->PrimToCons(w0_,u0_,0,n1m1,je+1,je+ng,0,n3m1);
  } else {
    auto &bcc0_ = pmbp->pmhd->bcc0;
    pmbp->pmhd->peos->PrimToCons(w0_,bcc0_,u0_,0,n1m1,js-ng,js-1,0,n3m1);
    pmbp->pmhd->peos->PrimToCons(w0_,bcc0_,u0_,0,n1m1,je+1,je+ng,0,n3m1);
  }

  // X3-Boundary
  // Set X3-BCs on b0 if Meshblock face is at the edge of computational domain
  if (is_mhd) {
    auto &b0 = pmbp->pmhd->b0;
    par_for("noinflow_field_x3", DevExeSpace(),0,(nmb-1),0,n2m1,0,n1m1,
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
  // ConsToPrim over all X3 ghost zones *and* at the innermost/outermost X3-active zones
  // of Meshblocks, even if Meshblock face is not at the edge of computational domain
  if (!is_mhd) {
    pmbp->phydro->peos->ConsToPrim(u0_,w0_,false,0,n1m1,0,n2m1,ks-ng,ks);
    pmbp->phydro->peos->ConsToPrim(u0_,w0_,false,0,n1m1,0,n2m1,ke,ke+ng);
  } else {
    auto &bcc0_ = pmbp->pmhd->bcc0;
    auto &b0_ = pmbp->pmhd->b0;
    pmbp->pmhd->peos->ConsToPrim(u0_,b0_,w0_,bcc0_,false,0,n1m1,0,n2m1,ks-ng,ks-1);
    pmbp->pmhd->peos->ConsToPrim(u0_,b0_,w0_,bcc0_,false,0,n1m1,0,n2m1,ke+1,ke+ng);
  }
  // Set X3-BCs on w0 if Meshblock face is at the edge of computational domain
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
      ComputePrimitiveSingle(x1v,x2v,x3v,coord,acc_,rho,pgas,uu1,uu2,uu3);
      w0_(m,IDN,k,j,i) = rho;
      w0_(m,IEN,k,j,i) = pgas/(acc_.gm - 1.0);
      w0_(m,IM1,k,j,i) = uu1;
      w0_(m,IM2,k,j,i) = uu2;
      w0_(m,IM3,k,j,i) = uu3;
    }

    // outer x3 boundary
    x3v = CellCenterX((ke+k+1)-ks, indcs.nx3, x3min, x3max);

    if (mb_bcs.d_view(m,BoundaryFace::outer_x3) == BoundaryFlag::user) {
      ComputePrimitiveSingle(x1v,x2v,x3v,coord,acc_,rho,pgas,uu1,uu2,uu3);
      w0_(m,IDN,(ke+k+1),j,i) = rho;
      w0_(m,IEN,(ke+k+1),j,i) = pgas/(acc_.gm - 1.0);
      w0_(m,IM1,(ke+k+1),j,i) = uu1;
      w0_(m,IM2,(ke+k+1),j,i) = uu2;
      w0_(m,IM3,(ke+k+1),j,i) = uu3;
    }
  });
  // PrimToCons on X3 physical boundary ghost zones
  if (!is_mhd) {
    pmbp->phydro->peos->PrimToCons(w0_,u0_,0,n1m1,0,n2m1,ks-ng,ks-1);
    pmbp->phydro->peos->PrimToCons(w0_,u0_,0,n1m1,0,n2m1,ke+1,ke+ng);
  } else {
    auto &bcc0_ = pmbp->pmhd->bcc0;
    pmbp->pmhd->peos->PrimToCons(w0_,bcc0_,u0_,0,n1m1,0,n2m1,ks-ng,ks-1);
    pmbp->pmhd->peos->PrimToCons(w0_,bcc0_,u0_,0,n1m1,0,n2m1,ke+1,ke+ng);
  }

  // TODO: assuming MHD!
  // TODO: think whether need refining check as we don't use u1
  // bool refining = (pm->pmr != nullptr) ? pm->pmr->refining : false;
  if (is_mhd && acc_.bc_type == 1) { // && !refining
    Real rbout = acc_.rb_out;
    auto &b = pmbp->pmhd->b0;
    par_for("fixed_radial", DevExeSpace(),0,nmb-1,ks-ng,ke+ng,js-ng,je+ng,is-ng,ie+ng,
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

      // apply initial conditions to boundary cells
      if (rad > rbout) {
        Real glower[4][4], gupper[4][4];
        ComputeMetricAndInverse(x1v, x2v, x3v, flat, spin, glower, gupper);
        Real rho, pgas, uu1, uu2, uu3;
        ComputePrimitiveSingle(x1v,x2v,x3v,coord,acc_,rho,pgas,uu1,uu2,uu3);
        HydCons1D u;
        MHDPrim1D w_m;
        // load single state of primitive variables
        w_m.d  = rho;
        w_m.vx = uu1;
        w_m.vy = uu2;
        w_m.vz = uu3;
        w_m.e  = pgas/(acc_.gm - 1.0);
        // load cell-centered fields into primitive state
        w_m.bx = 0.5*(b.x1f(m,k,j,i) + b.x1f(m,k,j,i+1));
        w_m.by = 0.5*(b.x2f(m,k,j,i) + b.x2f(m,k,j,i+1));
        w_m.bz = 0.5*(b.x3f(m,k,j,i) + b.x3f(m,k,j,i+1));
        SingleP2C_IdealGRMHD(glower, gupper, w_m, gamma, u);
        u0_(m,IDN,k,j,i) = u.d;
        u0_(m,IM1,k,j,i) = u.mx;
        u0_(m,IM2,k,j,i) = u.my;
        u0_(m,IM3,k,j,i) = u.mz;
        u0_(m,IEN,k,j,i) = u.e;
      }
    });
  }

  if (pm->pmb_pack->pzoom != nullptr && pm->pmb_pack->pzoom->is_set) {
    pm->pmb_pack->pzoom->BoundaryConditions();
  }

  return;
}

//----------------------------------------------------------------------------------------
// Function for computing accretion fluxes through constant spherical KS radius surfaces

void BondiFluxes(HistoryData *pdata, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;

  // extract BH parameters
  bool &flat = pmbp->pcoord->coord_data.is_minkowski;
  Real &spin = pmbp->pcoord->coord_data.bh_spin;

  // set nvars, adiabatic index, primitive array w0, and field array bcc0 if is_mhd
  int nvars; Real gamma; Real tfloor; bool is_mhd = false;
  DvceArray5D<Real> w0_, bcc0_;
  if (pmbp->phydro != nullptr) {
    nvars = pmbp->phydro->nhydro + pmbp->phydro->nscalars;
    gamma = pmbp->phydro->peos->eos_data.gamma;
    tfloor = pmbp->phydro->peos->eos_data.tfloor;
    w0_ = pmbp->phydro->w0;
  } else if (pmbp->pmhd != nullptr) {
    is_mhd = true;
    nvars = pmbp->pmhd->nmhd + pmbp->pmhd->nscalars;
    gamma = pmbp->pmhd->peos->eos_data.gamma;
    tfloor = pmbp->pmhd->peos->eos_data.tfloor;
    w0_ = pmbp->pmhd->w0;
    bcc0_ = pmbp->pmhd->bcc0;
  }

  // extract grids, number of radii, number of fluxes, and history appending index
  auto &grids = pm->pgen->spherical_grids;
  int nradii = grids.size();
  // int nflux = (is_mhd) ? 4 : 3;
  const int nflux = 30;

  // set number of and names of history variables for hydro or mhd
  //  (1) mass accretion rate
  //  (2) energy flux
  //  (3) angular momentum flux
  //  (4) magnetic flux (iff MHD)
  pdata->nhist = 0;

  // history of zoom variables
  if (pmbp->pzoom != nullptr && pmbp->pzoom->is_set) {
    pdata->nhist += 1;
    pdata->label[0] = "zone";
    pdata->hdata[0] = (global_variable::my_rank == 0)? pmbp->pzoom->zrun.zone : 0.0;
    if (pmbp->pzoom->calc_cons_change) {
      pdata->nhist += 2;
      pdata->label[1] = "dm";
      pdata->hdata[1] = (global_variable::my_rank == 0)? pmbp->pzoom->zchg.dmass : 0.0;
      pdata->label[2] = "de";
      pdata->hdata[2] = (global_variable::my_rank == 0)? pmbp->pzoom->zchg.dengy : 0.0;
    }
  }

  // history of global sum variables
  int sn0 = pdata->nhist;
  const int nsum = 9;
  pdata->nhist += nsum;
  std::string sdata_label[nsum] = {
    // 6 letters for the first 7 labels, 5 for the rest
    "Vglb ", "Mglb ", "Eglb ", "Cglb ",  "Eint ", "Ekin ", "Emag ",
    "Vcold", "Mcold",
  };
  for (int n=0; n<nsum; ++n) {
    pdata->label[sn0+n] = sdata_label[n];
  }

  // history of spherical grid variables
  int gn0 = pdata->nhist;
  pdata->nhist += nradii*nflux;
  // TODO: add angular momentum components
  // no more than 7 characters per label
  std::string gdata_label[nflux] = {"r","out","m","mout","mdot","mdotout","edot","edotout",
    "lx","ly","lz","lzout","phi","eint","b^2","alpha","lor","u0","u_0","ur","uph","b0",
    "b_0","br","bph","edothyd","edho","edotadv","edao","cooling"
  };
  for (int g=0; g<nradii; ++g) {
    std::string gstr = std::to_string(g);
    for (int n=0; n<nflux; ++n) {
      pdata->label[gn0+nflux*g+n] = gdata_label[n] + "_" + gstr;
    }
  }

  if (pdata->nhist > NHISTORY_VARIABLES) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "User history function specified pdata->nhist larger than"
              << " NHISTORY_VARIABLES" << std::endl;
    exit(EXIT_FAILURE);
  }

  Real temp_unit = pmbp->punit->temperature_cgs();
  Real n_h_unit = pmbp->punit->density_cgs()/acc.mu_h/pmbp->punit->atomic_mass_unit_cgs;
  Real cooling_unit = pmbp->punit->pressure_cgs()/pmbp->punit->time_cgs()/SQR(n_h_unit);

  // compute global sum
  // capture class variabels for kernel
  auto &size = pmbp->pmb->mb_size;
  // loop over all MeshBlocks in this pack
  auto &indcs = pm->mb_indcs;
  int is = indcs.is; int nx1 = indcs.nx1;
  int js = indcs.js; int nx2 = indcs.nx2;
  int ks = indcs.ks; int nx3 = indcs.nx3;
  const int nmkji = (pmbp->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;
  array_sum::GlobalSum sum_this_mb0;
  // store data into hdata array
  for (int n=0; n<NREDUCTION_VARIABLES; ++n) {
    sum_this_mb0.the_array[n] = 0.0;
  }
  Real rglb = std::min(acc.rb_out, acc.r_entropy);
  Kokkos::parallel_reduce("HistSums",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, array_sum::GlobalSum &mb_sum0) {
    // compute n,k,j,i indices of thread
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
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

    Real rad = sqrt(SQR(x1v)+SQR(x2v)+SQR(x3v));
    Real vol = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;

    Real glower[4][4], gupper[4][4];
    ComputeMetricAndInverse(x1v,x2v,x3v,flat,spin,glower,gupper);

    Real dens = w0_(m,IDN,k,j,i);
    Real temp = (gamma-1.0) * w0_(m,IEN,k,j,i)/w0_(m,IDN,k,j,i);
    
    // flags
    Real on = (rad <= rglb)? 1.0 : 0.0; // check if inside the radius of interest
    Real dv_cold = (temp<0.1/rad)? vol : 0.0;

    // TODO: consider relativistic correction
    Real mass = vol * w0_(m,IDN,k,j,i);
    Real eint = vol * w0_(m,IEN,k,j,i);
    Real lambda_cooling = (temp>tfloor) ? ISMCoolFn(temp*temp_unit)/cooling_unit : 0.0;
    Real cooling_rate = dens * dens * lambda_cooling;
    Real cooling = vol * cooling_rate;
    Real dm_cold = dv_cold * w0_(m,IDN,k,j,i);
    // TODO: consider add these terms, either in Newtonian or relativistic
    Real ekin = mass * 0.5 * (SQR(w0_(m,IVX,k,j,i)) +
                              SQR(w0_(m,IVY,k,j,i)) +
                              SQR(w0_(m,IVZ,k,j,i)));
    Real emag = vol * 0.5 * (SQR(bcc0_(m,IBX,k,j,i)) +
                             SQR(bcc0_(m,IBY,k,j,i)) +
                             SQR(bcc0_(m,IBZ,k,j,i)));
    Real etot = eint + ekin + emag;

    Real vars[nsum] = {
      vol,     mass,    etot,    cooling, eint,    ekin,    emag,
      dv_cold, dm_cold,
    };

    // sum variables:
    array_sum::GlobalSum hvars0;
    for (int n=0; n<nsum; ++n) {
      hvars0.the_array[n] = vars[n]*on;
    }

    // sum into parallel reduce
    mb_sum0 += hvars0;
  }, Kokkos::Sum<array_sum::GlobalSum>(sum_this_mb0));

  // store data into hdata array
  for (int n=0; n<nsum; ++n) {
    pdata->hdata[sn0+n] = sum_this_mb0.the_array[n];
  }

  // go through angles at each radii:
  DualArray2D<Real> interpolated_bcc;  // needed for MHD
  for (int g=0; g<nradii; ++g) {
    // zero fluxes at this radius
    for (int n=0; n<nflux; ++n) {
      pdata->hdata[gn0+nflux*g+n] = 0.0;
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
      Real dphdx = (-x2/(SQR(x1)+SQR(x2)) + (spin/(r2 + a2))*drdx);
      Real dphdy = ( x1/(SQR(x1)+SQR(x2)) + (spin/(r2 + a2))*drdy);
      Real dphdz = (spin/(r2 + a2)*drdz);
      // contravariant r component of 4-velocity
      Real ur  = drdx *u1 + drdy *u2 + drdz *u3;
      // contravariant r component of 4-magnetic field (returns zero if not MHD)
      Real br  = drdx *b1 + drdy *b2 + drdz *b3;
      // phi component of 4-velocity in spherical KS
      Real uph = dphdx*u1 + dphdy*u2 + dphdz*u3;
      // phi component of 4-magnetic field in spherical KS
      Real bph = dphdx*b1 + dphdy*b2 + dphdz*b3;
      // covariant phi component of 4-velocity
      Real u_ph = (-r*sph-spin*cph)*sth*u_1 + (r*cph-spin*sph)*sth*u_2;
      // covariant phi component of 4-magnetic field (returns zero if not MHD)
      Real b_ph = (-r*sph-spin*cph)*sth*b_1 + (r*cph-spin*sph)*sth*b_2;

      // integration params
      Real &domega = grids[g]->solid_angles.h_view(n);
      Real sqrtmdet = (r2+SQR(spin*cos(theta)));

      // flags
      Real on = (int_dn != 0.0)? 1.0 : 0.0; // check if angle is on this rank
      Real is_out = (ur>0.0)? 1.0 : 0.0;

      // compute mass flux
      Real m_flx = int_dn*ur;

      // compute energy flux
      Real t1_0 = (int_dn + gamma*int_ie + b_sq)*ur*u_0 - br*b_0;
      // compute angular momentum flux
      // TODO(@mhguo): write a correct function to compute x,y angular momentum flux
      Real t1_1 = 0.0;
      Real t1_2 = 0.0;
      Real t1_3 = (int_dn + gamma*int_ie + b_sq)*ur*u_ph - br*b_ph;
      Real phi_flx = (is_mhd) ? 0.5*fabs(br*u0 - b0*ur) : 0.0;
      Real t1_0_hyd = (int_dn + gamma*int_ie)*ur*u_0;
      Real bernl_hyd = (on)? -(1.0 + gamma*int_ie/int_dn)*u_0-1.0 : 0.0;
      Real temp = (on) ? (gamma-1.0)*int_ie/int_dn : 0.0;
      Real lambda_cooling = (temp>tfloor) ? ISMCoolFn(temp*temp_unit)/cooling_unit : 0.0;
      Real cooling_rate = int_dn * int_dn * lambda_cooling;

      Real flux_data[nflux] = {r, is_out, int_dn, int_dn*is_out, m_flx, m_flx*is_out,
        t1_0, t1_0*is_out, t1_1, t1_2, t1_3, t1_3*is_out, phi_flx, 
        int_ie, b_sq, alpha, lor, u0, u_0, ur, uph, b0, b_0, br, bph,
        t1_0_hyd, t1_0_hyd*is_out, bernl_hyd, bernl_hyd*is_out, cooling_rate
      };

      pdata->hdata[gn0+nflux*g+0] = (global_variable::my_rank == 0)? flux_data[0] : 0.0;
      for (int n=1; n<nflux; ++n) {
        pdata->hdata[gn0+nflux*g+n] += flux_data[n]*sqrtmdet*domega*on;
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
//! \fn void AddUserSrcs()
//! \brief Add User Source Terms
// NOTE source terms must all be computed using primitive (w0) and NOT conserved (u0) vars
void AddUserSrcs(Mesh *pm, const Real bdt) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  DvceArray5D<Real> &u0 = (pmbp->pmhd != nullptr) ? pmbp->pmhd->u0 : pmbp->phydro->u0;
  DvceArray5D<Real> &w0 = (pmbp->pmhd != nullptr) ? pmbp->pmhd->w0 : pmbp->phydro->w0;
  const EOS_Data &eos_data = (pmbp->pmhd != nullptr) ?
                             pmbp->pmhd->peos->eos_data : pmbp->phydro->peos->eos_data;
  if (acc.potential) {
    //std::cout << "AddAccel" << std::endl;
    AddAccel(pm,bdt,u0,w0);
  }
  if (acc.cooling) {
    //std::cout << "AddISMCooling" << std::endl;
    AddISMCooling(pm,bdt,u0,w0,eos_data);
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void AddAccel()
//! \brief Add Acceleration
// NOTE source terms must all be computed using primitive (w0) and NOT conserved (u0) vars
void AddAccel(Mesh *pm, const Real bdt, DvceArray5D<Real> &u0,
              const DvceArray5D<Real> &w0) {
              // const DvceArray5D<Real> &bcc, const AthenaArray<Real> &rad_arr,
              // const AthenaArray<Real> &press_arr, const AthenaArray<Real> &mom_arr)
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pmbp->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie, nx1 = indcs.nx1;
  int js = indcs.js, je = indcs.je, nx2 = indcs.nx2;
  int ks = indcs.ks, ke = indcs.ke, nx3 = indcs.nx3;
  int nmb1 = pmbp->nmb_thispack - 1;
  auto size = pmbp->pmb->mb_size;
  auto acc_ = acc;
  Real rin = 1.0;
  bool is_gr = pmbp->pcoord->is_general_relativistic;

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

    Real accel = Acceleration(acc_, rad);
    if (rad < rin) {
      accel = 0.0;
    }
    Real dmomr = bdt*w0(m,IDN,k,j,i)*accel;
    Real dmomx1 = dmomr*x1v/rad;
    Real dmomx2 = dmomr*x2v/rad;
    Real dmomx3 = dmomr*x3v/rad;
    Real denergy = bdt*w0(m,IDN,k,j,i)*accel/rad*
                  (w0(m,IVX,k,j,i)*x1v+w0(m,IVY,k,j,i)*x2v+w0(m,IVZ,k,j,i)*x3v);
    if (is_gr) {
      denergy = -denergy;
    }

    u0(m,IM1,k,j,i) += dmomx1;
    u0(m,IM2,k,j,i) += dmomx2;
    u0(m,IM3,k,j,i) += dmomx3;
    u0(m,IEN,k,j,i) += denergy;
  });
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void AddISMCooling()
//! \brief Add explict ISM cooling and heating source terms in the energy equations.
// NOTE source terms must all be computed using primitive (w0) and NOT conserved (u0) vars

void AddISMCooling(Mesh *pm, const Real bdt, DvceArray5D<Real> &u0,
                   const DvceArray5D<Real> &w0, const EOS_Data &eos_data) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pmbp->pmesh->mb_indcs;
  int is = indcs.is, nx1 = indcs.nx1;
  int js = indcs.js, nx2 = indcs.nx2;
  int ks = indcs.ks, nx3 = indcs.nx3;
  const int nmkji = (pmbp->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;
  auto &size = pmbp->pmb->mb_size;
  Real beta = bdt/pm->dt;
  Real cfl_no = pm->cfl_no;
  auto &eos = eos_data;
  Real use_e = eos_data.use_e;
  Real tfloor = eos_data.tfloor;
  Real gamma = eos_data.gamma;
  Real gm1 = gamma - 1.0;
  Real temp_unit = pmbp->punit->temperature_cgs();
  Real n_h_unit = pmbp->punit->density_cgs()/acc.mu_h/pmbp->punit->atomic_mass_unit_cgs;
  Real cooling_unit = pmbp->punit->pressure_cgs()/pmbp->punit->time_cgs()/SQR(n_h_unit);
  // Real gamma_heating = 2.0e-26/heating_unit; // add a small heating
  bool is_gr = pmbp->pcoord->is_general_relativistic;
  // extract BH parameters
  bool &flat = pmbp->pcoord->coord_data.is_minkowski;
  Real &spin = pmbp->pcoord->coord_data.bh_spin;

  bool is_hydro = true;
  DvceArray5D<Real> bcc;
  if (pmbp->pmhd != nullptr) {
    is_hydro = false;
    // using bcc is ok here because b0 is not updated yet
    bcc = pmbp->pmhd->bcc0;
  }

  int nsubcycle=0, nsubcycle_count=0;
  Kokkos::parallel_reduce("cooling", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, int &sum0, int &sum1) {
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;
    Real dens = w0(m,IDN,k,j,i);
    Real temp = w0(m,IEN,k,j,i)/w0(m,IDN,k,j,i)*gm1;
    Real eint = w0(m,IEN,k,j,i);
    // compute cooling/heating terms
    Real lambda_cooling = (temp<=tfloor) ? 0.0 : ISMCoolFn(temp*temp_unit)/cooling_unit;
    Real cooling_heating = dens * dens * lambda_cooling;

    bool sub_cycling = true;
    bool sub_cycling_used = false;
    Real bdt_now = 0.0;
    if (is_gr) {
      // Extract components of metric
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
      Real glower[4][4], gupper[4][4];
      ComputeMetricAndInverse(x1v, x2v, x3v, flat, spin, glower, gupper);
      // load single state conserved variables
      HydPrim1D w;
      HydCons1D u;
      MHDCons1D u_m;
      MHDPrim1D w_m;
      bool dfloor_used=false, efloor_used=false, c2p_failure=false;
      int iter_used=0;
      if (is_hydro) {
        u.d  = u0(m,IDN,k,j,i);
        u.mx = u0(m,IM1,k,j,i);
        u.my = u0(m,IM2,k,j,i);
        u.mz = u0(m,IM3,k,j,i);
        u.e  = u0(m,IEN,k,j,i);
        HydCons1D u_sr;
        Real s2;
        // call c2p function
        TransformToSRHyd(u,glower,gupper,s2,u_sr);
        SingleC2P_IdealSRHyd(u_sr, eos, s2, w,
                            dfloor_used, efloor_used, c2p_failure, iter_used);
        // add cooling/heating term
        w.e -= bdt * cooling_heating;
        SingleP2C_IdealGRHyd(glower, gupper, w, gamma, u);
      } else {
        u_m.d  = u0(m,IDN,k,j,i);
        u_m.mx = u0(m,IM1,k,j,i);
        u_m.my = u0(m,IM2,k,j,i);
        u_m.mz = u0(m,IM3,k,j,i);
        u_m.e  = u0(m,IEN,k,j,i);
        //TODO: are you sure bcc is updated?
        u_m.bx = bcc(m,IBX,k,j,i);
        u_m.by = bcc(m,IBY,k,j,i);
        u_m.bz = bcc(m,IBZ,k,j,i);
        MHDCons1D u_sr;
        Real s2, b2, rpar;
        TransformToSRMHD(u_m,glower,gupper,s2,b2,rpar,u_sr);
        // call c2p function
        // (inline function in ideal_c2p_mhd.hpp file)
        SingleC2P_IdealSRMHD(u_sr, eos, s2, b2, rpar, x1v, x2v, x3v, w,
                            dfloor_used, efloor_used, c2p_failure, iter_used);
        // add cooling/heating term using subcycling if necessary
        // TODO: w.d is updated, not equal to dens, should think which one to use
        do {
          Real dt_cool = (w.e/(FLT_MIN + fabs(cooling_heating)));
          // half of the timestep
          Real bdt_cool = 0.5*beta*cfl_no*dt_cool;
          if (bdt_now+bdt_cool<bdt) {
            w.e -= bdt_cool * cooling_heating;
            temp = w.e/w.d*gm1;
            lambda_cooling = (temp>tfloor) ? ISMCoolFn(temp*temp_unit)/cooling_unit : 0.0;
            cooling_heating = w.d * w.d * lambda_cooling;
            sub_cycling_used = true;
            sum1++;
          } else {
            w.e -= (bdt-bdt_now) * cooling_heating;
            sub_cycling = false;
          }
          bdt_now += bdt_cool;
        } while (sub_cycling);
        // load single state of primitive variables
        w_m.d  = w.d;
        w_m.vx = w.vx;
        w_m.vy = w.vy;
        w_m.vz = w.vz;
        w_m.e  = w.e;
        // load cell-centered fields into primitive state
        w_m.bx = u_m.bx;
        w_m.by = u_m.by;
        w_m.bz = u_m.bz;
        // call p2c function
        SingleP2C_IdealGRMHD(glower, gupper, w_m, gamma, u);
      }
      u0(m,IDN,k,j,i) = u.d;
      u0(m,IM1,k,j,i) = u.mx;
      u0(m,IM2,k,j,i) = u.my;
      u0(m,IM3,k,j,i) = u.mz;
      u0(m,IEN,k,j,i) = u.e;
    } else {
      sub_cycling = false;
    }
    if (sub_cycling_used) {
      sum0++;
    }
  }, Kokkos::Sum<int>(nsubcycle), Kokkos::Sum<int>(nsubcycle_count));
#if MPI_PARALLEL_ENABLED
  int* pnsubcycle = &(nsubcycle);
  int* pnsubcycle_count = &(nsubcycle_count);
  MPI_Allreduce(MPI_IN_PLACE, pnsubcycle, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, pnsubcycle_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif
  if (global_variable::my_rank == 0) {
    if (acc.ndiag>0 && pm->ncycle % acc.ndiag == 0) {
      if (nsubcycle>0 || nsubcycle_count >0) {
        std::cout << " nsubcycle_cell=" << nsubcycle << std::endl
                  << " nsubcycle_count=" << nsubcycle_count << std::endl;
      }
    }
  }
  return;
}

void AccFinalWork(ParameterInput *pin, Mesh *pm) {
  delete accarr;
}

} // namespace