//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file acc->cpp
//! \brief Attempt to build a model for Accretion of Black Holes in Elliptical Galaxies

// Athena++ headers
#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "eos/ideal_c2p_hyd.hpp"
#include "eos/ideal_c2p_mhd.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "pgen.hpp"
#include "srcterms/ismcooling.hpp"
#include "globals.hpp"
#include "units/units.hpp"

#include "pgen/turb_init.hpp"
#include "pgen/turb_mhd.hpp"

#include <Kokkos_Random.hpp>

using RK4FnPtr = Real (*)(Real x, Real y);

#define NLEN_DENS_ARRAY 2048
//#define NREDUCTION_RADIAL 64
#define NREDUCTION_RADIAL 24
namespace array_acc {  // namespace helps with name resolution in reduction identity
typedef array_sum::array_type<Real,(NREDUCTION_RADIAL)> RadSum;  // simplifies code below
} // namespace array_acc
namespace Kokkos { //reduction identity must be defined in Kokkos namespace
template<>
struct reduction_identity< array_acc::RadSum > {
  KOKKOS_FORCEINLINE_FUNCTION static array_acc::RadSum sum() {
    return array_acc::RadSum();
  }
};
}

namespace {

static Real RK4(RK4FnPtr func, Real x, Real y, Real h);
static Real DrhoDr(Real rho, Real x);
void SolveDens(DualArray1D<Real> &d_arr);
Real DensFn(Real x, const DualArray1D<Real> &d_arr);
KOKKOS_INLINE_FUNCTION
Real DensFnDevice(Real x, const DualArray1D<Real> &d_arr);
KOKKOS_INLINE_FUNCTION
int GetRadialIndex(Real rad, Real logr0, Real logh, int rntot);
KOKKOS_INLINE_FUNCTION
Real GetRadialVar(DualArray1D<Real> vararr, Real rad, Real logr0, Real logh, int rntot);

struct pgenacc {
  Real grav;  // gravitational constant in code unit
  Real r_refine; // mesh refinement radius
  Real dens_refine;
  Real tfloor;
  Real heat_tceiling;
  Real gamma;    // EOS parameters
  bool potential;
  Real m_bh;
  Real r_in;
  Real r_in_old; // Inner boundary radius for restarting
  Real r_in_new; // Inner boundary radius for restarting
  Real r_in_beg_t; // start changing inner radius
  Real r_in_sof_t; // time for changing inner radius
  Real m_star, r_star;
  Real m_dm, r_dm;
  Real sink_d, sink_t;
  Real rho_0, temp_0;
  Real k0_entry, xi_entry; // Parameters for entropy
  Real rad_entry, dens_entry; // Parameters for entropy
  Real rb_in; // Inner boundary radius
  Real rb_out; // Outer boundary radius
  std::string user_bc_flag; // Boundary condition
  Real dt_floor; // floor of timestep
  Real sink_dt_floor; // floor of timestep
  bool turb; // turbulence
  Real turb_amp; // amplitude of the perturbations
  Real mu_h; // ratio of total number density to hydrogen nuclei number density
  bool cooling;
  bool heating_ini;
  bool heating_ana;
  bool heating_pow;
  Real rad_heat;
  Real radpow_heat;
  Real heatnorm;
  Real fac_heat;
  bool heating_equ;
  int heat_weight;
  int bins_heat;
  Real rmin_heat;
  Real rmax_heat;
  Real logr_heat;
  Real logh_heat;
  bool heating_mdot;
  Real epsilon;
  Real mdot_dr;
  Real heat_beg_time;
  Real heat_sof_time; // soft time
  int heat_cycle; // heating rate updating cycle
  int ndiag;
  Real t_cold;  // criterion of cold gas
  Real tf_hot;  // criterion of hot gas as fraction of initial temperature
  DualArray1D<Real> dens_arr;
  DualArray1D<Real> logcooling_arr;
  array_acc::RadSum v_arr;
  array_acc::RadSum c_arr;
  //TurbulenceInit *pturb;
};

pgenacc* acc = new pgenacc();

//DualArray1D<Real> dens_arr("dens_arr", 512);

// prototypes for user-defined BCs and source terms
void RadialBoundary(Mesh *pm);
void AddUserSrcs(Mesh *pm, const Real bdt);
void AccRefine(MeshBlockPack* pmbp);
void AccHistOutput(HistoryData *pdata, Mesh *pm);
void AccFinalWork(ParameterInput *pin, Mesh *pm);

void Diagnostic(Mesh *pm, const DvceArray5D<Real> &w0, const EOS_Data &eos_data);
void AddAccel(Mesh *pm, const Real bdt, DvceArray5D<Real> &u0,
              const DvceArray5D<Real> &w0);
void AddISMCooling(Mesh *pm, const Real bdt, DvceArray5D<Real> &u0,
                   const DvceArray5D<Real> &w0, const EOS_Data &eos_data);
void AddIniHeating(Mesh *pm, const Real bdt, DvceArray5D<Real> &u0,
                const DvceArray5D<Real> &w0, const EOS_Data &eos_data);
void AddAnaHeating(Mesh *pm, const Real bdt, DvceArray5D<Real> &u0,
                   const DvceArray5D<Real> &w0, const EOS_Data &eos_data);
void AddEquHeating(Mesh *pm, const Real bdt, DvceArray5D<Real> &u0,
                   const DvceArray5D<Real> &w0, const EOS_Data &eos_data);
void AddPowHeating(Mesh *pm, const Real bdt, DvceArray5D<Real> &u0,
                   const DvceArray5D<Real> &w0);
void AddMdotHeating(Mesh *pm, const Real bdt, DvceArray5D<Real> &u0,
                    const DvceArray5D<Real> &w0, const EOS_Data &eos_data);

//----------------------------------------------------------------------------------------
//! \f computes gravitational acceleration

KOKKOS_INLINE_FUNCTION
static Real NFWMass(const Real r, const Real ms, const Real rs) {
  return ms*(log(1.0+(r/(rs)))-(r)/((rs+r)));
}

KOKKOS_INLINE_FUNCTION
static Real Acceleration(const Real r, const Real m, const Real mc, const Real rc,
                         const Real ms, const Real rs, const Real g) {
  return -g*(m+NFWMass(r,mc,rc)+NFWMass(r,ms,rs))/SQR(r);
}

} // namespace

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem()
//! \brief Problem Generator for accretion

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  user_bcs_func = RadialBoundary;
  user_srcs_func = AddUserSrcs;
  user_ref_func = AccRefine;
  user_hist_func = AccHistOutput;
  pgen_final_func = AccFinalWork;
  Kokkos::realloc(acc->dens_arr,NLEN_DENS_ARRAY);
  Kokkos::realloc(acc->logcooling_arr,NREDUCTION_RADIAL);
  //dens_arr("dens_arr", pp->nmb_thispack, 512),

  // capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  int &ng = indcs.ng;
  int n1m1 = indcs.nx1 + 2*ng - 1;
  int n2m1 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng - 1) : 0;
  int n3m1 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng - 1) : 0;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &size = pmbp->pmb->mb_size;
  int nmb1 = (pmbp->nmb_thispack-1);
  bool is_gr = pmbp->pcoord->is_general_relativistic;

  // Check if turn on the problem
  acc->potential = pin->GetOrAddBoolean("problem","potential",false);
  bool profile = pin->GetOrAddBoolean("problem","profile",false);
  if (!profile && !acc->potential) {
    auto &u0 = (pmbp->pmhd != nullptr) ? pmbp->pmhd->u0 : pmbp->phydro->u0;
    par_for("pgen_accretion", DevExeSpace(),0,nmb1,ks-ng,ke+ng,js-ng,je+ng,is-ng,ie+ng,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      u0(m,IDN,k,j,i) = 1.0;
      u0(m,IM1,k,j,i) = 0.0;
      u0(m,IM2,k,j,i) = 0.0;
      u0(m,IM3,k,j,i) = 0.0;
      u0(m,IEN,k,j,i) = 1.0;
    });
    return;
  }

  // Get parameters
  acc->grav = pmbp->punit->grav_constant();
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
  acc->r_refine = pin->GetOrAddReal("problem","r_refine",0.0);
  acc->dens_refine = pin->GetOrAddReal("problem","dens_refine",0.0);
  acc->ndiag = pin->GetOrAddInteger("problem","ndiag",-1);
  acc->rho_0 = pin->GetOrAddReal("problem","dens",1.0);
  Real temp_kelvin = pin->GetOrAddReal("problem","temp",1.0);
  bool rst_flag = pin->GetOrAddBoolean("problem","rst",false);
  std::string rst_file = pin->GetOrAddString("problem", "rst_file", "none");
  int old_level = pin->GetOrAddInteger("problem", "old_level", 0);

  if (acc->potential) {
    acc->m_bh = pin->GetReal("problem","m_bh");
    acc->r_in = pin->GetReal("problem","r_in");
    acc->r_in_old = pin->GetOrAddReal("problem","r_in_old",acc->r_in);
    acc->r_in_new = pin->GetOrAddReal("problem","r_in_new",acc->r_in);
    acc->r_in_beg_t = pin->GetOrAddReal("problem","r_in_beg_t",0.0);
    acc->r_in_sof_t = pin->GetOrAddReal("problem","r_in_sof_t",0.0);
    if (acc->r_in_new<acc->r_in_old) {
      Real t_now = pmbp->pmesh->time-acc->r_in_beg_t;
      if (t_now<=0.0) {
        acc->r_in = acc->r_in_old;
      } else if (t_now<acc->r_in_sof_t) {
        Real t_ratio = t_now/acc->r_in_sof_t;
        //acc->r_in = std::pow(rinn,t_ratio)*std::pow(rino,(1.0-t_ratio));
        acc->r_in = acc->r_in_new*t_ratio+acc->r_in_old*(1.0-t_ratio);
      } else {
        acc->r_in = acc->r_in_new;
      }
    }
    acc->m_star = pin->GetReal("problem","m_star");
    acc->r_star = pin->GetReal("problem","r_star");
    acc->m_dm = pin->GetReal("problem","m_dm");
    acc->r_dm = pin->GetReal("problem","r_dm");
    acc->sink_d = pin->GetReal("problem","sink_d");
    acc->sink_t = pin->GetReal("problem","sink_t");
    acc->rad_entry = pin->GetReal("problem","rad_entry");
    acc->dens_entry = pin->GetReal("problem","dens_entry");
    acc->k0_entry = pin->GetReal("problem","k0_entry");
    acc->xi_entry = pin->GetReal("problem","xi_entry");
    acc->rb_in = pin->GetOrAddReal("problem","rb_in",0.0);
    acc->rb_out = pin->GetOrAddReal("problem","rb_out",std::numeric_limits<Real>::max());
  } else {
    return;
  }
  acc->user_bc_flag = pin->GetOrAddString("problem","bc_flag","none");
  if (acc->user_bc_flag == "none") {
    std::cout << "### ERROR in " << __FILE__ << " at line " << __LINE__  << std::endl
              << "Boundary condition is not set!" << std::endl
              << "Please set either fixed or outflow!" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  acc->turb = pin->GetOrAddBoolean("problem","turb",false);
  if (acc->turb) {
    acc->turb_amp = pin->GetOrAddReal("problem","turb_amp",0.0);
  }
  acc->mu_h = pin->GetOrAddReal("problem","mu_h",1.4);
  acc->cooling = pin->GetOrAddBoolean("problem","cooling",false);
  acc->heating_ini = pin->GetOrAddBoolean("problem","heating_ini",false);
  acc->heating_ana = pin->GetOrAddBoolean("problem","heating_ana",false);
  acc->heating_equ = pin->GetOrAddBoolean("problem","heating_equ",false);
  acc->heating_pow = pin->GetOrAddBoolean("problem","heating_pow",false);
  acc->heat_beg_time = pin->GetOrAddReal("problem","heat_beg_time",0.0);
  acc->heat_sof_time = pin->GetOrAddReal("problem","heat_sof_time",0.0);
  acc->heat_cycle = pin->GetOrAddInteger("problem","heat_cycle",10);
  acc->fac_heat = pin->GetOrAddReal("problem","fac_heat",1.0);
  if (acc->heating_ana) {
    acc->radpow_heat = pin->GetOrAddReal("problem","radpow_heat",-1.0);
  }
  if (acc->heating_equ) {
    acc->bins_heat = NREDUCTION_RADIAL;
    acc->heat_weight = pin->GetOrAddInteger("problem","heat_weight",0);
    acc->rmin_heat = pin->GetOrAddReal("problem","rmin_heat",1e-2);
    acc->rmax_heat = pin->GetOrAddReal("problem","rmax_heat",1e2);
    acc->logh_heat = std::log10(acc->rmax_heat/acc->rmin_heat)/acc->bins_heat;
    acc->logr_heat = std::log10(acc->rmin_heat)+0.5*acc->logh_heat;
    for ( int i = 0; i < acc->bins_heat; i++ ) {
      acc->v_arr.the_array[i] = 0.0;
      acc->c_arr.the_array[i] = 0.0;
      acc->logcooling_arr.h_view(i) = -100.0; //1e-100
      acc->logcooling_arr.template modify<HostMemSpace>();
      acc->logcooling_arr.template sync<DevExeSpace>();
    }
  }
  acc->heating_mdot = pin->GetOrAddBoolean("problem","heating_mdot",false);
  if (acc->heating_mdot) {
    acc->epsilon = pin->GetOrAddReal("problem","epsilon",1e-6);
    acc->mdot_dr = pin->GetOrAddReal("problem","mdot_dr",0.2*acc->r_in);
    acc->rad_heat = pin->GetOrAddReal("problem","rad_heat",0.0);
    acc->radpow_heat = pin->GetOrAddReal("problem","radpow_heat",-2.0);
  }
  if (acc->heating_pow) {
    acc->rad_heat = pin->GetOrAddReal("problem","rad_heat",2.0);
    acc->radpow_heat = pin->GetOrAddReal("problem","radpow_heat",-1.5);
  }
  acc->heat_tceiling = pin->GetOrAddReal("problem","heat_tceiling",
                        std::numeric_limits<float>::max());
  acc->t_cold = pin->GetOrAddReal("problem","t_cold",0.03);
  acc->tf_hot = pin->GetOrAddReal("problem","tf_hot",0.3);
  // End get parameters

  Real &mbh = acc->m_bh;
  Real &mstar = acc->m_star;
  Real &rstar = acc->r_star;
  Real &mdm = acc->m_dm;
  Real &rdm = acc->r_dm;
  Real &radentry = acc->rad_entry;
  Real &k0 = acc->k0_entry;
  Real &xi = acc->xi_entry;
  Real grav = acc->grav;
  //Real r_in_old = acc->r_in_old;

  Real cs_iso = std::sqrt(temp_kelvin/pmbp->punit->temperature_cgs());

  // Initialize Hydro/MHD variables -------------------------------
  auto &w0 = (pmbp->pmhd != nullptr) ? pmbp->pmhd->w0 : pmbp->phydro->w0;
  auto &u0 = (pmbp->pmhd != nullptr) ? pmbp->pmhd->u0 : pmbp->phydro->u0;
  EOS_Data &eos = (pmbp->pmhd != nullptr) ?
                  pmbp->pmhd->peos->eos_data : pmbp->phydro->peos->eos_data;

  // update problem-specific parameters
  eos.r_in = acc->r_in;
  acc->dt_floor = eos.dt_floor;
  acc->sink_dt_floor = pin->GetOrAddReal("problem","sink_dt",0.0);
  if (acc->sink_dt_floor < acc->dt_floor) {acc->sink_dt_floor = acc->dt_floor;}
  acc->tfloor = eos.tfloor;
  acc->gamma = eos.gamma;
  acc->temp_0 = cs_iso*cs_iso;
  Real gm1 = eos.gamma - 1.0;
  Real &gamma=acc->gamma;
  Real &rho0=acc->rho_0;
  Real &temp0=acc->temp_0;
  Real pgas_0 = acc->rho_0*cs_iso*cs_iso;
  Real cs = std::sqrt(eos.gamma*pgas_0/acc->rho_0);

  auto &d_arr = acc->dens_arr;
  //auto &d_arr = pm->pgen->dens_arr;
  SolveDens(d_arr);

  if (acc->heating_pow) {
    Real temp_unit = pmbp->punit->temperature_cgs();
    Real n_h_unit = pmbp->punit->density_cgs()/acc->mu_h
                  /pmbp->punit->atomic_mass_unit_cgs;
    Real cooling_unit = pmbp->punit->pressure_cgs()/pmbp->punit->time_cgs()/SQR(n_h_unit);
    Real &rbout = acc->rb_out;
    Real radh = acc->rad_heat;
    Real radpow = acc->radpow_heat;
    Real coolrate = 0.0;
    Real heatrate = 0.0;
    int n_sum = 32000;
    for (int i=100; i<=n_sum; i++) {
      //std::cout << "  i = " << i << std::endl;
      Real rad = static_cast<Real>(i)/static_cast<Real>(n_sum)*rbout;
      Real x = rad/radentry;
      Real rho = DensFn(x,d_arr);
      Real pgas = 0.5*k0*(1.0+pow(x,xi))*pow(rho,gamma);
      // temperature in cgs unit
      Real temp = temp_unit*pgas/rho;

      coolrate += SQR(rad)*SQR(rho)*ISMCoolFn(temp)/cooling_unit;
      heatrate += SQR(rad)*rho*pow(rad+radh,radpow);
    }
    acc->heatnorm = acc->fac_heat*coolrate/heatrate;
  }

  // Spherical Grid for user-defined history
  auto &grids = spherical_grids;
  Real hist_a1 = pin->GetOrAddReal("problem","hist_a1",1.1);
  Real hist_a2 = pin->GetOrAddReal("problem","hist_a2",3.0);
  Real hist_a3 = pin->GetOrAddReal("problem","hist_a3",10.0);
  Real hist_a4 = pin->GetOrAddReal("problem","hist_a4",100.0);
  grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, hist_a1*acc->r_in));
  // Enroll additional radii for flux analysis by
  // pushing back the grids vector with additional SphericalGrid instances
  grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, hist_a2*acc->r_in));
  grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, hist_a3*acc->r_in));
  grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, hist_a4*acc->r_in));

  // Print info
  if (global_variable::my_rank == 0) {
    std::cout << "============== Check Initialization ===============" << std::endl;
    std::cout << "  rho_0 (code) = " << rho0 << std::endl;
    std::cout << "  sound speed (code) = " << cs << std::endl;
    std::cout << "  mu = " << pmbp->punit->mu() << std::endl;
    std::cout << "  temperature (code) = " << temp0 << std::endl;
    std::cout << "  temperature (c.g.s) = " << temp_kelvin << std::endl;
    std::cout << "  cooling function (c.g.s) = " << ISMCoolFn(temp_kelvin) << std::endl;
    std::cout << "  grav const (code) = " << grav << std::endl;
    std::cout << "  user_bcs = " << user_bcs << std::endl;
    std::cout << "  r_in = " << acc->r_in << std::endl;
    std::cout << "  r_in_old = " << acc->r_in_old << std::endl;
    std::cout << "  r_in_new = " << acc->r_in_new << std::endl;
    std::cout << "  r_in_beg_t = " << acc->r_in_beg_t << std::endl;
    std::cout << "  r_in_sof_t = " << acc->r_in_sof_t << std::endl;
    std::cout << "  rb_in = " << acc->rb_in << std::endl;
    std::cout << "  sink_d = " << acc->sink_d << std::endl;
    std::cout << "  sink_t = " << acc->sink_t << std::endl;
    std::cout << "  user_srcs = " << user_srcs << std::endl;
    std::cout << "  potential = " << acc->potential << std::endl;
    std::cout << "  cooling = " << acc->cooling << std::endl;
    std::cout << "  heating_ini = " << acc->heating_ini << std::endl;
    std::cout << "  heating_ana = " << acc->heating_ana << std::endl;
    std::cout << "  heating_equ = " << acc->heating_equ << std::endl;
    std::cout << "  heating_pow = " << acc->heating_pow << std::endl;
    std::cout << "  heating_mdot = " << acc->heating_mdot << std::endl;
    std::cout << "  radpow_heat = " << acc->radpow_heat << std::endl;
    std::cout << "  heat_beg_time = " << acc->heat_beg_time << std::endl;
    std::cout << "  heat_sof_time = " << acc->heat_sof_time << std::endl;
    std::cout << "  fac_heat = " << acc->fac_heat << std::endl;
    if (acc->heating_pow) {
      std::cout << "  rad_heat = " << acc->rad_heat << std::endl;
      std::cout << "  heatnorm = " << acc->heatnorm << std::endl;
    }
    if (acc->heating_equ) {
      std::cout << "  heat_weight = " << acc->heat_weight << std::endl;
    }
    std::cout << "===================================================" << std::endl;
    for (int i=0; i<7; i++) {
      Real r = pow(10.0,static_cast<Real>(i-3));
      std::cout << "  r: " << r << ", a: "
                << Acceleration(r,mbh,mstar,rstar,mdm,rdm,grav) << std::endl;
    }
    for (int i=0; i<41; i++) {
      Real x = pow(10,0.1*i-2.);
      Real r = x * radentry;
      Real rho = DensFn(x,d_arr);
      Real pgas = 0.5*k0*(1.0+pow(x,xi))*pow(rho,gamma);
      Real r_l = r*0.9;
      Real x_l = r_l / radentry;
      Real rho_l = DensFn(x_l,d_arr);
      Real pgas_l = 0.5*k0*(1.0+pow(x_l,xi))*pow(rho_l,gamma);
      Real r_r = r*1.1;
      Real x_r = r_r / radentry;
      Real rho_r = DensFn(x_r,d_arr);
      Real pgas_r = 0.5*k0*(1.0+pow(x_r,xi))*pow(rho_r,gamma);
      Real dpdr = (pgas_r-pgas_l)/(r*0.2);
      Real rhog = rho*Acceleration(r,mbh,mstar,rstar,mdm,rdm,grav);
      std::cout << "  r: " << r << " rho: " << rho << " pres: " << pgas
                << " dpdr: " << dpdr << " rhog: " << rhog << std::endl;
    }
  }
  // End print info

  // Reset the BH mass to 0.0, note that radial profile is solved in Newtonian already
  if (is_gr) {
    acc->m_bh = 0.0;
  }
  // Convert Newtonian to GR
  bool newton_to_gr = pin->GetOrAddBoolean("problem","newton_to_gr",false);
  if (newton_to_gr) {
    // Convert primitives to conserved
    if (pmbp->phydro != nullptr) {
      EquationOfState *peos = new IdealHydro(pmbp, pin);
      peos->ConsToPrim(u0, w0, false, 0, n1m1, 0, n2m1, 0, n3m1);
      pmbp->phydro->peos->PrimToCons(w0, u0, 0, n1m1, 0, n2m1, 0, n3m1);
      delete peos;
    } else if (pmbp->pmhd != nullptr) {
      EquationOfState *peos = new IdealMHD(pmbp, pin);
      auto &bcc0_ = pmbp->pmhd->bcc0;
      auto &b0_ = pmbp->pmhd->b0;
      peos->ConsToPrim(u0, b0_, w0, bcc0_, false, 0, n1m1, 0, n2m1, 0, n3m1);
      pmbp->pmhd->peos->PrimToCons(w0, bcc0_, u0, 0, n1m1, 0, n2m1, 0, n3m1);
      delete peos;
    }
  }

  if (restart) return;

  // Set initial conditions
  Kokkos::Random_XorShift64_Pool<> rand_pool64(pmbp->gids);
  par_for("pgen_accretion", DevExeSpace(),0,nmb1,ks-ng,ke+ng,js-ng,je+ng,is-ng,ie+ng,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    int nx1 = indcs.nx1;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    int nx2 = indcs.nx2;
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    int nx3 = indcs.nx3;
    Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

    Real rad = sqrt(SQR(x1v)+SQR(x2v)+SQR(x3v));

    //Real rho = rho0/(1.0+rad/2.0);
    //Real temp = temp0*(1.0+rad/2.0);
    //Real pgas = rho*temp;

    Real x = rad/radentry;
    Real rho = DensFnDevice(x,d_arr);
    Real pgas = 0.5*k0*(1.0+pow(x,xi))*pow(rho,gamma);

    w0(m,IDN,k,j,i) = rho;
    w0(m,IVX,k,j,i) = 0.0;
    w0(m,IVY,k,j,i) = 0.0;
    w0(m,IVZ,k,j,i) = 0.0;
    if (eos.is_ideal) {
      w0(m,IEN,k,j,i) = pgas/gm1;
    }
  });
  // Convert primitives to conserved
  if (pmbp->phydro != nullptr) {
    pmbp->phydro->peos->PrimToCons(w0, u0, is-ng, ie+ng, js-ng, je+ng, ks-ng, ke+ng);
  } else if (pmbp->pmhd != nullptr) {
    auto &bcc0_ = pmbp->pmhd->bcc0;
    pmbp->pmhd->peos->PrimToCons(w0, bcc0_, u0, is-ng, ie+ng, js-ng, je+ng, ks-ng, ke+ng);
  }
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

  // TODO(@mhguo): read and interpolate data
  // TODO(@mhguo): now only work for hydro!!!
  if (rst_flag) {
    ParameterInput* pinput = new ParameterInput;
    IOWrapper resfile;
    // read parameters from restart file
    resfile.Open(rst_file.c_str(), IOWrapper::FileMode::read);
    pinput->LoadFromFile(resfile);

    int nmb = pmbp->nmb_thispack;
    int nout1 = indcs.nx1 + 2*(indcs.ng);
    int nout2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
    int nout3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;

    // calculate total number of CC variables
    hydro::Hydro* phydro = pmbp->phydro;
    mhd::MHD* pmhd = pmbp->pmhd;
    int nhydro_tot = 0, nmhd_tot = 0;
    if (phydro != nullptr) {
      nhydro_tot = phydro->nhydro + phydro->nscalars;
    }
    if (pmhd != nullptr) {
      nmhd_tot = pmhd->nmhd + pmhd->nscalars;
    }

    // TODO(@mhguo): get old rank
    int maxlevel = pmy_mesh_->max_level-2;
    int oldlevel = old_level;
    int myrank = global_variable::my_rank;
    int newpart = 7*maxlevel+8;
    int oldpart = 7*oldlevel+8;
    int difpart = newpart-oldpart;
    int m_old = -1;
    for (int m=0; m<nmb; ++m) { // TODO(@mhguo): only work for nmb=1 now!
      int mid = myrank+m;
      m_old = mid;
      for (int i=0; i<8; i++) {
        if (mid>=i*newpart+(1-i)*(oldlevel+1) && mid<i*newpart+(7-i)*(oldlevel+1)) {
          m_old = mid-i*difpart;
        }
        if (mid>=i*newpart+(7-i)*(oldlevel+1) && mid<(i+1)*newpart+(-i)*(oldlevel+1)) {
          m_old = -1;
        }
      }
      if (mid>=8*newpart-7*(oldlevel+1)) {
        m_old = mid-8*difpart;
      }
      //m_old = 0;
    }
    if (global_variable::my_rank == 0) { // the master process reads the header data
      for (int m=0; m<nmb; ++m) {
        for (int i=0; i<8; i++) {
          std::cout << " i=" << i
                    << " 0=" << i*newpart+(1-i)*(oldlevel+1)
                    << " 1=" << i*newpart+(7-i)*(oldlevel+1)
                    << " 2=" << (i+1)*newpart+(-i)*(oldlevel+1)
                    << std::endl;
        }
      }
    }

    IOWrapperSizeT headersize = 3*sizeof(int) + 2*sizeof(Real)
      + sizeof(RegionSize) + 2*sizeof(RegionIndcs);
    char *headerdata = new char[headersize];
    if (global_variable::my_rank == 0) { // the master process reads the header data
      if (resfile.Read_bytes(headerdata, 1, headersize) != headersize) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Header size read from restart file is incorrect, "
                  << "restart file is broken." << std::endl;
        exit(EXIT_FAILURE);
      }
    }
#if MPI_PARALLEL_ENABLED
    // then broadcast the header data
    MPI_Bcast(headerdata, headersize, MPI_CHAR, 0, MPI_COMM_WORLD);
#endif
    // get old time and cycle
    IOWrapperSizeT hdos = 0;
    hdos += sizeof(int);
    hdos += sizeof(int);
    hdos += sizeof(RegionSize);
    hdos += sizeof(RegionIndcs);
    hdos += sizeof(RegionIndcs);
    //std::memcpy(&(pmbp->pmesh->time), &(headerdata[hdos]), sizeof(Real));
    hdos += sizeof(Real);
    //std::memcpy(&(pmbp->pmesh->dt), &(headerdata[hdos]), sizeof(Real));
    hdos += sizeof(Real);
    //std::memcpy(&(pmbp->pmesh->ncycle), &(headerdata[hdos]), sizeof(int));
    delete [] headerdata;

    // allocate idlist buffer and read list of logical locations and cost
    IOWrapperSizeT listsize = sizeof(LogicalLocation) + sizeof(float);
    int nmb_old = 8*oldpart;
    char *idlist = new char[listsize*nmb_old];
    if (global_variable::my_rank == 0) { // only the master process reads the ID list
      if (resfile.Read_bytes(idlist,listsize,nmb_old)
          != static_cast<unsigned int>(nmb_old)) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Incorrect number of MeshBlocks in restart file; "
                  << "restart file is broken." << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }

    // root process reads size of CC and FC data arrays from restart file
    IOWrapperSizeT variablesize = 2*sizeof(IOWrapperSizeT);
    char *variabledata = new char[variablesize];
    if (global_variable::my_rank == 0) { // the master process reads the variables data
      if (resfile.Read_bytes(variabledata, 1, variablesize) != variablesize) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl
                  << "Variable data size read from restart file is incorrect, "
                  << "restart file is broken." << std::endl;
        exit(EXIT_FAILURE);
      }
    }
#if MPI_PARALLEL_ENABLED
    // then broadcast the datasize information
    MPI_Bcast(variabledata, variablesize, MPI_CHAR, 0, MPI_COMM_WORLD);
#endif

    // TODO(@mhguo): tmpcout!
    //if (global_variable::my_rank == 0) {
    //  std::cout << " gets file offset" << std::endl;
    //}
    IOWrapperSizeT headeroffset;
    // master process gets file offset
    if (global_variable::my_rank == 0) {
      headeroffset = resfile.GetPosition();
    }
#if MPI_PARALLEL_ENABLED
    // then broadcasts it
    MPI_Bcast(&headeroffset, sizeof(IOWrapperSizeT), MPI_CHAR, 0, MPI_COMM_WORLD);
#endif

    IOWrapperSizeT ccdata_size, fcdata_size = 0;
    hdos = 0;
    std::memcpy(&ccdata_size, &(variabledata[hdos]), sizeof(IOWrapperSizeT));
    hdos += sizeof(IOWrapperSizeT);
    std::memcpy(&fcdata_size, &(variabledata[hdos]), sizeof(IOWrapperSizeT));

    // allocate arrays for CC data
    HostArray5D<Real> ccin("pgen-ccin", nmb, (nhydro_tot+nmhd_tot), nout3, nout2, nout1);
    if (ccin.size()*sizeof(Real) != ccdata_size) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << " ccin.size=" << ccin.size()*sizeof(Real)
                << " ccdata_size=" << ccdata_size << " nmb=" << nmb
                << " myrank=" << myrank << " m_old=" << m_old
                << std::endl << "CC data size read from restart file not equal to size "
                << "of Hydro and MHD arrays, restart file is broken." << std::endl;
      exit(EXIT_FAILURE);
    }

    // TODO(@mhguo): tmpcout!
    //if (global_variable::my_rank == 0) {
    //  std::cout << " read" << std::endl;
    //}
    // read CC data into host array
    if (m_old>=0) {
      int mygids = m_old;
      IOWrapperSizeT myoffset = headeroffset + (ccdata_size+fcdata_size)*mygids;
      if (resfile.Read_bytes_at_all(ccin.data(), ccdata_size, 1, myoffset) != 1) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Input hydro data not read correctly from restart "
                  << "file, restart file is broken." << std::endl;
        exit(EXIT_FAILURE);
      }
      myoffset += ccdata_size;

      // copy CC Hydro data to device
      if (phydro != nullptr) {
        DvceArray5D<Real>::HostMirror host_u0 = Kokkos::create_mirror(phydro->u0);
        auto hst_slice = Kokkos::subview(ccin, Kokkos::ALL, std::make_pair(0,nhydro_tot),
                                        Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        Kokkos::deep_copy(host_u0, hst_slice);
        Kokkos::deep_copy(phydro->u0, host_u0);
      }
    } else {
      int mygids = 0;
      IOWrapperSizeT myoffset = headeroffset + (ccdata_size+fcdata_size)*mygids;
      if (resfile.Read_bytes_at_all(ccin.data(), ccdata_size, 1, myoffset) != 1) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Input hydro data not read correctly from restart "
                  << "file, restart file is broken." << std::endl;
        exit(EXIT_FAILURE);
      }
      myoffset += ccdata_size;
    }
    // TODO(@mhguo): tmpcout!
    //std::cout << " Close:" << myrank << std::endl;
    resfile.Close();
    delete pinput;
  }
  // TODO(@mhguo): tmpcout!
  if (global_variable::my_rank == 0) {
    std::cout << " return" << std::endl;
  }

  return;
}

namespace {
//----------------------------------------------------------------------------------------
//! \fn RadialBoundary
//! \brief Sets boundary condition on surfaces of computational domain and radial regions
// Quantities are held fixed to sink cell for r<r_in
// Quantities at boundaryies are held fixed to initial condition values
// for r<rb_in and r>rb_out

void RadialBoundary(Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;

  int nmb = pmbp->nmb_thispack;
  auto u0_ = (pmbp->pmhd != nullptr) ? pmbp->pmhd->u0 : pmbp->phydro->u0;
  auto w0_ = (pmbp->pmhd != nullptr) ? pmbp->pmhd->w0 : pmbp->phydro->w0;
  bool is_mhd = false;
  DvceArray5D<Real> bcc;
  if (pmbp->pmhd != nullptr) {
    is_mhd = true;
    // TODO(@mhguo) using bcc is not good here because b0 is already updated
    bcc = pmbp->pmhd->bcc0;
  }
  auto &mb_bcs = pmbp->pmb->mb_bcs;
  int nvar = u0_.extent_int(1);

  auto &d_arr = acc->dens_arr;
  std::string bc_flag = acc->user_bc_flag;
  Real &gamma=acc->gamma;
  Real gm1=(acc->gamma-1.0);
  Real &radentry = acc->rad_entry;
  Real &k0 = acc->k0_entry;
  Real &xi = acc->xi_entry;
  Real &rbin = acc->rb_in;
  Real &rbout = acc->rb_out;
  Real &rin = acc->r_in;
  Real &sinkd = acc->sink_d;
  Real &sinkt = acc->sink_t;
  Real sinke = sinkd*sinkt/gm1;
  Real dtfloor = acc->sink_dt_floor;
  //Real tfloor = acc->tfloor;
  bool is_gr = pmbp->pcoord->is_general_relativistic;
  bool &flat = pmbp->pcoord->coord_data.is_minkowski;
  Real &spin = pmbp->pcoord->coord_data.bh_spin;

  //bool tmp = mb_bcs.h_view(0,BoundaryFace::inner_x1)==BoundaryFlag::user;
  //std::cout << tmp << std::endl;

  par_for("fixed_radial", DevExeSpace(),0,nmb-1,ks-ng,ke+ng,js-ng,je+ng,is-ng,ie+ng,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    int nx1 = indcs.nx1;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    int nx2 = indcs.nx2;
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    int nx3 = indcs.nx3;
    Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

    Real rad = sqrt(SQR(x1v)+SQR(x2v)+SQR(x3v));

    if (rad < rin) {
      u0_(m,IDN,k,j,i) = sinkd;
      u0_(m,IEN,k,j,i) = sinke;
      u0_(m,IM1,k,j,i) = 0.0;
      u0_(m,IM2,k,j,i) = 0.0;
      u0_(m,IM3,k,j,i) = 0.0;
    }

    // apply initial conditions to boundary cells
    if (rad < rbin || rad > rbout) {
      Real x = rad/radentry;
      Real rho = DensFnDevice(x,d_arr);
      Real eint = 0.5*k0*(1.0+pow(x,xi))*pow(rho,gamma)/gm1;

      if (is_gr) {
        Real glower[4][4], gupper[4][4];
        ComputeMetricAndInverse(x1v, x2v, x3v, flat, spin, glower, gupper);

        HydCons1D u;
        if (is_mhd) {
          MHDPrim1D w;
          w.d  = rho;
          w.vx = 0.0;
          w.vy = 0.0;
          w.vz = 0.0;
          w.e  = eint;
          // load cell-centered fields into primitive state
          w.bx = bcc(m,IBX,k,j,i);
          w.by = bcc(m,IBY,k,j,i);
          w.bz = bcc(m,IBZ,k,j,i);

          // call p2c function
          SingleP2C_IdealGRMHD(glower, gupper, w, gamma, u);
        } else {
          HydPrim1D w;
          w.d  = rho;
          w.vx = 0.0;
          w.vy = 0.0;
          w.vz = 0.0;
          w.e  = eint;

          // call p2c function
          SingleP2C_IdealGRHyd(glower, gupper, w, gamma, u);
        }
        // store conserved quantities in 3D array
        u0_(m,IDN,k,j,i) = u.d;
        u0_(m,IM1,k,j,i) = u.mx;
        u0_(m,IM2,k,j,i) = u.my;
        u0_(m,IM3,k,j,i) = u.mz;
        u0_(m,IEN,k,j,i) = u.e;
      } else {
        u0_(m,IDN,k,j,i) = rho;
        u0_(m,IM1,k,j,i) = 0.0;
        u0_(m,IM2,k,j,i) = 0.0;
        u0_(m,IM3,k,j,i) = 0.0;
        if (is_mhd) {
          u0_(m,IEN,k,j,i) = eint + 0.5*(SQR(bcc(m,IBX,k,j,i)) + SQR(bcc(m,IBY,k,j,i))
                                        + SQR(bcc(m,IBZ,k,j,i)));
        } else {
          u0_(m,IEN,k,j,i) = eint;
        }
      }
    }
  });

  if (pmbp->pmhd != nullptr) {
    auto b0 = pmbp->pmhd->b0;
    if (!is_gr) {
      par_for("bfield_radial", DevExeSpace(),0,nmb-1,ks-ng,ke+ng,js-ng,je+ng,is-ng,ie+ng,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        Real &x1min = size.d_view(m).x1min;
        Real &x1max = size.d_view(m).x1max;
        Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

        Real &x2min = size.d_view(m).x2min;
        Real &x2max = size.d_view(m).x2max;
        Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

        Real &x3min = size.d_view(m).x3min;
        Real &x3max = size.d_view(m).x3max;
        Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

        Real rad = sqrt(SQR(x1v)+SQR(x2v)+SQR(x3v));

        if (rad < rin) {
          Real va1_ceil = size.d_view(m).dx1/dtfloor;
          Real va2_ceil = size.d_view(m).dx2/dtfloor;
          Real va3_ceil = size.d_view(m).dx3/dtfloor;
          Real bx = 0.5*(b0.x1f(m,k,j,i) + b0.x1f(m,k,j,i+1));
          Real by = 0.5*(b0.x2f(m,k,j,i) + b0.x2f(m,k,j+1,i));
          Real bz = 0.5*(b0.x3f(m,k,j,i) + b0.x3f(m,k+1,j,i));
          u0_(m,IDN,k,j,i) = fmax(u0_(m,IDN,k,j,i),SQR(bx/va1_ceil));
          u0_(m,IDN,k,j,i) = fmax(u0_(m,IDN,k,j,i),SQR(by/va2_ceil));
          u0_(m,IDN,k,j,i) = fmax(u0_(m,IDN,k,j,i),SQR(bz/va3_ceil));
          Real dens = u0_(m,IDN,k,j,i);
          Real etot = dens*sinkt/gm1 + 0.5*(SQR(bx) + SQR(by) + SQR(bz));
          u0_(m,IEN,k,j,i) = etot;
          /*b0.x1f(m,k,j,i) = 0.0;
          b0.x2f(m,k,j,i) = 0.0;
          b0.x3f(m,k,j,i) = 0.0;
          b0.x1f(m,k,j,i+1) = 0.0;
          b0.x2f(m,k,j+1,i) = 0.0;
          b0.x3f(m,k+1,j,i) = 0.0;*/
        }
      });
    }

    // outflow condition
    par_for("outflow_bfield_x1", DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n2-1),0,(ng-1),
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      if (mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::user) {
        b0.x1f(m,k,j,is-i-1) = b0.x1f(m,k,j,is);
        b0.x2f(m,k,j,is-i-1) = b0.x2f(m,k,j,is);
        if (j == n2-1) {b0.x2f(m,k,j+1,is-i-1) = b0.x2f(m,k,j+1,is);}
        b0.x3f(m,k,j,is-i-1) = b0.x3f(m,k,j,is);
        if (k == n3-1) {b0.x3f(m,k+1,j,is-i-1) = b0.x3f(m,k+1,j,is);}
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x1) == BoundaryFlag::user) {
        b0.x1f(m,k,j,ie+i+2) = b0.x1f(m,k,j,ie+1);
        b0.x2f(m,k,j,ie+i+1) = b0.x2f(m,k,j,ie);
        if (j == n2-1) {b0.x2f(m,k,j+1,ie+i+1) = b0.x2f(m,k,j+1,ie);}
        b0.x3f(m,k,j,ie+i+1) = b0.x3f(m,k,j,ie);
        if (k == n3-1) {b0.x3f(m,k+1,j,ie+i+1) = b0.x3f(m,k+1,j,ie);}
      }
    });

    par_for("outflow_bfield_x2", DevExeSpace(),0,(nmb-1),0,(n3-1),0,(ng-1),0,(n1-1),
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      if (mb_bcs.d_view(m,BoundaryFace::inner_x2) == BoundaryFlag::user) {
        b0.x1f(m,k,js-j-1,i) = b0.x1f(m,k,js,i);
        if (i == n1-1) {b0.x1f(m,k,js-j-1,i+1) = b0.x1f(m,k,js,i+1);}
        b0.x2f(m,k,js-j-1,i) = b0.x2f(m,k,js,i);
        b0.x3f(m,k,js-j-1,i) = b0.x3f(m,k,js,i);
        if (k == n3-1) {b0.x3f(m,k+1,js-j-1,i) = b0.x3f(m,k+1,js,i);}
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x2) == BoundaryFlag::user) {
        b0.x1f(m,k,je+j+1,i) = b0.x1f(m,k,je,i);
        if (i == n1-1) {b0.x1f(m,k,je+j+1,i+1) = b0.x1f(m,k,je,i+1);}
        b0.x2f(m,k,je+j+2,i) = b0.x2f(m,k,je+1,i);
        b0.x3f(m,k,je+j+1,i) = b0.x3f(m,k,je,i);
        if (k == n3-1) {b0.x3f(m,k+1,je+j+1,i) = b0.x3f(m,k+1,je,i);}
      }
    });

    par_for("outflow_bfield_x3", DevExeSpace(),0,(nmb-1),0,(ng-1),0,(n2-1),0,(n1-1),
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      if (mb_bcs.d_view(m,BoundaryFace::inner_x3) == BoundaryFlag::user) {
        b0.x1f(m,ks-k-1,j,i) = b0.x1f(m,ks,j,i);
        if (i == n1-1) {b0.x1f(m,ks-k-1,j,i+1) = b0.x1f(m,ks,j,i+1);}
        b0.x2f(m,ks-k-1,j,i) = b0.x2f(m,ks,j,i);
        if (j == n2-1) {b0.x2f(m,ks-k-1,j+1,i) = b0.x2f(m,ks,j+1,i);}
        b0.x3f(m,ks-k-1,j,i) = b0.x3f(m,ks,j,i);
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x3) == BoundaryFlag::user) {
        b0.x1f(m,ke+k+1,j,i) = b0.x1f(m,ke,j,i);
        if (i == n1-1) {b0.x1f(m,ke+k+1,j,i+1) = b0.x1f(m,ke,j,i+1);}
        b0.x2f(m,ke+k+1,j,i) = b0.x2f(m,ke,j,i);
        if (j == n2-1) {b0.x2f(m,ke+k+1,j+1,i) = b0.x2f(m,ke,j+1,i);}
        b0.x3f(m,ke+k+2,j,i) = b0.x3f(m,ke+1,j,i);
      }
    });
  }

  // fixed condition: do nothing
  // if (bc_flag == "fixed") {}
  // outlfow condition
  if (bc_flag == "outflow") {
    // ConsToPrim over all ghost zones *and* at the innermost/outermost X1-active zones
    // of Meshblocks, even if Meshblock face is not at the edge of computational domain
    if (pmbp->phydro != nullptr) {
      pmbp->phydro->peos->ConsToPrim(u0_,w0_,false,is-ng,is,0,(n2-1),0,(n3-1));
      pmbp->phydro->peos->ConsToPrim(u0_,w0_,false,ie,ie+ng,0,(n2-1),0,(n3-1));
      pmbp->phydro->peos->ConsToPrim(u0_,w0_,false,0,(n1-1),js-ng,js,0,(n3-1));
      pmbp->phydro->peos->ConsToPrim(u0_,w0_,false,0,(n1-1),je,je+ng,0,(n3-1));
      pmbp->phydro->peos->ConsToPrim(u0_,w0_,false,0,(n1-1),0,(n2-1),ks-ng,ks);
      pmbp->phydro->peos->ConsToPrim(u0_,w0_,false,0,(n1-1),0,(n2-1),ke,ke+ng);
    } else if (pmbp->pmhd != nullptr) {
      auto &b0 = pmbp->pmhd->b0;
      auto &bcc = pmbp->pmhd->bcc0;
      pmbp->pmhd->peos->ConsToPrim(u0_,b0,w0_,bcc,false,is-ng,is,0,(n2-1),0,(n3-1));
      pmbp->pmhd->peos->ConsToPrim(u0_,b0,w0_,bcc,false,ie,ie+ng,0,(n2-1),0,(n3-1));
      pmbp->pmhd->peos->ConsToPrim(u0_,b0,w0_,bcc,false,0,(n1-1),js-ng,js,0,(n3-1));
      pmbp->pmhd->peos->ConsToPrim(u0_,b0,w0_,bcc,false,0,(n1-1),je,je+ng,0,(n3-1));
      pmbp->pmhd->peos->ConsToPrim(u0_,b0,w0_,bcc,false,0,(n1-1),0,(n2-1),ks-ng,ks);
      pmbp->pmhd->peos->ConsToPrim(u0_,b0,w0_,bcc,false,0,(n1-1),0,(n2-1),ke,ke+ng);
    }
    // Set X1-BCs on w0 if Meshblock face is at the edge of computational domain
    par_for("outflow_hydro_x1", DevExeSpace(),0,(nmb-1),0,(nvar-1),0,(n3-1),0,(n2-1),
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
    // Set X2-BCs on w0 if Meshblock face is at the edge of computational domain
    par_for("outflow_hydro_x2", DevExeSpace(),0,(nmb-1),0,(nvar-1),0,(n3-1),0,(n1-1),
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
    // Set X3-BCs on w0 if Meshblock face is at the edge of computational domain
    par_for("outflow_hydro_x3", DevExeSpace(),0,(nmb-1),0,(nvar-1),0,(n2-1),0,(n1-1),
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
    // PrimToCons on X1, X2, X3 ghost zones
    if (pmbp->phydro != nullptr) {
      pmbp->phydro->peos->PrimToCons(w0_,u0_,is-ng,is-1,0,(n2-1),0,(n3-1));
      pmbp->phydro->peos->PrimToCons(w0_,u0_,ie+1,ie+ng,0,(n2-1),0,(n3-1));
      pmbp->phydro->peos->PrimToCons(w0_,u0_,0,(n1-1),js-ng,js-1,0,(n3-1));
      pmbp->phydro->peos->PrimToCons(w0_,u0_,0,(n1-1),je+1,je+ng,0,(n3-1));
      pmbp->phydro->peos->PrimToCons(w0_,u0_,0,(n1-1),0,(n2-1),ks-ng,ks-1);
      pmbp->phydro->peos->PrimToCons(w0_,u0_,0,(n1-1),0,(n2-1),ke+1,ke+ng);
    } else if (pmbp->pmhd != nullptr) {
      auto &bcc0_ = pmbp->pmhd->bcc0;
      pmbp->pmhd->peos->PrimToCons(w0_,bcc0_,u0_,is-ng,is-1,0,(n2-1),0,(n3-1));
      pmbp->pmhd->peos->PrimToCons(w0_,bcc0_,u0_,ie+1,ie+ng,0,(n2-1),0,(n3-1));
      pmbp->pmhd->peos->PrimToCons(w0_,bcc0_,u0_,0,(n1-1),js-ng,js-1,0,(n3-1));
      pmbp->pmhd->peos->PrimToCons(w0_,bcc0_,u0_,0,(n1-1),je+1,je+ng,0,(n3-1));
      pmbp->pmhd->peos->PrimToCons(w0_,bcc0_,u0_,0,(n1-1),0,(n2-1),ks-ng,ks-1);
      pmbp->pmhd->peos->PrimToCons(w0_,bcc0_,u0_,0,(n1-1),0,(n2-1),ke+1,ke+ng);
    }
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn AccRefine
//! \brief User-defined refinement condition(s)

void AccRefine(MeshBlockPack* pmbp) {
  // capture variables for kernels
  Mesh *pmesh = pmbp->pmesh;
  auto &size = pmbp->pmb->mb_size;
  auto &indcs = pmesh->mb_indcs;
  int &is = indcs.is, nx1 = indcs.nx1;
  int &js = indcs.js, nx2 = indcs.nx2;
  int &ks = indcs.ks, nx3 = indcs.nx3;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;

  // check (on device) Hydro/MHD refinement conditions over all MeshBlocks
  auto refine_flag_ = pmesh->pmr->refine_flag;
  Real &rad_thresh  = acc->r_refine;
  Real &dens_thresh = acc->dens_refine;
  int nmb = pmbp->nmb_thispack;
  int mbs = pmesh->gids_eachrank[global_variable::my_rank];
  if ((pmbp->phydro != nullptr) || (pmbp->pmhd != nullptr)) {
    if (global_variable::my_rank == 0) {printf("AccRefine\n");}
    auto &w0 = (pmbp->phydro != nullptr)? pmbp->phydro->w0 : pmbp->pmhd->w0;
    par_for_outer("AccRefineCond",DevExeSpace(), 0, 0, 0, (nmb-1),
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
      Real vol = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
      // density threshold
      if (dens_thresh!= 0.0) {
        Real team_dmax=0.0;
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tmember, nkji),
        [=](const int idx, Real& dmax) {
          int k = (idx)/nji;
          int j = (idx - k*nji)/nx1;
          int i = (idx - k*nji - j*nx1) + is;
          j += js;
          k += ks;
          Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
          Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
          Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
          Real rad = sqrt(SQR(x1v)+SQR(x2v)+SQR(x3v));
          dmax = fmax(w0(m,IDN,k,j,i)*vol/SQR(rad), dmax);
        },Kokkos::Max<Real>(team_dmax));
        if (team_dmax > dens_thresh) {
          //printf("dens refine: m=%d r_min=%.6e vol=%.6e team_dmax=%0.6e\n",
          //m+mbs, rad_min, vol, team_dmax);
          refine_flag_.d_view(m+mbs) = 1;
        }
        if (team_dmax < 0.1 * dens_thresh) {
          refine_flag_.d_view(m+mbs) = -1;
        }
      }
      if (rad_min < rad_thresh) {refine_flag_.d_view(m+mbs) = 1;}
      //if (rad_min > rad_thresh) {refine_flag_.d_view(m+mbs) = -1;}
    });
  }
}

//----------------------------------------------------------------------------------------
//! \fn AccHistOutput
//! \brief User-defined history output

void AccHistOutput(HistoryData *pdata, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  Real &rin = acc->r_in;
  Real &rbout = acc->rb_out;
  Real &radentry = acc->rad_entry;
  Real &k0 = acc->k0_entry;
  Real &xi = acc->xi_entry;
  auto &d_arr = acc->dens_arr;
  Real gamma = acc->gamma;
  Real gm1 = gamma - 1.0;
  Real t_cold = acc->t_cold;
  Real tf_hot = acc->tf_hot;
  int nvars; bool is_mhd = false;
  DvceArray5D<Real> w0_, bcc0_;
  if (pmbp->phydro != nullptr) {
    nvars = pmbp->phydro->nhydro + pmbp->phydro->nscalars;
    w0_ = pmbp->phydro->w0;
  } else if (pmbp->pmhd != nullptr) {
    is_mhd = true;
    nvars = pmbp->pmhd->nmhd + pmbp->pmhd->nscalars;
    w0_ = pmbp->pmhd->w0;
    bcc0_ = pmbp->pmhd->bcc0;
  }
  // extract grids, number of radii, number of fluxes, and history appending index
  auto &grids = pm->pgen->spherical_grids;
  int nradii = grids.size();
  //const int nflux = (is_mhd) ? 7 : 6;
  const int nflux = 8;
  // TODO(@mhguo): check what is necessary!
  // TODO(@mhguo): In the long run, it would be great if we can directly plot radial
  // TODO(@mhguo): profiles using history variables. We may need around 10 points and
  // TODO(@mhguo): span roughly 2 orders of magnitude: 1,2,3,5,10,20,30,50,100,200,300
  // TODO(@mhguo): or 1,1.5,2,3,5,7,10,15,20,30,50,70,100
  // set number of and names of history variables for hydro or mhd
  //  (0) mass
  //  (1) mass accretion rate
  //  (2) energy flux
  //  (3) angular momentum flux * 3
  //  (4) magnetic flux (iff MHD)
  const int nsph = 4 * nflux;
  const int nreduce = 58;
  const int nuser = nsph + nreduce;
  pdata->nhist = nuser;
  const char *data_label[nuser] = {
    // 6 letters for the first 7 labels, 5 for the rest
    "m_1   ", "mdot1 ", "mdh1  ", "edot1 ", "lx_1  ", "ly_1  ", "lz_1  ", "phi_1",
    "m_2  ",  "mdot2",  "mdh2 ",  "edot2",  "lx_2 ",  "ly_2 ",  "lz_2 ",  "phi_2",
    "m_3  ",  "mdot3",  "mdh3 ",  "edot3",  "lx_3 ",  "ly_3 ",  "lz_3 ",  "phi_3",
    "m_4  ",  "mdot4",  "mdh4 ",  "edot4",  "lx_4 ",  "ly_4 ",  "lz_4 ",  "phi_4",
    "Mdot ",  "Mglb ",  "M_in ",  "Lx_in",  "Ly_in",  "Lz_in",  "L_in ",
    "Vcold",  "Mcold",  "Lx_c ",  "Ly_c ",  "Lz_c ",  "L_c  ",
    "Vwarm",  "Mwarm",  "Lx_w ",  "Ly_w ",  "Lz_w ",  "L_w  ",
    "Mdotc",  "Vinc ",  "Minc ",  "Lx_ic",  "Ly_ic",  "Lz_ic",  "L_ic ",
    "Mdotw",  "Vinw ",  "Minw ",  "Lx_iw",  "Ly_iw",  "Lz_iw",  "L_iw ",
    "Mdoth",  "Vinh ",  "Minh ",  "Lx_ih",  "Ly_ih",  "Lz_ih",  "L_ih ",
    "V0c  ",  "M0c  ",  "V0w  ",  "M0w  ",  "V0h  ",  "M0h  ",
    "V1c  ",  "M1c  ",  "V1w  ",  "M1w  ",  "V1h  ",  "M1h  ",
    "V2c  ",  "M2c  ",  "V2w  ",  "M2w  ",  "V2h  ",  "M2h  ",
    //"pr_in", "pr_ic", "pr_iw", "pr_ih",
    //"Lx0c",  "Ly0c",  "Lz0c",  "L0c",
  };
  for (int n=0; n<nuser; ++n) {
    pdata->label[n] = data_label[n];
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

        // compute temperature and compare with hot/cold threshold
        Real int_temp = gm1*int_ie/int_dn;
        Real x = r/radentry;
        Real rho_ini = DensFn(x,d_arr);
        Real pgas_ini = 0.5*k0*(1.0+pow(x,xi))*pow(rho_ini,gamma);
        Real temp_ini = pgas_ini/rho_ini;
        Real t_hot = tf_hot*temp_ini;
        Real is_hot = (int_temp>=t_hot)? 1.0 : 0.0;

        // compute mass density
        pdata->hdata[nflux*g+0] += int_dn*sqrtmdet*domega;

        // compute mass flux
        pdata->hdata[nflux*g+1] += 1.0*int_dn*ur*sqrtmdet*domega;
        pdata->hdata[nflux*g+2] += is_hot*int_dn*ur*sqrtmdet*domega;

        // compute energy flux
        Real t1_0 = (int_dn + gamma*int_ie + b_sq)*ur*u_0 - br*b_0;
        pdata->hdata[nflux*g+3] += 1.0*t1_0*sqrtmdet*domega;

        // compute angular momentum flux
        // TODO(@mhguo): write a correct function to compute x,y angular momentum flux
        Real t1_1 = 0.0;
        Real t1_2 = 0.0;
        Real t1_3 = (int_dn + gamma*int_ie + b_sq)*ur*u_ph - br*b_ph;
        pdata->hdata[nflux*g+4] += t1_1*sqrtmdet*domega;
        pdata->hdata[nflux*g+5] += t1_2*sqrtmdet*domega;
        pdata->hdata[nflux*g+6] += t1_3*sqrtmdet*domega;

        // compute magnetic flux
        if (is_mhd) {
          pdata->hdata[nflux*g+7] += 0.5*fabs(br*u0 - b0*ur)*sqrtmdet*domega;
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
        Real &int_ie = grids[g]->interp_vals.h_view(n,IEN);
        Real int_temp = gm1*int_ie/int_dn;
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
        Real x = r/radentry;
        Real rho_ini = DensFn(x,d_arr);
        Real pgas_ini = 0.5*k0*(1.0+pow(x,xi))*pow(rho_ini,gamma);
        Real temp_ini = pgas_ini/rho_ini;
        Real t_hot = tf_hot*temp_ini;
        Real is_hot = (int_temp>=t_hot)? 1.0 : 0.0;

        // compute mass density
        pdata->hdata[nflux*g+0] += int_dn*r_sq*domega;

        // compute mass flux
        pdata->hdata[nflux*g+1] += 1.0*int_dn*vr*r_sq*domega;
        pdata->hdata[nflux*g+2] += is_hot*int_dn*vr*r_sq*domega;

        // compute energy flux
        // TODO(@mhguo): check whether this is correct!
        Real t1_0 = (int_ie + 0.5*e_k + 0.5*b_sq)*vr;
        pdata->hdata[nflux*g+3] += 1.0*t1_0*r_sq*domega;

        // compute angular momentum flux
        // TODO(@mhguo): check whether this is correct!
        pdata->hdata[nflux*g+4] += int_dn*(x2*v3-x3*v2)*r_sq*domega;
        pdata->hdata[nflux*g+5] += int_dn*(x3*v1-x1*v3)*r_sq*domega;
        pdata->hdata[nflux*g+6] += int_dn*(x1*v2-x2*v1)*r_sq*domega;

        // compute magnetic flux
        if (is_mhd) {
          pdata->hdata[nflux*g+7] += 0.5*fabs(br)*r_sq*domega;
        }
      }
    }
  }

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
  array_sum::GlobalSum sum_this_mb1;
  array_sum::GlobalSum sum_this_mb2;
  // store data into hdata array
  for (int n=0; n<NREDUCTION_VARIABLES; ++n) {
    sum_this_mb0.the_array[n] = 0.0;
    sum_this_mb1.the_array[n] = 0.0;
    sum_this_mb2.the_array[n] = 0.0;
  }
  Kokkos::parallel_reduce("AccHistSums",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, array_sum::GlobalSum &mb_sum0,
  array_sum::GlobalSum &mb_sum1, array_sum::GlobalSum &mb_sum2) {
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

    Real &dens = w0_(m,IDN,k,j,i);
    Real mom1 = dens*w0_(m,IVX,k,j,i);
    Real mom2 = dens*w0_(m,IVY,k,j,i);
    Real mom3 = dens*w0_(m,IVZ,k,j,i);

    Real temp = gm1 * w0_(m,IEN,k,j,i)/w0_(m,IDN,k,j,i);

    Real x = rad/radentry;
    Real rho_ini = DensFnDevice(x,d_arr);
    Real pgas_ini = 0.5*k0*(1.0+pow(x,xi))*pow(rho_ini,gamma);
    Real temp_ini = pgas_ini/rho_ini;
    Real t_hot = tf_hot*temp_ini;

    //Real coldgas = (temp<1e-2)? vol*dens : 0.0;

    Real mdot = 0.0;
    Real mdot_c = 0.0;
    Real mdot_w = 0.0;
    Real mdot_h = 0.0;
    if (rad>rin && rad<rin*1.2) {
      mdot = (mom1*x1v+mom2*x2v+mom3*x3v)/rad*vol/(rin*0.2);
      if (temp<t_cold) {
        mdot_c = mdot;
      } else if (temp<t_hot) {
        mdot_w = mdot;
      } else {
        mdot_h = mdot;
      }
    }

    Real dv_glb = (rad>rin && rad<rbout)? vol : 0.0;
    Real dm_glb = dv_glb*dens;

    Real dv_cold = (temp<t_cold)? vol : 0.0;
    Real dv_warm = (temp>=t_cold && temp<t_hot)? vol : 0.0;
    Real dv_hot = (temp>=t_hot)? vol : 0.0;
    Real dv_in = (rad>rin && rad<rin*1.2)? vol : 0.0;
    Real dv_r0 = (rad>rin && rad<1.0e-3)? vol : 0.0;
    Real dv_r1 = (rad>rin && rad<1.0e-2)? vol : 0.0;
    Real dv_r2 = (rad>rin && rad<1.0e-1)? vol : 0.0;

    Real dv_inc = fmin(dv_in,dv_cold);
    Real dv_inw = fmin(dv_in,dv_warm);
    Real dv_inh = fmin(dv_in,dv_hot);

    Real dv_r0c = fmin(dv_r0,dv_cold);
    Real dv_r0w = fmin(dv_r0,dv_warm);
    Real dv_r0h = fmin(dv_r0,dv_hot);

    Real dv_r1c = fmin(dv_r1,dv_cold);
    Real dv_r1w = fmin(dv_r1,dv_warm);
    Real dv_r1h = fmin(dv_r1,dv_hot);

    Real dv_r2c = fmin(dv_r2,dv_cold);
    Real dv_r2w = fmin(dv_r2,dv_warm);
    Real dv_r2h = fmin(dv_r2,dv_hot);

    Real dm_cold = dv_cold*dens;
    Real lx_c = dv_cold*(x2v*mom3 - x3v*mom2);
    Real ly_c = dv_cold*(x3v*mom1 - x1v*mom3);
    Real lz_c = dv_cold*(x1v*mom2 - x2v*mom1);
    Real l_c = sqrt(SQR(lx_c)+SQR(ly_c)+SQR(lz_c));

    Real dm_warm = dv_warm*dens;
    Real lx_w = dv_warm*(x2v*mom3 - x3v*mom2);
    Real ly_w = dv_warm*(x3v*mom1 - x1v*mom3);
    Real lz_w = dv_warm*(x1v*mom2 - x2v*mom1);
    Real l_w = sqrt(SQR(lx_w)+SQR(ly_w)+SQR(lz_w));

    Real dm_in = dv_in*dens;
    Real lx_in = dv_in*(x2v*mom3 - x3v*mom2);
    Real ly_in = dv_in*(x3v*mom1 - x1v*mom3);
    Real lz_in = dv_in*(x1v*mom2 - x2v*mom1);
    Real l_in = sqrt(SQR(lx_in)+SQR(ly_in)+SQR(lz_in));

    Real dm_inc = dv_inc*dens;
    Real lx_inc = dv_inc*(x2v*mom3 - x3v*mom2);
    Real ly_inc = dv_inc*(x3v*mom1 - x1v*mom3);
    Real lz_inc = dv_inc*(x1v*mom2 - x2v*mom1);
    Real l_inc = sqrt(SQR(lx_inc)+SQR(ly_inc)+SQR(lz_inc));

    Real dm_inw = dv_inw*dens;
    Real lx_inw = dv_inw*(x2v*mom3 - x3v*mom2);
    Real ly_inw = dv_inw*(x3v*mom1 - x1v*mom3);
    Real lz_inw = dv_inw*(x1v*mom2 - x2v*mom1);
    Real l_inw = sqrt(SQR(lx_inw)+SQR(ly_inw)+SQR(lz_inw));

    Real dm_inh = dv_inh*dens;
    Real lx_inh = dv_inh*(x2v*mom3 - x3v*mom2);
    Real ly_inh = dv_inh*(x3v*mom1 - x1v*mom3);
    Real lz_inh = dv_inh*(x1v*mom2 - x2v*mom1);
    Real l_inh = sqrt(SQR(lx_inh)+SQR(ly_inh)+SQR(lz_inh));

    Real dm_r0c = dv_r0c*dens;
    Real dm_r0w = dv_r0w*dens;
    Real dm_r0h = dv_r0h*dens;

    Real dm_r1c = dv_r1c*dens;
    Real dm_r1w = dv_r1w*dens;
    Real dm_r1h = dv_r1h*dens;

    Real dm_r2c = dv_r2c*dens;
    Real dm_r2w = dv_r2w*dens;
    Real dm_r2h = dv_r2h*dens;

    Real vars[nreduce] = {
      mdot,    dm_glb,  dm_in,   lx_in,   ly_in,   lz_in,   l_in,
      dv_cold, dm_cold, lx_c,    ly_c,    lz_c,    l_c,
      dv_warm, dm_warm, lx_w,    ly_w,    lz_w,    l_w,
      mdot_c,  dv_inc,  dm_inc,  lx_inc,  ly_inc,  lz_inc,  l_inc,
      mdot_w,  dv_inw,  dm_inw,  lx_inw,  ly_inw,  lz_inw,  l_inw,
      mdot_h,  dv_inh,  dm_inh,  lx_inh,  ly_inh,  lz_inh,  l_inh,
      dv_r0c,  dm_r0c,  dv_r0w,  dm_r0w,  dv_r0h,  dm_r0h,
      dv_r1c,  dm_r1c,  dv_r1w,  dm_r1w,  dv_r1h,  dm_r1h,
      dv_r2c,  dm_r2c,  dv_r2w,  dm_r2w,  dv_r2h,  dm_r2h,
    };

    // Hydro conserved variables:
    array_sum::GlobalSum hvars0;
    array_sum::GlobalSum hvars1;
    array_sum::GlobalSum hvars2;
    for (int n=0; n<NREDUCTION_VARIABLES; ++n) {
      hvars0.the_array[n] = vars[n];
      hvars1.the_array[n] = vars[n+NREDUCTION_VARIABLES];
    }
    for (int n=0; n<nreduce-2*NREDUCTION_VARIABLES; ++n) {
      hvars2.the_array[n] = vars[n+2*NREDUCTION_VARIABLES];
    }

    // fill rest of the_array with zeros, if nhist < NHISTORY_VARIABLES
    for (int n=nreduce-2*NREDUCTION_VARIABLES; n<NREDUCTION_VARIABLES; ++n) {
      hvars2.the_array[n] = 0.0;
    }

    // sum into parallel reduce
    mb_sum0 += hvars0;
    mb_sum1 += hvars1;
    mb_sum2 += hvars2;
  }, Kokkos::Sum<array_sum::GlobalSum>(sum_this_mb0),
     Kokkos::Sum<array_sum::GlobalSum>(sum_this_mb1),
     Kokkos::Sum<array_sum::GlobalSum>(sum_this_mb2));

  // store data into hdata array
  for (int n=0; n<NREDUCTION_VARIABLES; ++n) {
    pdata->hdata[nsph+n] = sum_this_mb0.the_array[n];
    pdata->hdata[nsph+NREDUCTION_VARIABLES+n] = sum_this_mb1.the_array[n];
  }
  for (int n=0; n<nreduce-2*NREDUCTION_VARIABLES; ++n) {
    pdata->hdata[nsph+2*NREDUCTION_VARIABLES+n] = sum_this_mb2.the_array[n];
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
  const DvceArray5D<Real> &w0 = (pmbp->pmhd != nullptr) ?
                                pmbp->pmhd->w0 : pmbp->phydro->w0;
  const EOS_Data &eos_data = (pmbp->pmhd != nullptr) ?
                             pmbp->pmhd->peos->eos_data : pmbp->phydro->peos->eos_data;
  if (acc->ndiag>0 && pm->ncycle % acc->ndiag == 0) {
    Diagnostic(pm,w0,eos_data);
  }
  if (acc->potential) {
    //std::cout << "AddAccel" << std::endl;
    AddAccel(pm,bdt,u0,w0);
  }
  if (acc->cooling) {
    //std::cout << "AddISMCooling" << std::endl;
    AddISMCooling(pm,bdt,u0,w0,eos_data);
  }
  if (pm->time >= acc->heat_beg_time) {
    if (acc->heating_ini) {
      //std::cout << "AddIniHeating" << std::endl;
      AddIniHeating(pm,bdt,u0,w0,eos_data);
    }
    if (acc->heating_ana) {
      //std::cout << "AddAnaHeating" << std::endl;
      AddAnaHeating(pm,bdt,u0,w0,eos_data);
    }
    if (acc->heating_equ) {
      //std::cout << "AddEquHeating" << std::endl;
      AddEquHeating(pm,bdt,u0,w0,eos_data);
    }
    if (acc->heating_mdot) {
      //std::cout << "AddMdotHeating" << std::endl;
      AddMdotHeating(pm,bdt,u0,w0,eos_data);
    }
    if (acc->heating_pow) {
      //std::cout << "AddPowHeating" << std::endl;
      AddPowHeating(pm,bdt,u0,w0);
    }
  }
  // (@mhguo) change inner radius, if necessary
  if (acc->r_in_new<acc->r_in_old) {
    Real t_now = pm->time-acc->r_in_beg_t;
    if (t_now<=0.0) {
      acc->r_in = acc->r_in_old;
    } else if (t_now<acc->r_in_sof_t) {
      Real t_ratio = (pm->time-acc->r_in_beg_t)/acc->r_in_sof_t;
      Real &rino = acc->r_in_old;
      Real &rinn = acc->r_in_new;
      //acc->r_in = std::pow(rinn,t_ratio)*std::pow(rino,(1.0-t_ratio));
      acc->r_in = rinn*t_ratio+rino*(1.0-t_ratio);
    } else {
      acc->r_in = acc->r_in_new;
    }
    // update problem-specific parameters
    if (pmbp->phydro != nullptr) {
      pmbp->phydro->peos->eos_data.r_in = acc->r_in;
    }
    if (pmbp->pmhd != nullptr) {
      pmbp->pmhd->peos->eos_data.r_in = acc->r_in;
    }
    if (global_variable::my_rank == 0 && pm->ncycle%acc->ndiag==0) {
      std::cout << " r_in=" << acc->r_in << std::endl;
    }
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
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmbp->nmb_thispack - 1;
  auto size = pmbp->pmb->mb_size;
  Real &mbh = acc->m_bh;
  Real &mstar = acc->m_star;
  Real &rstar = acc->r_star;
  Real &mdm = acc->m_dm;
  Real &rdm = acc->r_dm;
  Real &rin = acc->r_in;
  Real grav = pmbp->punit->grav_constant();

  par_for("accel", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    int nx1 = indcs.nx1;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    int nx2 = indcs.nx2;
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    int nx3 = indcs.nx3;
    Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

    Real rad = sqrt(SQR(x1v)+SQR(x2v)+SQR(x3v));

    Real accel = Acceleration(rad,mbh,mstar,rstar,mdm,rdm,grav);
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
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void SourceTerms::AddIniHeating()
//! \brief Add heating source terms in the energy equations.
// NOTE source terms must all be computed using primitive (w0) and NOT conserved (u0) vars

void AddIniHeating(Mesh *pm, const Real bdt, DvceArray5D<Real> &u0,
                const DvceArray5D<Real> &w0, const EOS_Data &eos_data) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pmbp->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;
  int nmb1 = pmbp->nmb_thispack - 1;
  Real gamma = eos_data.gamma;
  Real temp_unit = pmbp->punit->temperature_cgs();
  Real n_h_unit = pmbp->punit->density_cgs()/acc->mu_h/pmbp->punit->atomic_mass_unit_cgs;
  Real cooling_unit = pmbp->punit->pressure_cgs()/pmbp->punit->time_cgs()/SQR(n_h_unit);

  Real &radentry = acc->rad_entry;
  Real &k0 = acc->k0_entry;
  Real &xi = acc->xi_entry;
  auto &d_arr = acc->dens_arr;
  Real fac_heat = acc->fac_heat;
  if (pm->time-acc->heat_beg_time < acc->heat_sof_time) {
    fac_heat *= SQR(sin(0.5*M_PI*(pm->time-acc->heat_beg_time)/acc->heat_sof_time));
  }

  par_for("add_heating", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    int nx1 = indcs.nx1;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    int nx2 = indcs.nx2;
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    int nx3 = indcs.nx3;
    Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

    Real rad = sqrt(SQR(x1v)+SQR(x2v)+SQR(x3v));

    Real x = rad/radentry;
    Real rho = DensFnDevice(x,d_arr);
    Real pgas = 0.5*k0*(1.0+pow(x,xi))*pow(rho,gamma);
    // temperature in cgs unit
    Real temp = temp_unit*pgas/rho;

    Real lambda_cooling = ISMCoolFn(temp)/cooling_unit;
    Real gamma_heating = fac_heat * rho * lambda_cooling;

    u0(m,IEN,k,j,i) += bdt * w0(m,IDN,k,j,i) * gamma_heating;
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
  Real beta = bdt/pm->dt;
  Real cfl_no = pm->cfl_no;
  auto &eos = eos_data;
  Real use_e = eos_data.use_e;
  Real tfloor = eos_data.tfloor;
  Real gamma = eos_data.gamma;
  Real gm1 = gamma - 1.0;
  Real temp_unit = pmbp->punit->temperature_cgs();
  Real n_h_unit = pmbp->punit->density_cgs()/acc->mu_h/pmbp->punit->atomic_mass_unit_cgs;
  Real cooling_unit = pmbp->punit->pressure_cgs()/pmbp->punit->time_cgs()/SQR(n_h_unit);
  // Real gamma_heating = 2.0e-26/heating_unit; // add a small heating
  bool is_gr = pmbp->pcoord->is_general_relativistic;

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
    Real dens=1.0, temp = 1.0, eint = 1.0;
    dens = w0(m,IDN,k,j,i);
    if (use_e) {
      temp = w0(m,IEN,k,j,i)/w0(m,IDN,k,j,i)*gm1;
      eint = w0(m,IEN,k,j,i);
    } else {
      temp = w0(m,ITM,k,j,i);
      eint = w0(m,ITM,k,j,i)*w0(m,IDN,k,j,i)/gm1;
    }

    bool sub_cycling = true;
    bool sub_cycling_used = false;
    Real bdt_now = 0.0;
    if (is_gr) sub_cycling = false;
    do {
      Real lambda_cooling = ISMCoolFn(temp*temp_unit)/cooling_unit;
      // soft function
      lambda_cooling *= exp(-50.0*pow(tfloor/temp,4.0));
      Real cooling_heating = dens * dens * lambda_cooling;
      Real dt_cool = (eint/(FLT_MIN + fabs(cooling_heating)));
      // half of the timestep
      Real bdt_cool = 0.5*beta*cfl_no*dt_cool;
      if (bdt_now+bdt_cool<bdt) {
        u0(m,IEN,k,j,i) -= bdt_cool * cooling_heating;

        // compute new temperature and internal energy

        // load single state conserved variables
        HydPrim1D w;
        if (is_hydro) {
          HydCons1D u;
          u.d  = u0(m,IDN,k,j,i);
          u.mx = u0(m,IM1,k,j,i);
          u.my = u0(m,IM2,k,j,i);
          u.mz = u0(m,IM3,k,j,i);
          u.e  = u0(m,IEN,k,j,i);
          // call c2p function
          bool dfloor_used=false, efloor_used=false, tfloor_used=false;
          SingleC2P_IdealHyd(u, eos, w, dfloor_used, efloor_used, tfloor_used);
        } else {
          MHDCons1D u;
          u.d  = u0(m,IDN,k,j,i);
          u.mx = u0(m,IM1,k,j,i);
          u.my = u0(m,IM2,k,j,i);
          u.mz = u0(m,IM3,k,j,i);
          u.e  = u0(m,IEN,k,j,i);
          u.bx = bcc(m,IBX,k,j,i);
          u.by = bcc(m,IBY,k,j,i);
          u.bz = bcc(m,IBZ,k,j,i);
          // call c2p function
          bool dfloor_used=false, efloor_used=false, tfloor_used=false;
          SingleC2P_IdealMHD(u, eos, w, dfloor_used, efloor_used, tfloor_used);
        }

        dens = w.d;
        temp = gm1*w.e/w.d;
        eint = w.e;
        sub_cycling_used = true;
        sum1++;
      } else {
        u0(m,IEN,k,j,i) -= (bdt-bdt_now) * cooling_heating;
        sub_cycling = false;
      }
      bdt_now += bdt_cool;
    } while (sub_cycling);
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
    if (acc->ndiag>0 && pm->ncycle % acc->ndiag == 0) {
      if (nsubcycle>0 || nsubcycle_count >0) {
        std::cout << " nsubcycle_cell=" << nsubcycle << std::endl
                  << " nsubcycle_count=" << nsubcycle_count << std::endl;
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void SourceTerms::AddAnaHeating()
//! \brief Add an analytical heating source terms in the energy equations.
// NOTE source terms must all be computed using primitive (w0) and NOT conserved (u0) vars

void AddAnaHeating(Mesh *pm, const Real bdt, DvceArray5D<Real> &u0,
                   const DvceArray5D<Real> &w0, const EOS_Data &eos_data) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pmbp->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie; int nx1 = indcs.nx1;
  int js = indcs.js, je = indcs.je; int nx2 = indcs.nx2;
  int ks = indcs.ks, ke = indcs.ke; int nx3 = indcs.nx3;
  auto &size = pmbp->pmb->mb_size;
  int nmb1 = pmbp->nmb_thispack - 1;
  const int nmkji = (pmbp->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;
  Real use_e = eos_data.use_e;
  Real tfloor = eos_data.tfloor;
  Real gamma = eos_data.gamma;
  Real gm1 = gamma - 1.0;
  Real radpow = acc->radpow_heat;
  Real temp_unit = pmbp->punit->temperature_cgs();
  Real n_h_unit = pmbp->punit->density_cgs()/acc->mu_h/pmbp->punit->atomic_mass_unit_cgs;
  Real cooling_unit = pmbp->punit->pressure_cgs()/pmbp->punit->time_cgs()/SQR(n_h_unit);

  Real s0 = 0.0, s1 = 0.0;
  Kokkos::parallel_reduce("sum_cooling", Kokkos::RangePolicy<>(DevExeSpace(),0,nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &sum_s0, Real &sum_s1) {
    // compute n,k,j,i indices of thread
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;

    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    int nx1 = indcs.nx1;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    int nx2 = indcs.nx2;
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    int nx3 = indcs.nx3;
    Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

    Real rad = sqrt(SQR(x1v)+SQR(x2v)+SQR(x3v));

    Real vol = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;

    // temperature in cgs unit
    Real temp = 1.0;
    Real eint = 1.0;
    if (use_e) {
      temp = temp_unit*w0(m,IEN,k,j,i)/w0(m,IDN,k,j,i)*gm1;
      eint = w0(m,IEN,k,j,i);
    } else {
      temp = temp_unit*w0(m,ITM,k,j,i);
      eint = w0(m,ITM,k,j,i)*w0(m,IDN,k,j,i)/gm1;
    }

    Real hnorm = w0(m,IDN,k,j,i)*pow(rad,radpow);
    Real lambda_cooling = ISMCoolFn(temp)/cooling_unit;
    Real q_cooling = w0(m,IDN,k,j,i)*w0(m,IDN,k,j,i)*lambda_cooling;
    Real dens = w0(m,IDN,k,j,i);
    if (gm1*(eint-q_cooling*bdt)/dens<tfloor) {
      q_cooling = (eint-dens*tfloor/gm1)/bdt;
    }
    if (temp/temp_unit<=tfloor) {
      q_cooling = 0.0;
    }

    sum_s0 += vol*hnorm;
    sum_s1 += vol*q_cooling;
  }, Kokkos::Sum<Real>(s0), Kokkos::Sum<Real>(s1));

#if MPI_PARALLEL_ENABLED
  Real s_arr[2] = {s0,s1};
  Real gs_arr[2] = {0.0,0.0};
  MPI_Allreduce(s_arr, gs_arr, 2, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  s0 = gs_arr[0];
  s1 = gs_arr[1];
#endif

  Real hnorm = s1/s0;

  if (global_variable::my_rank == 0 && pmbp->pmesh->ncycle % 10 == 0) {
    std::cout << " s0=" << s0 << " s1=" << s1 << " hnorm=" << hnorm << std::endl;
  }

  par_for("add_ana_heating", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    int nx1 = indcs.nx1;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    int nx2 = indcs.nx2;
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    int nx3 = indcs.nx3;
    Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

    Real rad = sqrt(SQR(x1v)+SQR(x2v)+SQR(x3v));

    Real gamma_heating = hnorm*pow(rad,radpow);

    u0(m,IEN,k,j,i) += bdt * w0(m,IDN,k,j,i) * gamma_heating;
  });
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void SourceTerms::AddEquHeating()
//! \brief Add heating source terms in the energy equations.
// NOTE source terms must all be computed using primitive (w0) and NOT conserved (u0) vars

void AddEquHeating(Mesh *pm, const Real bdt, DvceArray5D<Real> &u0,
                   const DvceArray5D<Real> &w0, const EOS_Data &eos_data) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pmbp->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie; int nx1 = indcs.nx1;
  int js = indcs.js, je = indcs.je; int nx2 = indcs.nx2;
  int ks = indcs.ks, ke = indcs.ke; int nx3 = indcs.nx3;
  auto &size = pmbp->pmb->mb_size;
  int nmb1 = pmbp->nmb_thispack - 1;
  const int nmkji = (pmbp->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;
  Real use_e = eos_data.use_e;
  Real tfloor = eos_data.tfloor;
  Real gm1 = eos_data.gamma - 1.0;
  Real temp_unit = pmbp->punit->temperature_cgs();
  Real n_h_unit = pmbp->punit->density_cgs()/acc->mu_h/pmbp->punit->atomic_mass_unit_cgs;
  Real cooling_unit = pmbp->punit->pressure_cgs()/pmbp->punit->time_cgs()/SQR(n_h_unit);

  Real rin = acc->r_in;
  Real rmin_heat = acc->rmin_heat;
  Real rmax_heat = acc->rmax_heat;
  int weight = acc->heat_weight;
  int bins = acc->bins_heat;
  Real logr0 = acc->logr_heat;
  Real logh = acc->logh_heat;
  auto &v_arr = acc->v_arr;
  auto &c_arr = acc->c_arr;
  auto &logcoolarr = acc->logcooling_arr;
  if (pm->ncycle % acc->heat_cycle == 0) {
    for ( int i = 0; i < acc->bins_heat; i++ ) {
      v_arr.the_array[i] = 0.0;
      c_arr.the_array[i] = 0.0;
      logcoolarr.h_view(i) = 0.0;
    }

    Kokkos::parallel_reduce("sum_radial",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, array_acc::RadSum &varr, array_acc::RadSum &carr) {
      // compute n,k,j,i indices of thread
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks;
      j += js;

      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      int nx1 = indcs.nx1;
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      int nx2 = indcs.nx2;
      Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      int nx3 = indcs.nx3;
      Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

      Real rad = sqrt(SQR(x1v)+SQR(x2v)+SQR(x3v));
      if (rmin_heat <= rad && rad <= rmax_heat) {
        int ipps = GetRadialIndex(rad,logr0,logh,bins);
        Real vol = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;

        // temperature in code unit
        Real w_dens = w0(m,IDN,k,j,i);
        Real w_temp = 1.0;
        if (use_e) {
          w_temp = w0(m,IEN,k,j,i)/w_dens*gm1;
        } else {
          w_temp = w0(m,ITM,k,j,i);
        }

        Real lambda_cooling = ISMCoolFn(temp_unit*w_temp)/cooling_unit;
        // soft function
        lambda_cooling *= exp(-50.0*pow(tfloor/w_temp,4.0));
        Real q_cooling = w_dens*w_dens*lambda_cooling;
        if (w_temp<=tfloor*10.0) {
          q_cooling = 0.0;
        }
        q_cooling = fmax(q_cooling,0.0);
        //if (m==19 && k==4 && j==4 && i==4) {
        //  printf("sum_radial: i=%d rad=%0.6e vol=%0.6e dens=%0.6e temp=%0.6e \
        //        q_cooling=%0.6e\n", ipps,rad,vol,w0(m,IDN,k,j,i),temp,q_cooling);
        //}
        // (@mhguo) tmpcout!
        Real dvar = (weight == 0)? vol : vol*w_dens;
        varr.the_array[ipps] += dvar;
        carr.the_array[ipps] += vol*q_cooling;
      }
    }, Kokkos::Sum<array_acc::RadSum>(v_arr), Kokkos::Sum<array_acc::RadSum>(c_arr));

    Real tmpfac = acc->fac_heat;
    if (pm->time-acc->heat_beg_time < acc->heat_sof_time) {
      tmpfac *= SQR(sin(0.5*M_PI*(pm->time-acc->heat_beg_time)/acc->heat_sof_time));
    }
    #if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, &(v_arr.the_array[0]), bins, MPI_ATHENA_REAL, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &(c_arr.the_array[0]), bins, MPI_ATHENA_REAL, MPI_SUM,
                  MPI_COMM_WORLD);
    #endif
    for ( int i = 0; i < bins; i++ ) {
      if (v_arr.the_array[i]>0.0) {
        c_arr.the_array[i] *= tmpfac/v_arr.the_array[i]; // a factor
        if (c_arr.the_array[i]>0.0) {
          logcoolarr.h_view(i) = std::log10(c_arr.the_array[i]);
        } else {
          logcoolarr.h_view(i) = -100.0; //1e-100
        }
        // TODO(@mhguo): tmpcout!
        //if (global_variable::my_rank == 0){
        //  printf("equheat i=%d c=%.4e logc=%.4e\n",
        //  i,c_arr.the_array[i],logcoolarr.h_view(i));
        //}
      }
    }

    logcoolarr.template modify<HostMemSpace>();
    logcoolarr.template sync<DevExeSpace>();
  }

  //if (global_variable::my_rank == 0){
  //  for ( int i = 0; i < bins; i++ ) {
  //    std::cout << "i=" << i << "  r="
  //    << std::pow(10.0,i*acc->logh_heat+acc->logr_heat)
  //    << "  cool=" << c_arr.the_array[i] << std::endl;
  //  }
  //}

  par_for("add_equ_heating", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    int nx1 = indcs.nx1;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    int nx2 = indcs.nx2;
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    int nx3 = indcs.nx3;
    Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

    Real rad = sqrt(SQR(x1v)+SQR(x2v)+SQR(x3v));

    if (rin <= rad && rmin_heat <= rad && rad <= rmax_heat) {
      Real var = GetRadialVar(logcoolarr,rad,logr0,logh,bins);
      Real q_heating = (weight == 0)? var : var*w0(m,IDN,k,j,i);
      // soft function
      Real w_dens = w0(m,IDN,k,j,i);
      Real w_temp = 1.0;
      if (use_e) {
        w_temp = w0(m,IEN,k,j,i)/w_dens*gm1;
      } else {
        w_temp = w0(m,ITM,k,j,i);
      }
      // add soft radius
      if (rad < 3.0*rmin_heat) {
        q_heating *= SQR(sin(0.25*M_PI*(rad/rmin_heat-1.0)));
      }
      // add soft ceiling
      // q_heating *= exp(-1.0e1*pow(w_temp/2.0e1,4.0));
      // Stop heating when T>1 in code unit (T>7e7K in cgs)
      q_heating *= exp(-1.0e1*pow(w_temp/2.0e0,4.0));
      //if (m==19 && k==4 && j==4 && i==4) {
      //  printf("q_heating: rad=%0.6e q_heating=%0.6e\n",rad,q_heating);
      //}
      u0(m,IEN,k,j,i) += bdt * q_heating;
    }
  });
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void SourceTerms::AddPowHeating()
//! \brief Add an analytical heating source terms in the energy equations.
// NOTE source terms must all be computed using primitive (w0) and NOT conserved (u0) vars

void AddPowHeating(Mesh *pm, const Real bdt, DvceArray5D<Real> &u0,
                   const DvceArray5D<Real> &w0) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pmbp->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;
  int nmb1 = pmbp->nmb_thispack - 1;
  Real radh = acc->rad_heat;
  Real radpow = acc->radpow_heat;
  Real hnorm = acc->heatnorm;

  par_for("add_pow_heating", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    int nx1 = indcs.nx1;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    int nx2 = indcs.nx2;
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    int nx3 = indcs.nx3;
    Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

    Real rad = sqrt(SQR(x1v)+SQR(x2v)+SQR(x3v));

    //Real gamma_heating = 2e-4*pow(rad,-1.5) + hnorm*pow(rad+radh,radpow);
    Real gamma_heating = hnorm*pow(rad+radh, radpow);

    u0(m,IEN,k,j,i) += bdt * w0(m,IDN,k,j,i) * gamma_heating;
  });
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void SourceTerms::AddMdotHeating()
//! \brief Add heating source terms in the energy equations.
// NOTE source terms must all be computed using primitive (w0) and NOT conserved (u0) vars

void AddMdotHeating(Mesh *pm, const Real bdt, DvceArray5D<Real> &u0,
                const DvceArray5D<Real> &w0, const EOS_Data &eos_data) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pmbp->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie; int nx1 = indcs.nx1;
  int js = indcs.js, je = indcs.je; int nx2 = indcs.nx2;
  int ks = indcs.ks, ke = indcs.ke; int nx3 = indcs.nx3;
  auto &size = pmbp->pmb->mb_size;
  int nmb1 = pmbp->nmb_thispack - 1;
  const int nmkji = (pmbp->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;

  Real &rin = acc->r_in;
  Real dr = acc->mdot_dr;
  Real use_e = eos_data.use_e;
  Real heat_tceiling = acc->heat_tceiling;
  Real radh = acc->rad_heat;
  Real radpow = acc->radpow_heat;
  Real gm1 = eos_data.gamma - 1.0;

  Real s0 = 0.0, s1 = 0.0;
  Kokkos::parallel_reduce("sum_mdot", Kokkos::RangePolicy<>(DevExeSpace(),0,nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &sum_s0, Real &sum_s1) {
    // compute n,k,j,i indices of thread
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;

    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    int nx1 = indcs.nx1;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    int nx2 = indcs.nx2;
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    int nx3 = indcs.nx3;
    Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

    Real rad = sqrt(SQR(x1v)+SQR(x2v)+SQR(x3v));

    Real vol = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;

    if (rad > rin && rad < rin+dr) {
      Real dens = w0(m,IDN,k,j,i);
      Real velr = -(w0(m,IVX,k,j,i)*x1v+w0(m,IVY,k,j,i)*x2v+w0(m,IVZ,k,j,i)*x3v)/rad;
      sum_s0 += vol;
      sum_s1 += vol*dens*velr;
    }
  }, Kokkos::Sum<Real>(s0), Kokkos::Sum<Real>(s1));

#if MPI_PARALLEL_ENABLED
  Real s_arr[2] = {s0,s1};
  Real gs_arr[2] = {0.0,0.0};
  MPI_Allreduce(s_arr, gs_arr, 2, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  s0 = gs_arr[0];
  s1 = gs_arr[1];
#endif

  s1 = std::max(s1, 0.0);
  Real mdot = s1/dr;
  Real epsilon = acc->epsilon;
  Real hnorm = epsilon * mdot * SQR(pmbp->punit->speed_of_light());

  if (global_variable::my_rank == 0 && pmbp->pmesh->ncycle % 10 == 0) {
    std::cout << " mdot=" << mdot << " hnorm=" << hnorm << std::endl;
  }

  par_for("add_mdot_heating", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    int nx1 = indcs.nx1;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    int nx2 = indcs.nx2;
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    int nx3 = indcs.nx3;
    Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

    Real rad = sqrt(SQR(x1v)+SQR(x2v)+SQR(x3v));

    Real gamma_heating = hnorm * pow(rad+radh, radpow);

    // Apply temperature ceiling
    Real dens = w0(m,IDN,k,j,i);
    Real eint = 1.0;
    if (use_e) {
      eint = w0(m,IEN,k,j,i);
    } else {
      eint = w0(m,ITM,k,j,i)*w0(m,IDN,k,j,i)/gm1;
    }

    if (gm1*(eint/dens+gamma_heating*bdt)>heat_tceiling) {
      gamma_heating = fmax((heat_tceiling/gm1-eint/dens)/bdt,0.0);
    }

    u0(m,IEN,k,j,i) += bdt * w0(m,IDN,k,j,i) * gamma_heating;
  });
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void SourceTerms::ISMCoolingNewTimeStep()
//! \brief Compute new time step for ISM cooling.
// NOTE source terms must all be computed using primitive (w0) and NOT conserved (u0) vars

void Diagnostic(Mesh *pm, const DvceArray5D<Real> &w0, const EOS_Data &eos_data) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pmbp->pmesh->mb_indcs;
  int is = indcs.is; int nx1 = indcs.nx1;
  int js = indcs.js; int nx2 = indcs.nx2;
  int ks = indcs.ks; int nx3 = indcs.nx3;
  auto &size = pmbp->pmb->mb_size;
  const int nmkji = (pmbp->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;

  Real use_e = eos_data.use_e;
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

  // find smallest (e/cooling_rate) in each cell
  Kokkos::parallel_reduce("cooling_newdt", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
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
    Real temp = 1.0;
    Real eint = 1.0;
    if (use_e) {
      temp = w0(m,IEN,k,j,i)/w0(m,IDN,k,j,i)*gm1;
      eint = w0(m,IEN,k,j,i);
    } else {
      temp = w0(m,ITM,k,j,i);
      eint = w0(m,ITM,k,j,i)*w0(m,IDN,k,j,i)/gm1;
    }

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
  }, Kokkos::Min<Real>(dtnew),
     Kokkos::Min<Real>(min_dens),
     Kokkos::Min<Real>(min_vtot),
     Kokkos::Min<Real>(min_temp),
     Kokkos::Min<Real>(min_eint),
     Kokkos::Max<Real>(max_dens),
     Kokkos::Max<Real>(max_vtot),
     Kokkos::Max<Real>(max_temp),
     Kokkos::Max<Real>(max_eint));
#if MPI_PARALLEL_ENABLED
  Real m_min[5] = {dtnew,min_dens,min_vtot,min_temp,min_eint};
  Real m_max[4] = {max_dens,max_vtot,max_temp,max_eint};
  Real gm_min[5];
  Real gm_max[4];
  //MPI_Allreduce(MPI_IN_PLACE, &dtnew, 1, MPI_ATHENA_REAL, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(m_min, gm_min, 5, MPI_ATHENA_REAL, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(m_max, gm_max, 4, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
  dtnew = gm_min[0];
  min_dens = gm_min[1];
  min_vtot = gm_min[2];
  min_temp = gm_min[3];
  min_eint = gm_min[4];
  max_dens = gm_max[0];
  max_vtot = gm_max[1];
  max_temp = gm_max[2];
  max_eint = gm_max[3];
#endif
  if (global_variable::my_rank == 0) {
    std::cout << " min_d=" << min_dens << " max_d=" << max_dens << std::endl
              << " min_v=" << min_vtot << " max_v=" << max_vtot << std::endl
              << " min_t=" << min_temp << " max_t=" << max_temp << std::endl
              << " min_e=" << min_eint << " max_e=" << max_eint << std::endl
              << " dt_cs=" << dtnew << std::endl;
  }
  return;
}

//---------------------------------------------------------------------------------------
//! \fn RK4
//! \brief 4th order runge kutta with method for calculating density

static Real RK4(RK4FnPtr func, Real x, Real y, Real h) {
  Real k1,k2,k3,k4;

  k1=func(x,y);
  x=x+0.5*h;
  k2=func(x,(y+0.5*k1*h));
  k3=func(x,(y+0.5*k2*h));
  x=x+0.5*h;
  k4=func(x,y+k3*h);
  y=y+(1./6.)*(k1+2.*k2+2.*k3+k4)*h;

  return y;
}

//-----------------------------------------------------------------------------------------
//! \fn DrhoDr()
//! \brief calculates the gradient of the density to be used in the RK4

static Real DrhoDr(Real x, Real rho) {
  Real &gamma = acc->gamma;
  Real &grav = acc->grav;
  Real &mbh = acc->m_bh;
  Real &mstar = acc->m_star;
  Real &rstar = acc->r_star;
  Real &mdm = acc->m_dm;
  Real &rdm = acc->r_dm;
  Real &k0 = acc->k0_entry;
  Real &xi = acc->xi_entry;
  Real &radentry = acc->rad_entry;
  Real r = x*radentry;

  Real accel = radentry*Acceleration(r,mbh,mstar,rstar,mdm,rdm,grav);
  Real grad = (2.0*pow(rho,2.0-gamma)*accel/k0-rho*xi*pow(x,xi-1.0))
              /((1+pow(x,xi))*gamma);
  //std::cout << "rho:" << rho << " gamma:" << gamma << " xi:" << xi << std::endl;
  //std::cout << "x:" << x << " accel:" << accel << " grad:" << grad << std::endl;

  return grad;
}

//-----------------------------------------------------------------------------------------
//! \fn SolveDens()
//! \brief calculates the profile of the density
void SolveDens(DualArray1D<Real> &d_arr) {
  int lntot = NLEN_DENS_ARRAY;
  int linner = lntot/2, louter = lntot/2-1;
  Real logh = 0.005;
  Real x,h,dentry;
  Real &densentry = acc->dens_entry;
  d_arr.h_view(static_cast<int>(linner))=densentry;

  //solve inward from boundary
  dentry = densentry;
  for (int ent=0; ent<linner; ent++) {
    x = pow(10.0,-ent*logh);
    h = pow(10.0,-(ent+1)*logh)-x;
    dentry = RK4(DrhoDr,x,dentry,h);
    d_arr.h_view(static_cast<int>(linner)-ent-1)=dentry;
  }

  //solve outward from boundary
  dentry = densentry;
  for (int ent=0; ent<louter; ent++) {
    x = pow(10.0,ent*logh);
    h = pow(10.0,(ent+1)*logh)-x;
    dentry = RK4(DrhoDr,x,dentry,h);
    d_arr.h_view(static_cast<int>(linner)+ent+1)=dentry;
  }
  d_arr.template modify<HostMemSpace>();
  d_arr.template sync<DevExeSpace>();
}

Real DensFn(Real x, const DualArray1D<Real> &d_arr) {
  int lntot = NLEN_DENS_ARRAY;
  Real logx = log10(x);
  Real logh = 0.005;
  int ipps  = static_cast<int>(logx/logh) + lntot/2;
  ipps = (ipps < lntot-2)? ipps : lntot-2;
  ipps = (ipps > 0 )? ipps : 0;
  Real logx0 = logh*static_cast<Real>(ipps-lntot/2);
  Real dx = logx - logx0;
  Real dens = (d_arr.h_view(ipps+1)*dx - d_arr.h_view(ipps)*(dx - logh))/logh;
  //std::cout << "ipps:" << ipps << " logx0:" << logx0 << " logx:" << logx << std::endl;
  return dens;
}

KOKKOS_INLINE_FUNCTION
Real DensFnDevice(Real x, const DualArray1D<Real> &d_arr) {
  int lntot = NLEN_DENS_ARRAY;
  Real logx = log10(x);
  Real logh = 0.005;
  int ipps  = static_cast<int>(logx/logh) + lntot/2;
  ipps = (ipps < lntot-2)? ipps : lntot-2;
  ipps = (ipps > 0 )? ipps : 0;
  Real logx0 = logh*static_cast<Real>(ipps-lntot/2);
  Real dx = logx - logx0;
  Real dens = (d_arr.d_view(ipps+1)*dx - d_arr.d_view(ipps)*(dx - logh))/logh;
  return dens;
}

KOKKOS_INLINE_FUNCTION
int GetRadialIndex(Real rad, Real logr0, Real logh, int rntot) {
  Real logr = log10(rad)-logr0;
  int ipps  = static_cast<int>(logr/logh+0.5);
  ipps = (ipps > 0 )? ipps : 0;
  ipps = (ipps < rntot-1)? ipps : rntot-1;
  return ipps;
}

KOKKOS_INLINE_FUNCTION
Real GetRadialVar(DualArray1D<Real> vararr, Real rad, Real logr0, Real logh, int rntot) {
  Real logr = log10(rad)-logr0;
  int ipps  = static_cast<int>(logr/logh);
  ipps = (ipps > 0 )? ipps : 0;
  ipps = (ipps < rntot-2)? ipps : rntot-2;
  Real x0 = logr0 + logh*static_cast<Real>(ipps);
  Real dx = log10(rad) - x0;
  Real var = (vararr.d_view(ipps+1)*dx - vararr.d_view(ipps)*(dx-logh))/logh;
  return pow(10.0,var);
}

void AccFinalWork(ParameterInput *pin, Mesh *pm) {
  delete acc;
}

} // namespace
