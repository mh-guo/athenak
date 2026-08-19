#ifndef PGEN_TURB_SEED_HPP_
#define PGEN_TURB_SEED_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file turb_seed.hpp
//! \brief One-shot initial turbulence seed perturbations, confined to a spherical
//! window, built from a small number of Fourier modes defined on a *local* box of
//! size l_turb (default 2*r_out) rather than the full mesh.  This keeps the mode
//! count independent of the mesh size, so a small region of interest inside a very
//! large box needs only a few tens of modes.
//!
//! The random field is a continuous function of position (compact mode list with
//! random amplitudes/phases generated host-side from a seed), therefore:
//!  - identical across MPI decompositions, meshblock layouts, and resolutions;
//!  - evaluable at any point (cell centers now; face/edge sampling for the future
//!    bfld consumer).
//!
//! Supported fields (one <turb_seed*> input block per field, applied in the order
//! the blocks appear, all before PrimToCons):
//!   field = vel   : velocity seed, added to w0 velocities.
//!                   solenoidal = true (default): dv = curl(g(r) W(r) A(x)) with
//!                   A a random vector potential -> exactly divergence-free even
//!                   with the window and radial amplitude shaping applied.
//!                   solenoidal = false: three independent scalar fields, with
//!                   optional nominal Helmholtz weight sol_frac (applied in k-space
//!                   BEFORE windowing; the window leaks O(1/(k dr)) between the
//!                   solenoidal and compressive parts).
//!                   Amplitude: mach = mass-weighted rms of |dv|/c_s in the window.
//!                   c_s^2 = gamma p_gas/rho, plus (4/3) p_rad/rho in radiation
//!                   runs (LTE p_rad from the gas temperature), so mach is measured
//!                   against the effective sound speed even in radiation pressure
//!                   dominated regions.
//!   field = dens  : isobaric density perturbation (= entropy/buoyancy seed;
//!                   recommended for triggering convection). rho *= (1 + d),
//!                   window-mean removed (net mass conserved), d floored at -0.9.
//!                   Amplitude: amp = weighted rms of d in the window.
//!   field = pres  : isochoric pressure perturbation (seeds sound waves+buoyancy).
//!                   Same normalization semantics as dens.
//!   field = bfld  : turbulent magnetic field, ADDED to the existing b0/bcc0
//!                   (requires MHD; call after the pgen's own field setup).
//!                   The windowed potential Phi(r)*A(x) is sampled at cell edges
//!                   and differenced (constrained-transport curl), so the discrete
//!                   divB is zero to machine precision. At faces shared with a
//!                   finer MeshBlock the edge value is replaced by the average of
//!                   the two fine-level samples (same scheme as the torus field),
//!                   which makes coarse face flux = sum of fine fluxes, keeping
//!                   divB = 0 across static refinement boundaries and under later
//!                   restriction (e.g. cyclic zoom regrid). A one-time max|divB|
//!                   diagnostic of the total b0 is printed to the log.
//!                   Amplitude: inv_beta = <p_mag>/<p_gas> (volume-weighted means
//!                   over the window; seed contribution only). In radiation runs
//!                   the denominator is p_gas + p_rad (LTE p_rad from the gas
//!                   temperature), i.e. a ptot = p_gas + p_rad convention.
//!   field = erad  : NOT YET IMPLEMENTED (LTE-consistent radiation perturbation).
//!
//! Caveats (documented design decisions):
//!  - vel: a one-shot seed deposits a small random net momentum (reported in the
//!    log); it is NOT subtracted because any confined subtraction would break the
//!    divergence-free property. For seeding this kick is O(mach*c_s/sqrt(nmode)).
//!  - dens/pres in radiation runs: the radiation field is initialized from the
//!    unperturbed temperature; optically thick regions partially erase thermal
//!    perturbations within a few source steps. Prefer field=vel there.
//!  - vel in GR: dv is added to the normal-frame velocities w0(IVX..IVZ) assuming
//!    a near-flat metric; keep the window at r >> r_horizon.
//!
//! Example input blocks:
//!   <turb_seed_vel>
//!   field      = vel
//!   mach       = 0.05    # rms dv/c_s in window
//!   r_in       = 30.0    # window inner radius
//!   r_out      = 1000.0  # window outer radius
//!   dr_in      = 15.0    # inner taper width (default 0.5*r_in)
//!   dr_out     = 200.0   # outer taper width (default 0.2*(r_out-r_in))
//!   nlow       = 1       # mode range relative to l_turb (default 2*r_out)
//!   nhigh      = 4
//!   expo       = 1.6667  # spectral index of the physical field
//!   amp_rpow   = -0.5    # radial amplitude shape g(r) = (r/r_out)^amp_rpow
//!   seed       = 1       # use a different seed for each block
//!
//!   <turb_seed_ent>
//!   field      = dens
//!   amp        = 0.02
//!   r_in       = 30.0
//!   r_out      = 1000.0
//!   nlow       = 1
//!   nhigh      = 4
//!   seed       = 2

#include <array>
#include <cmath>
#include <cstdlib>
#include <complex>
#include <iostream>
#include <string>
#include <vector>

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "radiation/radiation.hpp"
#include "eos/eos.hpp"
#include "utils/random.hpp"

//----------------------------------------------------------------------------------------
//! \struct TurbSeedField
//! \brief Device-copyable container holding the mode list and the window/profile
//! parameters, with point-evaluators usable from any kernel. All spatial dependence
//! (window W(r), radial shape g(r), modes) lives here; normalization is a single
//! global constant applied by the consumer, so solenoidality is never broken.

struct TurbSeedField {
  // window center and radii
  Real x0, y0, z0;
  Real r_in, r_out, dr_in, dr_out;
  Real rpow;              // g(r) = (r/r_out)^rpow
  int nmode;
  DualArray2D<Real> kv;   // (nmode,3) wavevectors
  DualArray2D<Real> av;   // (nmode,3) per-component amplitudes (scalar uses c=0)
  DualArray2D<Real> ph;   // (nmode,3) per-component phases

  // Envelope Phi(r) = g(r)*W(r) and dPhi/dr; also returns r and offsets from center
  KOKKOS_INLINE_FUNCTION
  void Envelope(Real x, Real y, Real z,
                Real &dx, Real &dy, Real &dz, Real &r,
                Real &phi, Real &dphidr) const {
    dx = x - x0; dy = y - y0; dz = z - z0;
    r = sqrt(dx*dx + dy*dy + dz*dz);
    phi = 0.0; dphidr = 0.0;
    if (r <= r_in || r >= r_out) return;
    Real w, dwdr;
    if (dr_in > 0.0 && r < r_in + dr_in) {
      Real t = 0.5*M_PI*(r - r_in)/dr_in;
      Real s = sin(t), c = cos(t);
      w = s*s;
      dwdr = M_PI/dr_in*s*c;
    } else if (dr_out > 0.0 && r > r_out - dr_out) {
      Real t = 0.5*M_PI*(r_out - r)/dr_out;
      Real s = sin(t), c = cos(t);
      w = s*s;
      dwdr = -M_PI/dr_out*s*c;
    } else {
      w = 1.0; dwdr = 0.0;
    }
    Real rg = fmax(r, static_cast<Real>(1.0e-12));
    Real g = pow(rg/r_out, rpow);
    Real dgdr = rpow*g/rg;
    phi = g*w;
    dphidr = dgdr*w + g*dwdr;
  }

  // Scalar field: returns envelope Phi and raw mode sum f (not Phi*f).
  // Consumer: q *= 1 + s*Phi*(f - mu) with mu = Phi-weighted mean of f.
  KOKKOS_INLINE_FUNCTION
  void ScalarEval(Real x, Real y, Real z, Real &phi, Real &f) const {
    Real dx, dy, dz, r, dphidr;
    Envelope(x, y, z, dx, dy, dz, r, phi, dphidr);
    f = 0.0;
    if (phi <= 0.0) return;
    for (int n=0; n<nmode; ++n) {
      Real arg = kv.d_view(n,0)*dx + kv.d_view(n,1)*dy + kv.d_view(n,2)*dz
               + ph.d_view(n,0);
      f += av.d_view(n,0)*cos(arg);
    }
  }

  // Solenoidal vector field: dv = curl(phi(r) A(x))
  //                             = phi*(curl A) + (dphi/dr)*(rhat x A)
  // with A_c(x) = sum_n a_{n,c} cos(k_n.(x-x0) + ph_{n,c}); exactly div-free.
  KOKKOS_INLINE_FUNCTION
  void VecEvalSol(Real x, Real y, Real z, Real dv[3]) const {
    dv[0] = 0.0; dv[1] = 0.0; dv[2] = 0.0;
    Real dx, dy, dz, r, phi, dphidr;
    Envelope(x, y, z, dx, dy, dz, r, phi, dphidr);
    if (phi <= 0.0) return;
    Real a[3] = {0.0, 0.0, 0.0};      // A_c
    Real da[3][3];                     // da[c][j] = dA_c/dx_j
    for (int c=0; c<3; ++c) {for (int j=0; j<3; ++j) {da[c][j] = 0.0;}}
    for (int n=0; n<nmode; ++n) {
      Real kx = kv.d_view(n,0), ky = kv.d_view(n,1), kz = kv.d_view(n,2);
      Real kdx = kx*dx + ky*dy + kz*dz;
      for (int c=0; c<3; ++c) {
        Real arg = kdx + ph.d_view(n,c);
        Real amp = av.d_view(n,c);
        a[c] += amp*cos(arg);
        Real msn = -amp*sin(arg);
        da[c][0] += msn*kx;
        da[c][1] += msn*ky;
        da[c][2] += msn*kz;
      }
    }
    // curl A
    Real ca[3];
    ca[0] = da[2][1] - da[1][2];
    ca[1] = da[0][2] - da[2][0];
    ca[2] = da[1][0] - da[0][1];
    // grad(phi) = dphidr * (dx,dy,dz)/r
    Real gp = (r > 1.0e-12) ? dphidr/r : 0.0;
    dv[0] = phi*ca[0] + gp*(dy*a[2] - dz*a[1]);
    dv[1] = phi*ca[1] + gp*(dz*a[0] - dx*a[2]);
    dv[2] = phi*ca[2] + gp*(dx*a[1] - dy*a[0]);
  }

  // Windowed vector potential component c at an arbitrary point (for edge sampling
  // by the bfld consumer; the discrete curl is taken by the caller)
  KOKKOS_INLINE_FUNCTION
  Real PotEval(int c, Real x, Real y, Real z) const {
    Real dx, dy, dz, r, phi, dphidr;
    Envelope(x, y, z, dx, dy, dz, r, phi, dphidr);
    if (phi <= 0.0) return 0.0;
    Real a = 0.0;
    for (int n=0; n<nmode; ++n) {
      Real arg = kv.d_view(n,0)*dx + kv.d_view(n,1)*dy + kv.d_view(n,2)*dz
               + ph.d_view(n,c);
      a += av.d_view(n,c)*cos(arg);
    }
    return phi*a;
  }

  // Generic vector field: three independent windowed scalars (no div constraint)
  KOKKOS_INLINE_FUNCTION
  void VecEvalGen(Real x, Real y, Real z, Real dv[3]) const {
    dv[0] = 0.0; dv[1] = 0.0; dv[2] = 0.0;
    Real dx, dy, dz, r, phi, dphidr;
    Envelope(x, y, z, dx, dy, dz, r, phi, dphidr);
    if (phi <= 0.0) return;
    for (int n=0; n<nmode; ++n) {
      Real kdx = kv.d_view(n,0)*dx + kv.d_view(n,1)*dy + kv.d_view(n,2)*dz;
      for (int c=0; c<3; ++c) {
        dv[c] += av.d_view(n,c)*cos(kdx + ph.d_view(n,c));
      }
    }
    dv[0] *= phi; dv[1] *= phi; dv[2] *= phi;
  }
};

//----------------------------------------------------------------------------------------
//! \class TurbSeed
//! \brief Reads one <turb_seed*> input block, generates the mode list (host-side,
//! deterministic from seed), and applies the perturbation to w0 (call before
//! PrimToCons). Stateless after Apply(); nothing needs to survive restarts.

class TurbSeed {
 public:
  TurbSeed(std::string bk, MeshBlockPack *pp, ParameterInput *pin);
  void Apply();

  // CUDA/nvcc: extended __host__ __device__ lambdas cannot live in private/protected
  // member functions (AMD/HIP does not enforce this). Declared public for that reason.
  void ApplyVel();
  void ApplyScalar();
  void ApplyBfld();

 private:
  MeshBlockPack *pmy_pack;
  std::string bname;
  std::string field;
  bool solenoidal = true;
  Real sol_frac = -1.0;
  Real target;
  Real l_turb, expo;
  int nlow, nhigh;
  int64_t seed;
  TurbSeedField fld;

  void GenerateModes();
};


inline TurbSeed::TurbSeed(std::string bk, MeshBlockPack *pp, ParameterInput *pin) :
    pmy_pack(pp), bname(bk) {
    field = pin->GetString(bk, "field");
    if (field == "erad") {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<" << bk << "> field=" << field
                << " is not implemented yet (planned)" << std::endl;
      exit(EXIT_FAILURE);
    }
    if (field != "vel" && field != "dens" && field != "pres" && field != "bfld") {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<" << bk << "> unrecognized field=" << field
                << " (must be vel|dens|pres|bfld|erad)" << std::endl;
      exit(EXIT_FAILURE);
    }

    // amplitude key is field-specific by design (self-documenting)
    if (field == "vel") {
      target = pin->GetReal(bk, "mach");
      solenoidal = pin->GetOrAddBoolean(bk, "solenoidal", true);
      sol_frac = pin->GetOrAddReal(bk, "sol_frac", -1.0);  // <0: no projection
    } else if (field == "bfld") {
      target = pin->GetReal(bk, "inv_beta");
      if (pmy_pack->pmhd == nullptr) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<" << bk << "> field=bfld requires MHD"
                  << std::endl;
        exit(EXIT_FAILURE);
      }
    } else {
      target = pin->GetReal(bk, "amp");
      if (target > 0.5 && global_variable::my_rank == 0) {
        std::cout << "### WARNING in <" << bk << ">: amp = " << target
                  << " > 0.5; perturbation will be floored at -0.9" << std::endl;
      }
    }

    // window geometry
    fld.x0 = pin->GetOrAddReal(bk, "x1_0", 0.0);
    fld.y0 = pin->GetOrAddReal(bk, "x2_0", 0.0);
    fld.z0 = pin->GetOrAddReal(bk, "x3_0", 0.0);
    fld.r_in  = pin->GetOrAddReal(bk, "r_in", 0.0);
    fld.r_out = pin->GetReal(bk, "r_out");
    fld.dr_in  = pin->GetOrAddReal(bk, "dr_in",  0.5*fld.r_in);
    fld.dr_out = pin->GetOrAddReal(bk, "dr_out", 0.2*(fld.r_out - fld.r_in));
    fld.rpow = pin->GetOrAddReal(bk, "amp_rpow", 0.0);
    if (fld.r_out <= fld.r_in) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<" << bk << "> requires r_out > r_in" << std::endl;
      exit(EXIT_FAILURE);
    }
    if (fld.r_in + fld.dr_in > fld.r_out - fld.dr_out) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<" << bk << "> tapers overlap: requires "
                << "r_in + dr_in <= r_out - dr_out" << std::endl;
      exit(EXIT_FAILURE);
    }
    if (fld.r_in <= 0.0 && fld.rpow < 0.0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<" << bk << "> amp_rpow < 0 requires r_in > 0"
                << std::endl;
      exit(EXIT_FAILURE);
    }

    // spectrum: wavenumbers relative to the local box l_turb, NOT the mesh
    l_turb = pin->GetOrAddReal(bk, "l_turb", 2.0*fld.r_out);
    nlow  = pin->GetOrAddInteger(bk, "nlow", 1);
    nhigh = pin->GetOrAddInteger(bk, "nhigh", 4);
    expo  = pin->GetOrAddReal(bk, "expo", 5.0/3.0);
    if (nlow < 1 || nhigh < nlow) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<" << bk << "> requires 1 <= nlow <= nhigh"
                << std::endl;
      exit(EXIT_FAILURE);
    }
    seed = pin->GetOrAddInteger(bk, "seed", 1);

    GenerateModes();
  }


inline void TurbSeed::Apply() {
    if (field == "vel") {
      ApplyVel();
    } else if (field == "bfld") {
      ApplyBfld();
    } else {
      ApplyScalar();
    }
  }


//--------------------------------------------------------------------------------------
// Generate the compact mode list on the host, deterministically from seed.
// k runs over a half-space (kz>0; kz=0,ky>0; kz=ky=0,kx>0) so that, with a random
// phase per mode and component, a general real random field is represented (this
// replaces the historical 8-fold sin/cos coefficient bookkeeping).
inline void TurbSeed::GenerateModes() {
    // number of independent scalar components carrying randomness
    const int ncomp = (field == "vel" || field == "bfld") ? 3 : 1;
    // solenoidal velocity and bfld are built as curl of a potential: potential
    // amplitudes carry an extra 1/k so `expo` always refers to the physical field
    const bool potential_norm = ((field == "vel") && solenoidal) || (field == "bfld");

    const Real dk = 2.0*M_PI/l_turb;
    const int nlow_sq = nlow*nlow, nhigh_sq = nhigh*nhigh;

    // negative idum (re)initializes the Ran2/RanGaussian static state, making each
    // block's sequence deterministic and identical on all ranks
    int64_t idum = -(std::abs(seed) + 1);
    RanGaussian(&idum);  // warm-up call performs the initialization

    std::vector<std::array<Real,3>> kvec, amp, phs;
    for (int n3=0; n3<=nhigh; ++n3) {
      for (int n2=-nhigh; n2<=nhigh; ++n2) {
        for (int n1=-nhigh; n1<=nhigh; ++n1) {
          if (n3 == 0) {
            if (n2 < 0) continue;
            if (n2 == 0 && n1 <= 0) continue;
          }
          int nsq = n1*n1 + n2*n2 + n3*n3;
          if (nsq < nlow_sq || nsq > nhigh_sq) continue;

          Real kx = dk*n1, ky = dk*n2, kz = dk*n3;
          Real kmag = sqrt(kx*kx + ky*ky + kz*kz);
          // per-mode amplitude ~ k^-(expo+2)/2 => shell-summed E(k) ~ k^-expo
          Real norm = pow(kmag, -0.5*(expo + 2.0));
          if (potential_norm) norm /= kmag;

          std::array<Real,3> a{0.0, 0.0, 0.0}, p{0.0, 0.0, 0.0};
          for (int c=0; c<ncomp; ++c) {
            a[c] = norm*RanGaussian(&idum);
            p[c] = 2.0*M_PI*Ran2(&idum);
          }

          // optional nominal Helmholtz weighting for the generic-vector path,
          // done on the complex amplitude vector (phases mix under projection)
          if (field == "vel" && !solenoidal && sol_frac >= 0.0) {
            std::complex<Real> cv[3], cpar[3];
            for (int c=0; c<3; ++c) {
              cv[c] = std::polar(a[c], p[c]);
            }
            Real kh[3] = {kx/kmag, ky/kmag, kz/kmag};
            std::complex<Real> kdotc = kh[0]*cv[0] + kh[1]*cv[1] + kh[2]*cv[2];
            for (int c=0; c<3; ++c) {
              cpar[c] = kh[c]*kdotc;
              std::complex<Real> cnew = sol_frac*(cv[c] - cpar[c])
                                      + (1.0 - sol_frac)*cpar[c];
              a[c] = std::abs(cnew);
              p[c] = std::arg(cnew);
            }
          }

          kvec.push_back({kx, ky, kz});
          amp.push_back(a);
          phs.push_back(p);
        }
      }
    }

    fld.nmode = static_cast<int>(kvec.size());
    if (fld.nmode == 0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<" << bname << "> selected zero modes" << std::endl;
      exit(EXIT_FAILURE);
    }

    Kokkos::realloc(fld.kv, fld.nmode, 3);
    Kokkos::realloc(fld.av, fld.nmode, 3);
    Kokkos::realloc(fld.ph, fld.nmode, 3);
    for (int n=0; n<fld.nmode; ++n) {
      for (int c=0; c<3; ++c) {
        fld.kv.h_view(n,c) = kvec[n][c];
        fld.av.h_view(n,c) = amp[n][c];
        fld.ph.h_view(n,c) = phs[n][c];
      }
    }
    fld.kv.template modify<HostMemSpace>();
    fld.kv.template sync<DevExeSpace>();
    fld.av.template modify<HostMemSpace>();
    fld.av.template sync<DevExeSpace>();
    fld.ph.template modify<HostMemSpace>();
    fld.ph.template sync<DevExeSpace>();
  }


//--------------------------------------------------------------------------------------
// Velocity seed: dv added to w0 velocities; normalized so that the mass-weighted
// rms of |dv|/c_s over the window equals `mach`. Field evaluated twice (reduce +
// apply) instead of stored -- no persistent arrays needed.
inline void TurbSeed::ApplyVel() {
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    int is = indcs.is, js = indcs.js, ks = indcs.ks;
    int ie = indcs.ie, je = indcs.je, ke = indcs.ke;
    int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
    int nmb = pmy_pack->nmb_thispack;
    auto &size = pmy_pack->pmb->mb_size;

    DvceArray5D<Real> w0_;
    Real gamma;
    if (pmy_pack->phydro != nullptr) {
      w0_ = pmy_pack->phydro->w0;
      gamma = pmy_pack->phydro->peos->eos_data.gamma;
    } else {
      w0_ = pmy_pack->pmhd->w0;
      gamma = pmy_pack->pmhd->peos->eos_data.gamma;
    }
    const Real gm1 = gamma - 1.0;
    const bool use_dyngr = (pmy_pack->pdyngr != nullptr);
    const bool sol = solenoidal;
    // in radiation runs include LTE radiation pressure in the sound speed, else
    // mach would be measured against the (much smaller) gas-only c_s in radiation
    // pressure dominated regions; exact here since the pgen initializes in LTE
    const bool is_rad = (pmy_pack->prad != nullptr);
    const Real arad_ = is_rad ? pmy_pack->prad->arad : 0.0;
    auto fld_ = fld;

    // normalization sums: t0 = sum(dm), t1 = sum(dm |dv|^2/c_s^2) over window,
    // p1..p3 = sum(dm dv_c) for the net-momentum report
    const int nmkji = nmb*nx3*nx2*nx1;
    const int nkji = nx3*nx2*nx1;
    const int nji = nx2*nx1;
    Real t0 = 0.0, t1 = 0.0, p1 = 0.0, p2 = 0.0, p3 = 0.0;
    Kokkos::parallel_reduce("tseed_vnorm", Kokkos::RangePolicy<>(DevExeSpace(),0,nmkji),
    KOKKOS_LAMBDA(const int &idx, Real &s0, Real &s1, Real &q1, Real &q2, Real &q3) {
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks; j += js;

      Real x1v = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
      Real x2v = CellCenterX(j-js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
      Real x3v = CellCenterX(k-ks, nx3, size.d_view(m).x3min, size.d_view(m).x3max);

      Real dv[3];
      if (sol) {
        fld_.VecEvalSol(x1v, x2v, x3v, dv);
      } else {
        fld_.VecEvalGen(x1v, x2v, x3v, dv);
      }
      if (dv[0] == 0.0 && dv[1] == 0.0 && dv[2] == 0.0) return;

      Real vol = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
      Real rho = w0_(m,IDN,k,j,i);
      Real pgas = use_dyngr ? w0_(m,IPR,k,j,i) : gm1*w0_(m,IEN,k,j,i);
      Real cs2 = gamma*pgas/rho;
      if (is_rad) {
        Real tgas = pgas/rho;
        Real prad = arad_*SQR(SQR(tgas))/3.0;
        cs2 += (4.0/3.0)*prad/rho;  // Gamma1 -> 4/3 in the radiation-dominated limit
      }
      cs2 = fmax(cs2, static_cast<Real>(1.0e-30));
      Real dm = rho*vol;
      s0 += dm;
      s1 += dm*(dv[0]*dv[0] + dv[1]*dv[1] + dv[2]*dv[2])/cs2;
      q1 += dm*dv[0];
      q2 += dm*dv[1];
      q3 += dm*dv[2];
    }, Kokkos::Sum<Real>(t0), Kokkos::Sum<Real>(t1),
       Kokkos::Sum<Real>(p1), Kokkos::Sum<Real>(p2), Kokkos::Sum<Real>(p3));

#if MPI_PARALLEL_ENABLED
    Real sums[5] = {t0, t1, p1, p2, p3};
    MPI_Allreduce(MPI_IN_PLACE, sums, 5, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
    t0 = sums[0]; t1 = sums[1]; p1 = sums[2]; p2 = sums[3]; p3 = sums[4];
#endif

    if (t1 <= 0.0 || t0 <= 0.0) {
      if (global_variable::my_rank == 0) {
        std::cout << "### WARNING in <" << bname << ">: window contains no "
                  << "perturbed cells; velocity seed skipped" << std::endl;
      }
      return;
    }
    const Real s = target/sqrt(t1/t0);

    par_for("tseed_vapply", DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real x1v = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
      Real x2v = CellCenterX(j-js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
      Real x3v = CellCenterX(k-ks, nx3, size.d_view(m).x3min, size.d_view(m).x3max);
      Real dv[3];
      if (sol) {
        fld_.VecEvalSol(x1v, x2v, x3v, dv);
      } else {
        fld_.VecEvalGen(x1v, x2v, x3v, dv);
      }
      w0_(m,IVX,k,j,i) += s*dv[0];
      w0_(m,IVY,k,j,i) += s*dv[1];
      w0_(m,IVZ,k,j,i) += s*dv[2];
    });

    if (global_variable::my_rank == 0) {
      std::cout << "<" << bname << "> field=vel"
                << (sol ? " (solenoidal)" : " (generic)")
                << " nmode=" << fld.nmode
                << " mach_rms=" << target
                << " norm=" << s
                << " net_dv=(" << s*p1/t0 << "," << s*p2/t0 << "," << s*p3/t0 << ")"
                << std::endl;
    }
  }


//--------------------------------------------------------------------------------------
// Scalar seed (dens or pres): q *= 1 + s*phi*(f - mu). The mean mu is q-weighted
// over the window so that the net integral of q is conserved exactly (pre-floor);
// rms is q-weighted so `amp` = rms relative fluctuation.
inline void TurbSeed::ApplyScalar() {
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    int is = indcs.is, js = indcs.js, ks = indcs.ks;
    int ie = indcs.ie, je = indcs.je, ke = indcs.ke;
    int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
    int nmb = pmy_pack->nmb_thispack;
    auto &size = pmy_pack->pmb->mb_size;

    DvceArray5D<Real> w0_;
    if (pmy_pack->phydro != nullptr) {
      w0_ = pmy_pack->phydro->w0;
    } else {
      w0_ = pmy_pack->pmhd->w0;
    }
    const bool use_dyngr = (pmy_pack->pdyngr != nullptr);
    // dens -> IDN; pres -> IPR (dyngr) or IEN (multiplying e multiplies p_gas)
    const int iq = (field == "dens") ? static_cast<int>(IDN)
                   : (use_dyngr ? static_cast<int>(IPR) : static_cast<int>(IEN));
    auto fld_ = fld;

    // sums: S0 = sum(q dV) over window; M0 = sum(q dV phi); M1 = sum(q dV phi f);
    //       V0 = sum(q dV phi^2); V1 = sum(q dV phi^2 f); V2 = sum(q dV phi^2 f^2)
    const int nmkji = nmb*nx3*nx2*nx1;
    const int nkji = nx3*nx2*nx1;
    const int nji = nx2*nx1;
    Real s0 = 0.0, m0 = 0.0, m1 = 0.0, v0 = 0.0, v1 = 0.0, v2 = 0.0;
    Kokkos::parallel_reduce("tseed_snorm", Kokkos::RangePolicy<>(DevExeSpace(),0,nmkji),
    KOKKOS_LAMBDA(const int &idx, Real &r0, Real &r1, Real &r2,
                  Real &r3, Real &r4, Real &r5) {
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks; j += js;

      Real x1v = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
      Real x2v = CellCenterX(j-js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
      Real x3v = CellCenterX(k-ks, nx3, size.d_view(m).x3min, size.d_view(m).x3max);

      Real phi, f;
      fld_.ScalarEval(x1v, x2v, x3v, phi, f);
      if (phi <= 0.0) return;

      Real vol = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
      Real wq = w0_(m,iq,k,j,i)*vol;
      r0 += wq;
      r1 += wq*phi;
      r2 += wq*phi*f;
      r3 += wq*phi*phi;
      r4 += wq*phi*phi*f;
      r5 += wq*phi*phi*f*f;
    }, Kokkos::Sum<Real>(s0), Kokkos::Sum<Real>(m0), Kokkos::Sum<Real>(m1),
       Kokkos::Sum<Real>(v0), Kokkos::Sum<Real>(v1), Kokkos::Sum<Real>(v2));

#if MPI_PARALLEL_ENABLED
    Real sums[6] = {s0, m0, m1, v0, v1, v2};
    MPI_Allreduce(MPI_IN_PLACE, sums, 6, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
    s0 = sums[0]; m0 = sums[1]; m1 = sums[2];
    v0 = sums[3]; v1 = sums[4]; v2 = sums[5];
#endif

    if (s0 <= 0.0 || m0 <= 0.0) {
      if (global_variable::my_rank == 0) {
        std::cout << "### WARNING in <" << bname << ">: window contains no "
                  << "perturbed cells; scalar seed skipped" << std::endl;
      }
      return;
    }
    const Real mu = m1/m0;
    const Real var = (v2 - 2.0*mu*v1 + mu*mu*v0)/s0;
    if (var <= 0.0) {
      if (global_variable::my_rank == 0) {
        std::cout << "### WARNING in <" << bname << ">: zero variance; "
                  << "scalar seed skipped" << std::endl;
      }
      return;
    }
    const Real s = target/sqrt(var);

    par_for("tseed_sapply", DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real x1v = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
      Real x2v = CellCenterX(j-js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
      Real x3v = CellCenterX(k-ks, nx3, size.d_view(m).x3min, size.d_view(m).x3max);
      Real phi, f;
      fld_.ScalarEval(x1v, x2v, x3v, phi, f);
      if (phi <= 0.0) return;
      Real d = fmax(s*phi*(f - mu), static_cast<Real>(-0.9));
      w0_(m,iq,k,j,i) *= (1.0 + d);
    });

    if (global_variable::my_rank == 0) {
      std::cout << "<" << bname << "> field=" << field
                << " nmode=" << fld.nmode
                << " amp_rms=" << target
                << " norm=" << s << " mean_removed=" << mu << std::endl;
    }
  }


//--------------------------------------------------------------------------------------
// Magnetic seed: windowed potential sampled at cell edges, discrete (CT) curl to
// face fields, ADDED to existing b0. At faces/edges shared with a finer MeshBlock
// the edge value is the average of the two fine-level samples so that the coarse
// face flux equals the sum of the fine fluxes (scheme copied from zoom_gr_torus);
// discrete divB stays zero across refinement boundaries. Normalized by a single
// global constant so divB is unaffected: <p_mag>/<p_gas> over window = inv_beta.
inline void TurbSeed::ApplyBfld() {
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    int is = indcs.is, js = indcs.js, ks = indcs.ks;
    int ie = indcs.ie, je = indcs.je, ke = indcs.ke;
    int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
    int ncells1 = nx1 + 2*(indcs.ng);
    int ncells2 = (nx2 > 1)? (nx2 + 2*(indcs.ng)) : 1;
    int ncells3 = (nx3 > 1)? (nx3 + 2*(indcs.ng)) : 1;
    int nmb = pmy_pack->nmb_thispack;
    auto &size = pmy_pack->pmb->mb_size;
    auto &nghbr = pmy_pack->pmb->nghbr;
    auto &mblev = pmy_pack->pmb->mb_lev;

    DvceArray5D<Real> w0_ = pmy_pack->pmhd->w0;
    const Real gamma = pmy_pack->pmhd->peos->eos_data.gamma;
    const Real gm1 = gamma - 1.0;
    const bool use_dyngr = (pmy_pack->pdyngr != nullptr);
    // in radiation runs inv_beta is measured against p_gas + p_rad (LTE), matching
    // the ptot convention of the pgen's own potential_beta_min normalization
    const bool is_rad = (pmy_pack->prad != nullptr);
    const Real arad_ = is_rad ? pmy_pack->prad->arad : 0.0;
    auto fld_ = fld;

    // 1. sample windowed potential Phi*A at cell edges, with fine-neighbor
    //    correction on shared faces/edges
    DvceArray4D<Real> a1, a2, a3;
    Kokkos::realloc(a1, nmb, ncells3, ncells2, ncells1);
    Kokkos::realloc(a2, nmb, ncells3, ncells2, ncells1);
    Kokkos::realloc(a3, nmb, ncells3, ncells2, ncells1);

    par_for("tseed_apot", DevExeSpace(), 0,nmb-1,ks,ke+1,js,je+1,is,ie+1,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
      Real x1f = LeftEdgeX(i-is, nx1, x1min, x1max);

      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
      Real x2f = LeftEdgeX(j-js, nx2, x2min, x2max);

      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
      Real x3f = LeftEdgeX(k-ks, nx3, x3min, x3max);

      Real dx1 = size.d_view(m).dx1;
      Real dx2 = size.d_view(m).dx2;
      Real dx3 = size.d_view(m).dx3;

      a1(m,k,j,i) = fld_.PotEval(0, x1v, x2f, x3f);
      a2(m,k,j,i) = fld_.PotEval(1, x1f, x2v, x3f);
      a3(m,k,j,i) = fld_.PotEval(2, x1f, x2f, x3v);

      // When neighboring MeshBlock is at finer level, compute potential as average
      // of values at fine grid resolution; guarantees flux on shared fine/coarse
      // faces is identical (same scheme as zoom_gr_torus.cpp)

      // Correct A1 at x2-faces, x3-faces, and x2x3-edges
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
        a1(m,k,j,i) = 0.5*(fld_.PotEval(0,xl,x2f,x3f) + fld_.PotEval(0,xr,x2f,x3f));
      }

      // Correct A2 at x1-faces, x3-faces, and x1x3-edges
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
        a2(m,k,j,i) = 0.5*(fld_.PotEval(1,x1f,xl,x3f) + fld_.PotEval(1,x1f,xr,x3f));
      }

      // Correct A3 at x1-faces, x2-faces, and x1x2-edges
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
        a3(m,k,j,i) = 0.5*(fld_.PotEval(2,x1f,x2f,xl) + fld_.PotEval(2,x1f,x2f,xr));
      }
    });

    // 2. discrete (CT) curl -> seed face fields
    DvceFaceFld4D<Real> bs("tseed_b", nmb, ncells3, ncells2, ncells1);
    par_for("tseed_bface", DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real dx1 = size.d_view(m).dx1;
      Real dx2 = size.d_view(m).dx2;
      Real dx3 = size.d_view(m).dx3;

      bs.x1f(m,k,j,i) = ((a3(m,k,j+1,i) - a3(m,k,j,i))/dx2 -
                         (a2(m,k+1,j,i) - a2(m,k,j,i))/dx3);
      bs.x2f(m,k,j,i) = ((a1(m,k+1,j,i) - a1(m,k,j,i))/dx3 -
                         (a3(m,k,j,i+1) - a3(m,k,j,i))/dx1);
      bs.x3f(m,k,j,i) = ((a2(m,k,j,i+1) - a2(m,k,j,i))/dx1 -
                         (a1(m,k,j+1,i) - a1(m,k,j,i))/dx2);

      // Include extra face-component at edge of block in each direction
      if (i==ie) {
        bs.x1f(m,k,j,i+1) = ((a3(m,k,j+1,i+1) - a3(m,k,j,i+1))/dx2 -
                             (a2(m,k+1,j,i+1) - a2(m,k,j,i+1))/dx3);
      }
      if (j==je) {
        bs.x2f(m,k,j+1,i) = ((a1(m,k+1,j+1,i) - a1(m,k,j+1,i))/dx3 -
                             (a3(m,k,j+1,i+1) - a3(m,k,j+1,i))/dx1);
      }
      if (k==ke) {
        bs.x3f(m,k+1,j,i) = ((a2(m,k+1,j,i+1) - a2(m,k+1,j,i))/dx1 -
                             (a1(m,k+1,j+1,i) - a1(m,k+1,j,i))/dx2);
      }
    });

    // 3. measure volume-weighted <p_mag>/<p_gas>: seed p_mag everywhere (it is
    //    confined to the window by construction), p_gas over the window
    const int nmkji = nmb*nx3*nx2*nx1;
    const int nkji = nx3*nx2*nx1;
    const int nji = nx2*nx1;
    Real pg_sum = 0.0, pm_sum = 0.0;
    Kokkos::parallel_reduce("tseed_bnorm", Kokkos::RangePolicy<>(DevExeSpace(),0,nmkji),
    KOKKOS_LAMBDA(const int &idx, Real &pg, Real &pm) {
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks; j += js;

      Real vol = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
      Real bx = 0.5*(bs.x1f(m,k,j,i) + bs.x1f(m,k,j,i+1));
      Real by = 0.5*(bs.x2f(m,k,j,i) + bs.x2f(m,k,j+1,i));
      Real bz = 0.5*(bs.x3f(m,k,j,i) + bs.x3f(m,k+1,j,i));
      pm += vol*0.5*(bx*bx + by*by + bz*bz);

      Real x1v = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
      Real x2v = CellCenterX(j-js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
      Real x3v = CellCenterX(k-ks, nx3, size.d_view(m).x3min, size.d_view(m).x3max);
      Real dx, dy, dz, r, phi, dphidr;
      fld_.Envelope(x1v, x2v, x3v, dx, dy, dz, r, phi, dphidr);
      if (phi > 0.0) {
        Real pgas = use_dyngr ? w0_(m,IPR,k,j,i) : gm1*w0_(m,IEN,k,j,i);
        Real ptot = pgas;
        if (is_rad) {
          Real tgas = pgas/w0_(m,IDN,k,j,i);
          ptot += arad_*SQR(SQR(tgas))/3.0;
        }
        pg += vol*ptot;
      }
    }, Kokkos::Sum<Real>(pg_sum), Kokkos::Sum<Real>(pm_sum));

#if MPI_PARALLEL_ENABLED
    Real bsums[2] = {pg_sum, pm_sum};
    MPI_Allreduce(MPI_IN_PLACE, bsums, 2, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
    pg_sum = bsums[0]; pm_sum = bsums[1];
#endif

    if (pm_sum <= 0.0 || pg_sum <= 0.0) {
      if (global_variable::my_rank == 0) {
        std::cout << "### WARNING in <" << bname << ">: window contains no "
                  << "perturbed cells; bfld seed skipped" << std::endl;
      }
      return;
    }
    const Real s = sqrt(target/(pm_sum/pg_sum));

    // 4. add scaled seed to b0 (constant factor -> divB unaffected), refresh bcc0
    auto &b0 = pmy_pack->pmhd->b0;
    auto &bcc_ = pmy_pack->pmhd->bcc0;
    par_for("tseed_badd", DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      b0.x1f(m,k,j,i) += s*bs.x1f(m,k,j,i);
      b0.x2f(m,k,j,i) += s*bs.x2f(m,k,j,i);
      b0.x3f(m,k,j,i) += s*bs.x3f(m,k,j,i);
      if (i==ie) {b0.x1f(m,k,j,i+1) += s*bs.x1f(m,k,j,i+1);}
      if (j==je) {b0.x2f(m,k,j+1,i) += s*bs.x2f(m,k,j+1,i);}
      if (k==ke) {b0.x3f(m,k+1,j,i) += s*bs.x3f(m,k+1,j,i);}
    });
    par_for("tseed_bcc", DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      bcc_(m,IBX,k,j,i) = 0.5*(b0.x1f(m,k,j,i) + b0.x1f(m,k,j,i+1));
      bcc_(m,IBY,k,j,i) = 0.5*(b0.x2f(m,k,j,i) + b0.x2f(m,k,j+1,i));
      bcc_(m,IBZ,k,j,i) = 0.5*(b0.x3f(m,k,j,i) + b0.x3f(m,k+1,j,i));
    });

    // 5. one-time diagnostic: max |divB| of the total b0 (absolute, and relative
    //    to |B|/dx); expect relative values at roundoff level
    Real divb_max = 0.0, divb_rel_max = 0.0;
    Kokkos::parallel_reduce("tseed_divb", Kokkos::RangePolicy<>(DevExeSpace(),0,nmkji),
    KOKKOS_LAMBDA(const int &idx, Real &dmax, Real &rmax) {
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks; j += js;

      Real dx1 = size.d_view(m).dx1;
      Real dx2 = size.d_view(m).dx2;
      Real dx3 = size.d_view(m).dx3;
      Real divb = (b0.x1f(m,k,j,i+1) - b0.x1f(m,k,j,i))/dx1
                + (b0.x2f(m,k,j+1,i) - b0.x2f(m,k,j,i))/dx2
                + (b0.x3f(m,k+1,j,i) - b0.x3f(m,k,j,i))/dx3;
      Real bmag = sqrt(SQR(bcc_(m,IBX,k,j,i)) + SQR(bcc_(m,IBY,k,j,i))
                     + SQR(bcc_(m,IBZ,k,j,i)));
      dmax = fmax(fabs(divb), dmax);
      if (bmag > 0.0) {
        rmax = fmax(fabs(divb)*fmin(fmin(dx1,dx2),dx3)/bmag, rmax);
      }
    }, Kokkos::Max<Real>(divb_max), Kokkos::Max<Real>(divb_rel_max));

#if MPI_PARALLEL_ENABLED
    Real dsums[2] = {divb_max, divb_rel_max};
    MPI_Allreduce(MPI_IN_PLACE, dsums, 2, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
    divb_max = dsums[0]; divb_rel_max = dsums[1];
#endif

    if (global_variable::my_rank == 0) {
      std::cout << "<" << bname << "> field=bfld"
                << " nmode=" << fld.nmode
                << " inv_beta=" << target
                << " norm=" << s
                << " divb_max=" << divb_max
                << " divb_rel_max=" << divb_rel_max << std::endl;
    }
  }


#endif  // PGEN_TURB_SEED_HPP_
