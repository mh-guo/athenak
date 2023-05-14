//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ideal_hyd.cpp
//! \brief derived class that implements ideal gas EOS in nonrelativistic hydro

#include "athena.hpp"
#include "hydro/hydro.hpp"
#include "eos/eos.hpp"
#include "eos/ideal_c2p_hyd.hpp"
#include "coordinates/cell_locations.hpp"

//----------------------------------------------------------------------------------------
// ctor: also calls EOS base class constructor

IdealHydro::IdealHydro(MeshBlockPack *pp, ParameterInput *pin) :
    EquationOfState("hydro", pp, pin) {
  eos_data.is_ideal = true;
  eos_data.gamma = pin->GetReal("hydro","gamma");
  eos_data.iso_cs = 0.0;
  eos_data.use_e = true;  // ideal gas EOS always uses internal energy
  eos_data.use_t = false;
}

//----------------------------------------------------------------------------------------
//! \fn void ConsToPrim()
//! \brief Converts conserved into primitive variables. Operates over range of cells given
//! in argument list. Number of times floors used stored into event counters.

void IdealHydro::ConsToPrim(DvceArray5D<Real> &cons, DvceArray5D<Real> &prim,
                            const bool only_testfloors,
                            const int il, const int iu, const int jl, const int ju,
                            const int kl, const int ku) {
  int &nhyd  = pmy_pack->phydro->nhydro;
  int &nscal = pmy_pack->phydro->nscalars;
  int &nmb = pmy_pack->nmb_thispack;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  auto &size = pmy_pack->pmb->mb_size;
  Real r_in = eos_data.r_in;
  Real gm1 = (eos_data.gamma - 1.0);

  Real efloor = eos_data.pfloor/gm1;
  Real &tfloor = eos_data.tfloor;
  Real &daverage = eos_data.daverage;
  Real &rdfloor = eos_data.rdfloor;
  auto &eos = eos_data;
  auto &fofc_ = pmy_pack->phydro->fofc;

  const int ni   = (iu - il + 1);
  const int nji  = (ju - jl + 1)*ni;
  const int nkji = (ku - kl + 1)*nji;
  const int nmkji = nmb*nkji;

  int nfloord_=0, nfloore_=0, nfloort_=0;
  int nfofc_d=0, nfofc_p=0, nfofc_g=0;
  Kokkos::parallel_reduce("hyd_c2p",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, int &sumd, int &sume, int &sumt,
  int &sum0, int &sum1, int &sum2) {
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/ni;
    int i = (idx - m*nkji - k*nji - j*ni) + il;
    j += jl;
    k += kl;

    // load single state conserved variables
    HydCons1D u;
    u.d  = cons(m,IDN,k,j,i);
    u.mx = cons(m,IM1,k,j,i);
    u.my = cons(m,IM2,k,j,i);
    u.mz = cons(m,IM3,k,j,i);
    u.e  = cons(m,IEN,k,j,i);

    // call c2p function
    // (inline function in ideal_c2p_hyd.hpp file)
    HydPrim1D w;
    bool dfloor_used=false, efloor_used=false, tfloor_used=false;
    SingleC2P_IdealHyd(u, eos, w, dfloor_used, efloor_used, tfloor_used);

    // (@mhguo): beg fofc criteria
    bool fofc_flag = false;
    if (only_testfloors) {
      if (rdfloor>0.0) {
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
        //int *psum_0 = &sum0;
        //int *psum_1 = &sum1;
        //int *psum_2 = &sum2;
        //if (FofcCrit(u,eos,psum_0,psum_1,psum_2,r_in,rad)) {
        //  fofc_flag = true;
        //}

        if (u.d < rdfloor/rad && rad>1.1*r_in) {
          sum1++;
          fofc_flag = true;
        }
        if (w.d <= 1e3*eos.rdfloor/rad && rad>1.1*r_in) {
          Real dfl = eos.dfloor;
          Real w_dkm = fmax(cons(m,IDN,k-1,j,i),dfl);
          Real w_dkp = fmax(cons(m,IDN,k+1,j,i),dfl);
          Real w_djm = fmax(cons(m,IDN,k,j-1,i),dfl);
          Real w_djp = fmax(cons(m,IDN,k,j+1,i),dfl);
          Real w_dim = fmax(cons(m,IDN,k,j,i-1),dfl);
          Real w_dip = fmax(cons(m,IDN,k,j,i+1),dfl);
          if (k<=ks) { w_dkm  = prim(m,IDN,k-1,j,i); }
          if (k>=ke) { w_dkp  = prim(m,IDN,k+1,j,i); }
          if (j<=js) { w_djm  = prim(m,IDN,k,j-1,i); }
          if (j>=je) { w_djp  = prim(m,IDN,k,j+1,i); }
          if (i<=is) { w_dim  = prim(m,IDN,k,j,i-1); }
          if (i>=ie) { w_dip  = prim(m,IDN,k,j,i+1); }
          if (w.d/w_dkm + w_dkm/w.d > 1e2 || w.d/w_dkp + w_dkp/w.d > 1e2 ||
              w.d/w_djm + w_djm/w.d > 1e2 || w.d/w_djp + w_djp/w.d > 1e2 ||
              w.d/w_dim + w_dim/w.d > 1e2 || w.d/w_dip + w_dip/w.d > 1e2) {
            sum2++;
            fofc_flag = true;
          }
        }
      }
    } else {
      // (@mhguo): beg of old floor
      Real dave = daverage;
      // (@mhguo) r-dependent floor
      if (rdfloor>0.0) {
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

        if (u.d < rdfloor/rad && rad>1.1*r_in) {
          u.d = rdfloor/rad;
          dave = rdfloor/rad;
        }
      }
      // apply cell averaging
      if (u.d <= dave && k>kl && k<ku && j>jl && j<ju && i>il && i<iu) {
        Real u_dkm  = fmax(cons(m,IDN,k-1,j,i),dave);
        Real u_dkp  = fmax(cons(m,IDN,k+1,j,i),dave);
        Real u_djm  = fmax(cons(m,IDN,k,j-1,i),dave);
        Real u_djp  = fmax(cons(m,IDN,k,j+1,i),dave);
        Real u_dim  = fmax(cons(m,IDN,k,j,i-1),dave);
        Real u_dip  = fmax(cons(m,IDN,k,j,i+1),dave);
        //u_d = 6.0/(1.0/u_dkm+1.0/u_dkp+1.0/u_djm+1.0/u_djp+1.0/u_dim+1.0/u_dip);
        u.d = (u_dkm+u_dkp+u_djm+u_djp+u_dim+u_dip)/6.0;
        Real& u_m1km = cons(m,IM1,k-1,j,i);
        Real& u_m1kp = cons(m,IM1,k+1,j,i);
        Real& u_m1jm = cons(m,IM1,k,j-1,i);
        Real& u_m1jp = cons(m,IM1,k,j+1,i);
        Real& u_m1im = cons(m,IM1,k,j,i-1);
        Real& u_m1ip = cons(m,IM1,k,j,i+1);
        u.mx = (u_m1km+u_m1kp+u_m1jm+u_m1jp+u_m1im+u_m1ip)/6.0;
        Real& u_m2km = cons(m,IM2,k-1,j,i);
        Real& u_m2kp = cons(m,IM2,k+1,j,i);
        Real& u_m2jm = cons(m,IM2,k,j-1,i);
        Real& u_m2jp = cons(m,IM2,k,j+1,i);
        Real& u_m2im = cons(m,IM2,k,j,i-1);
        Real& u_m2ip = cons(m,IM2,k,j,i+1);
        u.my = (u_m2km+u_m2kp+u_m2jm+u_m2jp+u_m2im+u_m2ip)/6.0;
        Real& u_m3km = cons(m,IM3,k-1,j,i);
        Real& u_m3kp = cons(m,IM3,k+1,j,i);
        Real& u_m3jm = cons(m,IM3,k,j-1,i);
        Real& u_m3jp = cons(m,IM3,k,j+1,i);
        Real& u_m3im = cons(m,IM3,k,j,i-1);
        Real& u_m3ip = cons(m,IM3,k,j,i+1);
        u.mz = (u_m3km+u_m3kp+u_m3jm+u_m3jp+u_m3im+u_m3ip)/6.0;
        Real e_kkm  = 0.5*(u_m1km*u_m1km + u_m2km*u_m2km + u_m3km*u_m3km)/u_dkm;
        Real e_kkp  = 0.5*(u_m1kp*u_m1kp + u_m2kp*u_m2kp + u_m3kp*u_m3kp)/u_dkp;
        Real e_kjm  = 0.5*(u_m1jm*u_m1jm + u_m2jm*u_m2jm + u_m3jm*u_m3jm)/u_djm;
        Real e_kjp  = 0.5*(u_m1jp*u_m1jp + u_m2jp*u_m2jp + u_m3jp*u_m3jp)/u_djp;
        Real e_kim  = 0.5*(u_m1im*u_m1im + u_m2im*u_m2im + u_m3im*u_m3im)/u_dim;
        Real e_kip  = 0.5*(u_m1ip*u_m1ip + u_m2ip*u_m2ip + u_m3ip*u_m3ip)/u_dip;
        Real u_ekm  = fmax(cons(m,IEN,k-1,j,i) - e_kkm, fmax(efloor, u_dkm*tfloor/gm1));
        Real u_ekp  = fmax(cons(m,IEN,k+1,j,i) - e_kkp, fmax(efloor, u_dkp*tfloor/gm1));
        Real u_ejm  = fmax(cons(m,IEN,k,j-1,i) - e_kjm, fmax(efloor, u_djm*tfloor/gm1));
        Real u_ejp  = fmax(cons(m,IEN,k,j+1,i) - e_kjp, fmax(efloor, u_djp*tfloor/gm1));
        Real u_eim  = fmax(cons(m,IEN,k,j,i-1) - e_kim, fmax(efloor, u_dim*tfloor/gm1));
        Real u_eip  = fmax(cons(m,IEN,k,j,i+1) - e_kip, fmax(efloor, u_dip*tfloor/gm1));
        Real e_k = 0.5*(u.mx*u.mx + u.my*u.my + u.mz*u.mz)/u.d;
        u.e = e_k+(u_ekm+u_ekp+u_ejm+u_ejp+u_eim+u_eip)/6.0;
      }
      w.d = u.d;

      Real di = 1.0/u.d;
      w.vx = u.mx*di;
      w.vy = u.my*di;
      w.vz = u.mz*di;

      // set internal energy, apply floor, correcting total energy
      Real e_k = 0.5*di*(u.mx*u.mx + u.my*u.my + u.mz*u.mz);
      w.e = (u.e - e_k);
      if (w.e < efloor) {
        w.e = efloor;
        u.e = efloor + e_k;
        sume++;
      }

      // apply temperature floor
      Real tmp = gm1*w.e*di;
      if (tmp < tfloor) {
        w.e = w.d*tfloor/gm1;
        u.e = w.e + e_k;
      }
      // (@mhguo): end of old floor
    }
    // set FOFC flag and quit loop if this function called only to check floors
    if (only_testfloors) {
      if (dfloor_used || fofc_flag) {
      //if (dfloor_used || efloor_used || tfloor_used) {
        fofc_(m,k,j,i) = true;
        sumd++;  // use dfloor as counter for when either is true
      }
    } else {
      // update counter, reset conserved if floor was hit
      if (dfloor_used) {
        cons(m,IDN,k,j,i) = u.d;
        sumd++;
      }
      if (efloor_used) {
        cons(m,IEN,k,j,i) = u.e;
        sume++;
      }
      if (tfloor_used) {
        cons(m,IEN,k,j,i) = u.e;
        sumt++;
      }
      // store primitive state in 3D array
      cons(m,IDN,k,j,i) = u.d;
      cons(m,IM1,k,j,i) = u.mx;
      cons(m,IM2,k,j,i) = u.my;
      cons(m,IM3,k,j,i) = u.mz;
      cons(m,IEN,k,j,i) = u.e;
      prim(m,IDN,k,j,i) = w.d;
      prim(m,IVX,k,j,i) = w.vx;
      prim(m,IVY,k,j,i) = w.vy;
      prim(m,IVZ,k,j,i) = w.vz;
      prim(m,IEN,k,j,i) = w.e;
      // convert scalars (if any)
      for (int n=nhyd; n<(nhyd+nscal); ++n) {
        // apply scalar floor
        if (cons(m,n,k,j,i) < 0.0) {
          cons(m,n,k,j,i) = 0.0;
        }
        prim(m,n,k,j,i) = cons(m,n,k,j,i)/u.d;
      }
    }
  }, Kokkos::Sum<int>(nfloord_), Kokkos::Sum<int>(nfloore_), Kokkos::Sum<int>(nfloort_),
  Kokkos::Sum<int>(nfofc_d), Kokkos::Sum<int>(nfofc_p), Kokkos::Sum<int>(nfofc_g));

  // store appropriate counters
  if (only_testfloors) {
    pmy_pack->pmesh->ecounter.nfofc += nfloord_;
    pmy_pack->pmesh->ecounter.nfofc_d += nfofc_d;
    pmy_pack->pmesh->ecounter.nfofc_p += nfofc_p;
    pmy_pack->pmesh->ecounter.nfofc_g += nfofc_g;
  } else {
    pmy_pack->pmesh->ecounter.neos_dfloor += nfloord_;
    pmy_pack->pmesh->ecounter.neos_efloor += nfloore_;
    pmy_pack->pmesh->ecounter.neos_tfloor += nfloort_;
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void PrimToCons()
//! \brief Converts primitive into conserved variables. Operates over range of cells given
//! in argument list.  Floors never needed.

void IdealHydro::PrimToCons(const DvceArray5D<Real> &prim, DvceArray5D<Real> &cons,
                            const int il, const int iu, const int jl, const int ju,
                            const int kl, const int ku) {
  int &nhyd  = pmy_pack->phydro->nhydro;
  int &nscal = pmy_pack->phydro->nscalars;
  int &nmb = pmy_pack->nmb_thispack;

  par_for("hyd_p2c", DevExeSpace(), 0, (nmb-1), kl, ku, jl, ju, il, iu,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // load single state primitive variables
    HydPrim1D w;
    w.d  = prim(m,IDN,k,j,i);
    w.vx = prim(m,IVX,k,j,i);
    w.vy = prim(m,IVY,k,j,i);
    w.vz = prim(m,IVZ,k,j,i);
    w.e  = prim(m,IEN,k,j,i);

    // call p2c function
    HydCons1D u;
    SingleP2C_IdealHyd(w, u);

    // store conserved state in 3D array
    cons(m,IDN,k,j,i) = u.d;
    cons(m,IM1,k,j,i) = u.mx;
    cons(m,IM2,k,j,i) = u.my;
    cons(m,IM3,k,j,i) = u.mz;
    cons(m,IEN,k,j,i) = u.e;

    // convert scalars (if any)
    for (int n=nhyd; n<(nhyd+nscal); ++n) {
      cons(m,n,k,j,i) = u.d*prim(m,n,k,j,i);
    }
  });

  return;
}
