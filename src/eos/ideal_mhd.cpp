//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ideal_mhd.cpp
//! \brief derived class that implements ideal gas EOS in nonrelativistic mhd

#include "athena.hpp"
#include "mhd/mhd.hpp"
#include "eos.hpp"
#include "eos/ideal_c2p_mhd.hpp"
#include "coordinates/cell_locations.hpp"

//----------------------------------------------------------------------------------------
// ctor: also calls EOS base class constructor

IdealMHD::IdealMHD(MeshBlockPack *pp, ParameterInput *pin) :
    EquationOfState("mhd", pp, pin) {
  eos_data.is_ideal = true;
  eos_data.gamma = pin->GetReal("mhd","gamma");
  eos_data.iso_cs = 0.0;
  eos_data.use_e = true;  // ideal gas EOS always uses internal energy
  eos_data.use_t = false;
}

//----------------------------------------------------------------------------------------
//! \!fn void ConsToPrim()
//! \brief Converts conserved into primitive variables.  Operates over range of cells
//! given in argument list.

void IdealMHD::ConsToPrim(DvceArray5D<Real> &cons, const DvceFaceFld4D<Real> &b,
                          DvceArray5D<Real> &prim, DvceArray5D<Real> &bcc,
                          const bool only_testfloors,
                          const int il, const int iu, const int jl, const int ju,
                          const int kl, const int ku) {
  int &nmhd  = pmy_pack->pmhd->nmhd;
  int &nscal = pmy_pack->pmhd->nscalars;
  int &nmb = pmy_pack->nmb_thispack;
  auto &eos = eos_data;
  auto &fofc_ = pmy_pack->pmhd->fofc;

  const int ni   = (iu - il + 1);
  const int nji  = (ju - jl + 1)*ni;
  const int nkji = (ku - kl + 1)*nji;
  const int nmkji = nmb*nkji;

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size = pmy_pack->pmb->mb_size;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int ng = indcs.ng;
  Real gm1 = (eos.gamma - 1.0);
  Real r_in = eos.r_in;
  Real efloor = eos.pfloor/gm1;
  Real &tfloor = eos.tfloor;
  Real &daverage = eos.daverage;
  Real &rdfloor = eos.rdfloor;

  int nfloord_=0, nfloore_=0, nfloort_=0;
  Kokkos::parallel_reduce("mhd_c2p",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, int &sumd, int &sume, int &sumt) {
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/ni;
    int i = (idx - m*nkji - k*nji - j*ni) + il;
    j += jl;
    k += kl;

    // load single state conserved variables
    MHDCons1D u;
    u.d  = cons(m,IDN,k,j,i);
    u.mx = cons(m,IM1,k,j,i);
    u.my = cons(m,IM2,k,j,i);
    u.mz = cons(m,IM3,k,j,i);
    u.e  = cons(m,IEN,k,j,i);

    // load cell-centered fields into conserved state
    // use input CC fields if only testing floors with FOFC
    if (only_testfloors) {
      u.bx = bcc(m,IBX,k,j,i);
      u.by = bcc(m,IBY,k,j,i);
      u.bz = bcc(m,IBZ,k,j,i);
    // else use simple linear average of face-centered fields
    } else {
      u.bx = 0.5*(b.x1f(m,k,j,i) + b.x1f(m,k,j,i+1));
      u.by = 0.5*(b.x2f(m,k,j,i) + b.x2f(m,k,j+1,i));
      u.bz = 0.5*(b.x3f(m,k,j,i) + b.x3f(m,k+1,j,i));
    }

    // call c2p function
    // (inline function in ideal_c2p_mhd.hpp file)
    HydPrim1D w;
    bool dfloor_used=false, efloor_used=false, tfloor_used=false;
    SingleC2P_IdealMHD(u, eos, w, dfloor_used, efloor_used, tfloor_used);

    // set FOFC flag and quit loop if this function called only to check floors
    if (only_testfloors) {
      if (dfloor_used || efloor_used || tfloor_used) {
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
      prim(m,IDN,k,j,i) = w.d;
      prim(m,IVX,k,j,i) = w.vx;
      prim(m,IVY,k,j,i) = w.vy;
      prim(m,IVZ,k,j,i) = w.vz;
      prim(m,IEN,k,j,i) = w.e;
      // store cell-centered fields in 3D array
      bcc(m,IBX,k,j,i) = u.bx;
      bcc(m,IBY,k,j,i) = u.by;
      bcc(m,IBZ,k,j,i) = u.bz;
      // convert scalars (if any), always stored at end of cons and prim arrays.
      for (int n=nmhd; n<(nmhd+nscal); ++n) {
        // apply scalar floor
        if (cons(m,n,k,j,i) < 0.0) {
          cons(m,n,k,j,i) = 0.0;
        }
        prim(m,n,k,j,i) = cons(m,n,k,j,i)/u.d;
      }
    }
  }, Kokkos::Sum<int>(nfloord_), Kokkos::Sum<int>(nfloore_), Kokkos::Sum<int>(nfloort_));

  int sum_0=0, sum_1=0, sum_2=0;
  Kokkos::parallel_reduce("mhd_c2p_fix",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, int &sum0, int &sum1, int &sum2) {
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/ni;
    int i = (idx - m*nkji - k*nji - j*ni) + il;
    j += jl;
    k += kl;

    // load single state conserved variables
    MHDCons1D u;
    u.d  = cons(m,IDN,k,j,i);
    u.mx = cons(m,IM1,k,j,i);
    u.my = cons(m,IM2,k,j,i);
    u.mz = cons(m,IM3,k,j,i);
    u.e  = cons(m,IEN,k,j,i);

    // load cell-centered fields into conserved state
    // use input CC fields if only testing floors with FOFC
    if (only_testfloors) {
      u.bx = bcc(m,IBX,k,j,i);
      u.by = bcc(m,IBY,k,j,i);
      u.bz = bcc(m,IBZ,k,j,i);
    // else use simple linear average of face-centered fields
    } else {
      u.bx = 0.5*(b.x1f(m,k,j,i) + b.x1f(m,k,j,i+1));
      u.by = 0.5*(b.x2f(m,k,j,i) + b.x2f(m,k,j+1,i));
      u.bz = 0.5*(b.x3f(m,k,j,i) + b.x3f(m,k+1,j,i));
    }

    MHDPrim1D w;
    w.d  = prim(m,IDN,k,j,i);
    w.vx = prim(m,IVX,k,j,i);
    w.vy = prim(m,IVY,k,j,i);
    w.vz = prim(m,IVZ,k,j,i);
    w.e  = prim(m,IEN,k,j,i);

    // (@mhguo): beg fofc criteria
    bool fofc_flag = false;
    bool rdf_flag = false, ave_flag = false, ceil_flag = false;
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

        if (rad < 2.0*r_in) {
          sum0++;
          fofc_flag = true;
        }

        if (u.d < rdfloor/rad && rad>1.1*r_in) {
          sum1++;
          fofc_flag = true;
        }
        if (w.d <= 1e3*eos.rdfloor/rad && rad>1.1*r_in) {
          Real w_dkm = (k>ks-ng)? prim(m,IDN,k-1,j,i) : w.d;
          Real w_dkp = (k<ke+ng)? prim(m,IDN,k+1,j,i) : w.d;
          Real w_djm = (j>js-ng)? prim(m,IDN,k,j-1,i) : w.d;
          Real w_djp = (j<je+ng)? prim(m,IDN,k,j+1,i) : w.d;
          Real w_dim = (i>is-ng)? prim(m,IDN,k,j,i-1) : w.d;
          Real w_dip = (i<ie+ng)? prim(m,IDN,k,j,i+1) : w.d;
          if (w.d/w_dkm + w_dkm/w.d > 1e2 || w.d/w_dkp + w_dkp/w.d > 1e2 ||
              w.d/w_djm + w_djm/w.d > 1e2 || w.d/w_djp + w_djp/w.d > 1e2 ||
              w.d/w_dim + w_dim/w.d > 1e2 || w.d/w_dip + w_dip/w.d > 1e2) {
            sum2++;
            fofc_flag = true;
          }
        }
      }
      if (fofc_flag) {
        fofc_(m,k,j,i) = true;
      }
    } else {
      Real dave = daverage;
      // (@mhguo) r-dependent floor
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

      if (rdfloor>0.0) {
        if (u.d < rdfloor/rad && rad>1.1*r_in) {
          u.d = rdfloor/rad;
          w.d = rdfloor/rad;
          dave = rdfloor/rad;
          rdf_flag = true;
          sum0++;
        }
      }
      // apply cell averaging
      if (u.d <= dave && rad>1.1*r_in && k>kl && k<ku && j>jl && j<ju && i>il && i<iu) {
        MHDCons1D ukm, ukp, ujm, ujp, uim, uip;
        ukm.d  = cons(m,IDN,k-1,j,i);
        ukm.mx = cons(m,IM1,k-1,j,i);
        ukm.my = cons(m,IM2,k-1,j,i);
        ukm.mz = cons(m,IM3,k-1,j,i);
        ukm.e  = cons(m,IEN,k-1,j,i);
        ukm.bx = 0.5*(b.x1f(m,k-1,j,i) + b.x1f(m,k-1,j,i+1));
        ukm.by = 0.5*(b.x2f(m,k-1,j,i) + b.x2f(m,k-1,j+1,i));
        ukm.bz = 0.5*(b.x3f(m,k-1,j,i) + b.x3f(m,k    ,j,i));

        ukp.d  = cons(m,IDN,k+1,j,i);
        ukp.mx = cons(m,IM1,k+1,j,i);
        ukp.my = cons(m,IM2,k+1,j,i);
        ukp.mz = cons(m,IM3,k+1,j,i);
        ukp.e  = cons(m,IEN,k+1,j,i);
        ukp.bx = 0.5*(b.x1f(m,k+1,j,i) + b.x1f(m,k+1,j,i+1));
        ukp.by = 0.5*(b.x2f(m,k+1,j,i) + b.x2f(m,k+1,j+1,i));
        ukp.bz = 0.5*(b.x3f(m,k+1,j,i) + b.x3f(m,k+2  ,j,i));

        ujm.d  = cons(m,IDN,k,j-1,i);
        ujm.mx = cons(m,IM1,k,j-1,i);
        ujm.my = cons(m,IM2,k,j-1,i);
        ujm.mz = cons(m,IM3,k,j-1,i);
        ujm.e  = cons(m,IEN,k,j-1,i);
        ujm.bx = 0.5*(b.x1f(m,k,j-1,i) + b.x1f(m,k,j-1,i+1));
        ujm.by = 0.5*(b.x2f(m,k,j-1,i) + b.x2f(m,k,j    ,i));
        ujm.bz = 0.5*(b.x3f(m,k,j-1,i) + b.x3f(m,k+1,j-1,i));

        ujp.d  = cons(m,IDN,k,j+1,i);
        ujp.mx = cons(m,IM1,k,j+1,i);
        ujp.my = cons(m,IM2,k,j+1,i);
        ujp.mz = cons(m,IM3,k,j+1,i);
        ujp.e  = cons(m,IEN,k,j+1,i);
        ujp.bx = 0.5*(b.x1f(m,k,j+1,i) + b.x1f(m,k,j+1,i+1));
        ujp.by = 0.5*(b.x2f(m,k,j+1,i) + b.x2f(m,k,j+2  ,i));
        ujp.bz = 0.5*(b.x3f(m,k,j+1,i) + b.x3f(m,k+1,j+1,i));

        uim.d  = cons(m,IDN,k,j,i-1);
        uim.mx = cons(m,IM1,k,j,i-1);
        uim.my = cons(m,IM2,k,j,i-1);
        uim.mz = cons(m,IM3,k,j,i-1);
        uim.e  = cons(m,IEN,k,j,i-1);
        uim.bx = 0.5*(b.x1f(m,k,j,i-1) + b.x1f(m,k,j,i    ));
        uim.by = 0.5*(b.x2f(m,k,j,i-1) + b.x2f(m,k,j+1,i-1));
        uim.bz = 0.5*(b.x3f(m,k,j,i-1) + b.x3f(m,k+1,j,i-1));

        uip.d  = cons(m,IDN,k,j,i+1);
        uip.mx = cons(m,IM1,k,j,i+1);
        uip.my = cons(m,IM2,k,j,i+1);
        uip.mz = cons(m,IM3,k,j,i+1);
        uip.e  = cons(m,IEN,k,j,i+1);
        uip.bx = 0.5*(b.x1f(m,k,j,i+1) + b.x1f(m,k,j,i+2  ));
        uip.by = 0.5*(b.x2f(m,k,j,i+1) + b.x2f(m,k,j+1,i+1));
        uip.bz = 0.5*(b.x3f(m,k,j,i+1) + b.x3f(m,k+1,j,i+1));

        u.d = (ukm.d+ukp.d+ujm.d+ujp.d+uim.d+uip.d)/6.0;
        u.mx = (ukm.mx+ukp.mx+ujm.mx+ujp.mx+uim.mx+uip.mx)/6.0;
        u.my = (ukm.my+ukp.my+ujm.my+ujp.my+uim.my+uip.my)/6.0;
        u.mz = (ukm.mz+ukp.mz+ujm.mz+ujp.mz+uim.mz+uip.mz)/6.0;
        Real e_kkm  = 0.5*(SQR(ukm.mx) + SQR(ukm.my) + SQR(ukm.mz))/ukm.d;
        Real e_kkp  = 0.5*(SQR(ukp.mx) + SQR(ukp.my) + SQR(ukp.mz))/ukp.d;
        Real e_kjm  = 0.5*(SQR(ujm.mx) + SQR(ujm.my) + SQR(ujm.mz))/ujm.d;
        Real e_kjp  = 0.5*(SQR(ujp.mx) + SQR(ujp.my) + SQR(ujp.mz))/ujp.d;
        Real e_kim  = 0.5*(SQR(uim.mx) + SQR(uim.my) + SQR(uim.mz))/uim.d;
        Real e_kip  = 0.5*(SQR(uip.mx) + SQR(uip.my) + SQR(uip.mz))/uip.d;
        Real e_mkm  = 0.5*(SQR(ukm.bx) + SQR(ukm.by) + SQR(ukm.bz));
        Real e_mkp  = 0.5*(SQR(ukp.bx) + SQR(ukp.by) + SQR(ukp.bz));
        Real e_mjm  = 0.5*(SQR(ujm.bx) + SQR(ujm.by) + SQR(ujm.bz));
        Real e_mjp  = 0.5*(SQR(ujp.bx) + SQR(ujp.by) + SQR(ujp.bz));
        Real e_mim  = 0.5*(SQR(uim.bx) + SQR(uim.by) + SQR(uim.bz));
        Real e_mip  = 0.5*(SQR(uip.bx) + SQR(uip.by) + SQR(uip.bz));
        Real u_ekm  = ukm.e - e_kkm - e_mkm;
        Real u_ekp  = ukp.e - e_kkp - e_mkp;
        Real u_ejm  = ujm.e - e_kjm - e_mjm;
        Real u_ejp  = ujp.e - e_kjp - e_mjp;
        Real u_eim  = uim.e - e_kim - e_mim;
        Real u_eip  = uip.e - e_kip - e_mip;
        Real e_k = 0.5*(u.mx*u.mx + u.my*u.my + u.mz*u.mz)/u.d;
        Real e_m = 0.5*(u.bx*u.bx + u.by*u.by + u.bz*u.bz);
        u.e = e_k+e_m+(u_ekm+u_ekp+u_ejm+u_ejp+u_eim+u_eip)/6.0;
        sum1++;
        ave_flag = true;
        w.d = u.d;
        w.vx = u.mx/u.d;
        w.vy = u.my/u.d;
        w.vz = u.mz/u.d;

        // set internal energy, apply floor, correcting total energy
        w.e = (u.e - e_k - e_m);
        if (w.e < efloor) {
          w.e = efloor;
          u.e = efloor + e_k + e_m;
        }

        // apply temperature floor
        Real tmp = gm1*w.e/w.d;
        if (tmp < tfloor) {
          w.e = w.d*tfloor/gm1;
          u.e = w.e + e_k + e_m;
        }
      }

      // apply ceiling
      Real vx1 = size.d_view(m).dx1/eos.dt_floor;
      Real vx2 = size.d_view(m).dx2/eos.dt_floor;
      Real vx3 = size.d_view(m).dx3/eos.dt_floor;
      Real vceil = fmin(fmin(vx1,vx2),vx3);
      Real dfloor_va = fmax(fmax(SQR(u.bx/vx1),SQR(u.by/vx2)),SQR(u.bz/vx3));
      if (rad < 2.0*r_in) {
        if (w.d<dfloor_va) {
          w.d = dfloor_va;
          ceil_flag = true;
        }
        if (fabs(w.vx)>vx1) {
          w.vx = (w.vx>0.0)*vx1;
          ceil_flag = true;
        }
        if (fabs(w.vy)>vx2) {
          w.vy = (w.vy>0.0)*vx2;
          ceil_flag = true;
        }
        if (fabs(w.vz)>vx3) {
          w.vz = (w.vz>0.0)*vx3;
          ceil_flag = true;
        }
        if (w.e/w.d*gm1>SQR(vceil)) {
          w.e = w.d*SQR(vceil)/gm1;
          ceil_flag = true;
        }
        if (ceil_flag) {
          u.d = w.d;
          u.mx = w.vx*w.d;
          u.my = w.vy*w.d;
          u.mz = w.vz*w.d;
          Real e_k = 0.5*(w.vx*w.vx + w.vy*w.vy + w.vz*w.vz)*w.d;
          Real e_m = 0.5*(u.bx*u.bx + u.by*u.by + u.bz*u.bz);
          u.e = w.e + e_k + e_m;
          sum2++;
        }
      }

      // apply all the fixes
      if (rdf_flag || ave_flag || ceil_flag) {
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
      }
    }
  }, Kokkos::Sum<int>(sum_0), Kokkos::Sum<int>(sum_1), Kokkos::Sum<int>(sum_2));

  // store appropriate counters
  if (only_testfloors) {
    pmy_pack->pmesh->ecounter.nfofc += nfloord_;
    pmy_pack->pmesh->ecounter.nfofc_d += sum_0;
    pmy_pack->pmesh->ecounter.nfofc_p += sum_1;
    pmy_pack->pmesh->ecounter.nfofc_g += sum_2;
  } else {
    pmy_pack->pmesh->ecounter.neos_dfloor += nfloord_;
    pmy_pack->pmesh->ecounter.neos_efloor += nfloore_;
    pmy_pack->pmesh->ecounter.neos_tfloor += nfloort_;
    pmy_pack->pmesh->ecounter.neos_rdfloor += sum_0;
    pmy_pack->pmesh->ecounter.neos_avfloor += sum_1;
    pmy_pack->pmesh->ecounter.neos_dtfloor += sum_2;
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \!fn void PrimToCons()
//! \brief Converts conserved into primitive variables.  Operates over range of cells
//! given in argument list.  Does not change cell- or face-centered magnetic fields.

void IdealMHD::PrimToCons(const DvceArray5D<Real> &prim, const DvceArray5D<Real> &bcc,
                          DvceArray5D<Real> &cons, const int il, const int iu,
                          const int jl, const int ju, const int kl, const int ku) {
  int &nmhd  = pmy_pack->pmhd->nmhd;
  int &nscal = pmy_pack->pmhd->nscalars;
  int &nmb = pmy_pack->nmb_thispack;

  par_for("mhd_p2c", DevExeSpace(), 0, (nmb-1), kl, ku, jl, ju, il, iu,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // load single state primitive variables
    MHDPrim1D w;
    w.d  = prim(m,IDN,k,j,i);
    w.vx = prim(m,IVX,k,j,i);
    w.vy = prim(m,IVY,k,j,i);
    w.vz = prim(m,IVZ,k,j,i);
    w.e  = prim(m,IEN,k,j,i);

    // load cell-centered fields into primitive state
    w.bx = bcc(m,IBX,k,j,i);
    w.by = bcc(m,IBY,k,j,i);
    w.bz = bcc(m,IBZ,k,j,i);

    // call p2c function
    HydCons1D u;
    SingleP2C_IdealMHD(w, u);

    // store conserved state in 3D array
    cons(m,IDN,k,j,i) = u.d;
    cons(m,IM1,k,j,i) = u.mx;
    cons(m,IM2,k,j,i) = u.my;
    cons(m,IM3,k,j,i) = u.mz;
    cons(m,IEN,k,j,i) = u.e;

    // convert scalars (if any), always stored at end of cons and prim arrays.
    for (int n=nmhd; n<(nmhd+nscal); ++n) {
      cons(m,n,k,j,i) = u.d*prim(m,n,k,j,i);
    }
  });

  return;
}
