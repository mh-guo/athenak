//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file time_scale.hpp
//  \brief implementation of functions in class Driver

#include <iostream>
#include <iomanip>    // std::setprecision()
#include <limits>
#include <algorithm>
#include <string> // string

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "outputs/outputs.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "z4c/z4c.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "ion-neutral/ion-neutral.hpp"
#include "radiation/radiation.hpp"

#include "time_scale.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

//----------------------------------------------------------------------------------------
//! \fn TimeScale::TimeScale
//! \brief constructor for TimeScale class
TimeScale::TimeScale(MeshBlockPack *ppack, ParameterInput *pin) :
  pmy_pack(ppack) {
  scale_data.r0 = pin->GetOrAddReal("time_scale", "scale_radius", 0.0);
  std::cout << "TimeScale: using scaling radius r0 = " << scale_data.r0 << std::endl;
}

//----------------------------------------------------------------------------------------
//! \fn void TimeScale::ScaleEField
//! \brief Modify E field on edges based on scaling factor

void TimeScale::ScaleEField(DvceEdgeFld4D<Real> emf) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  int nmb1 = pmy_pack->nmb_thispack-1;
  auto &size = pmy_pack->pmb->mb_size;
  auto &scaledata = scale_data;  // copy the object
  Real time = pmy_pack->pmesh->time;

  auto ef1 = emf.x1e;
  auto ef2 = emf.x2e;
  auto ef3 = emf.x3e;

  par_for("apply-emf", DevExeSpace(), 0, nmb1, ks-1, ke+1, js-1, je+1 ,is-1, ie+1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
    Real x1f = LeftEdgeX  (i-is, nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
    Real x2f = LeftEdgeX  (j-js, nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
    Real x3f = LeftEdgeX  (k-ks, nx3, x3min, x3max);

    // Scale E field components
    ef1(m,k,j,i) *= scaledata.ScaleFactor(x1v, x2f, x3f, time);
    ef2(m,k,j,i) *= scaledata.ScaleFactor(x1f, x2v, x3f, time);
    ef3(m,k,j,i) *= scaledata.ScaleFactor(x1f, x2f, x3v, time);
  });

  return;
}


//----------------------------------------------------------------------------------------
//! \fn TimeScale::NewTimeStep
//! \brief compute new timestep based on scaling factor
Real TimeScale::NewTimeStep(Mesh* pm) {
  Real &time = pm->time;
  Real &dt = pm->dt;
  return dt / pm->cfl_no / scale_data.ScaleFactor(0.0, 0.0, 8.0, time);
}

//----------------------------------------------------------------------------------------
//! \fn TimeScale::~TimeScale
//! \brief destructor for TimeScale class
TimeScale::~TimeScale() {
}
