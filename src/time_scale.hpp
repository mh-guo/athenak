#ifndef TIME_SCALE_HPP_
#define TIME_SCALE_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file time_scale.hpp
//  \brief definitions for TimeScale class

#include <map>
#include <memory>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "bvals/bvals.hpp"

//----------------------------------------------------------------------------------------
//! \struct ScaleData
//! \brief container for scaling variables and functions needed inside kernels. Storing
//! everything in a container makes them easier to capture, and pass to inline functions,
//! inside kernels.


struct ScaleData {
  // data
  Real r0;                  // scaling radius

  KOKKOS_INLINE_FUNCTION
  Real ScaleFactor(const Real x1, const Real x2, const Real x3, const Real t) const {
    Real rad = fmax(sqrt(SQR(x1) + SQR(x2) + SQR(x3)), 4.0);
    // Real r_ks = sqrt((SQR(rad)-SQR(a)+sqrt(SQR(SQR(rad)-SQR(a))+4.0*SQR(a)*SQR(x3)))/2.0);
    Real a = 1.0 / (1.0 + r0 / rad);
    return a;
  }
};

//----------------------------------------------------------------------------------------
//! \class TimeScale

class TimeScale {
 public:
  TimeScale(MeshBlockPack *ppack, ParameterInput *pin);
  ~TimeScale();

  // data
  ScaleData scale_data;      // scaling parameters

  void ScaleEField(DvceEdgeFld4D<Real> emf);
  Real NewTimeStep(Mesh* pm);

 private:
  MeshBlockPack* pmy_pack;   // ptr to MeshBlockPack containing this TimeScale
};

#endif // TIME_SCALE_HPP_