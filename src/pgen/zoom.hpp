#ifndef PGEN_ZOOM_HPP_
#define PGEN_ZOOM_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file zoom.hpp
//  \brief definitions for Zoom class

#include "geodesic-grid/spherical_grid.hpp"
#include "parameter_input.hpp"

namespace zoom {

//----------------------------------------------------------------------------------------
//! \struct ZoomAMR
//! \brief structure to hold zoom level information

struct ZoomAMR {
  int max_level;                // maximum level number
  int min_level;                // minimum level number
  int level;                    // level number
  int zone;                     // zone number = level_max - level
  int direction;                // direction of zoom
  Real radius;                  // radius of inner boundary
  Real interval;                // interval for zoom
  Real interval_fac;            // interval factor
  Real interval_pow;            // interval power law
  Real interval_fac_max_level;  // interval of maximum level
  Real interval_fac_min_level;  // interval of minimum level
  Real last_time;               // time of last zoom
  Real next_time;               // time of next zoom
  bool just_zoomed;             // flag for just zoomed
};

//----------------------------------------------------------------------------------------
//! \class Zoom

class Zoom;

class Zoom
{
 private:
  /* data */
 public:
  Zoom(MeshBlockPack *ppack, ParameterInput *pin);
  ~Zoom() = default;

  // data
  bool is_set;
  int nlevels;             // number of levels
  int nvars;               // number of variables
  Real r_in;               // radius of iner boundary
  Real d_zoom;             // density within inner boundary
  Real p_zoom;             // pressure within inner boundary

  ZoomAMR zamr;            // zoom AMR parameters

  DvceArray5D<Real> u0;    // conserved variables
  DvceArray5D<Real> w0;    // primitive variables
  
  DvceArray5D<Real> coarse_u0;  // coarse conserved variables
  DvceArray5D<Real> coarse_w0;  // coarse primitive variables

  // vector of SphericalGrid objects for analysis
  std::vector<std::unique_ptr<SphericalGrid>> spherical_grids;

  // functions
  void Initialize();
  void BoundaryConditions();
  void AMR();
  void Refine();
  void Interval();
  void UpdateVariables();
  void ApplyVariables();

 private:
  MeshBlockPack* pmy_pack;   // ptr to MeshBlockPack containing this MHD
};

} // namespace zoom
#endif // PGEN_ZOOM_HPP_