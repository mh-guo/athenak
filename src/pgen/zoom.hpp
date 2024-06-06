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

typedef struct ZoomAMR {
  int max_level;                // maximum level number
  int min_level;                // minimum level number
  int level;                    // level number
  int zone;                     // zone number = level_max - level
  int direction;                // direction of zoom
  Real radius;                  // radius of inner boundary
  Real runtime;                 // interval for zoom
  Real last_time;               // time of last zoom
  Real next_time;               // time of next zoom
  bool just_zoomed;             // flag for just zoomed
} ZoomAMR;

typedef struct ZoomInterval {
  Real t_run_fac;            // interval factor
  Real t_run_pow;            // interval power law
  Real t_run_fac_zone_0;        // runtime factor for zone 0
  Real t_run_fac_zone_1;        // runtime factor for zone 1
  Real t_run_fac_zone_2;        // runtime factor for zone 2
  Real t_run_fac_zone_3;        // runtime factor for zone 3
  Real t_run_fac_zone_4;        // runtime factor for zone 4
  Real t_run_fac_zone_max;      // runtime factor for zone max
} ZoomInterval;

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
  bool restart;            // flag for restart
  bool fix_efield;
  int nlevels;             // number of levels
  int mzoom;               // number of zoom meshblocks
  int nvars;               // number of variables
  Real r_in;               // radius of iner boundary
  Real d_zoom;             // density within inner boundary
  Real p_zoom;             // pressure within inner boundary

  ZoomAMR zamr;            // zoom AMR parameters
  ZoomInterval zint;       // zoom interval parameters

  DvceArray5D<Real> u0;    // conserved variables
  DvceArray5D<Real> w0;    // primitive variables
  
  DvceArray5D<Real> coarse_u0;  // coarse conserved variables
  DvceArray5D<Real> coarse_w0;  // coarse primitive variables

  // following only used for time-evolving flow
  DvceEdgeFld4D<Real> efld;   // edge-centered electric fields (fluxes of B)

  // vector of SphericalGrid objects for analysis
  std::vector<std::unique_ptr<SphericalGrid>> spherical_grids;

  // array_sum::GlobalSum nc1, nc2, nc3, em1, em2, em3;

  // functions
  void Initialize();
  void PrintInfo();
  void BoundaryConditions();
  void AMR();
  void RefineCondition();
  void SetInterval();
  void UpdateVariables();
  void ApplyVariables();
  void FixEField(DvceEdgeFld4D<Real> emf);
  void ApplyEField(DvceEdgeFld4D<Real> emf);

 private:
  MeshBlockPack* pmy_pack;   // ptr to MeshBlockPack containing this MHD
};

} // namespace zoom
#endif // PGEN_ZOOM_HPP_