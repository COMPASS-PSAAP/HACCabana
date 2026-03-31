#ifndef FORCE_SOLVERS_H
#define FORCE_SOLVERS_H

#include <memory>
#include <stdexcept>
#include <string>

#include <Cabana_Core.hpp>

#include "HACCabana_Config.h"
#include "HACCabana_ExactForceSolver.h"
#include "HACCabana_P3MForceSolver.h"
#ifdef HACCabana_ENABLE_CANOPY
#include "HACCabana_CanopyForceSolver.h"
#endif

namespace HACCabana
{

enum class force_solver_type
{
  p3m,
  exact,
  fmm
};

inline std::string to_string( const force_solver_type solver_type )
{
  switch ( solver_type )
  {
    case force_solver_type::p3m:
      return "p3m";
    case force_solver_type::exact:
      return "exact";
    case force_solver_type::fmm:
      return "fmm";
  }

  throw std::runtime_error( "Unknown force solver type." );
}

inline force_solver_type parse_force_solver_type( const std::string& solver_name )
{
  if ( solver_name == "p3m" )
    return force_solver_type::p3m;
  if ( solver_name == "exact" )
    return force_solver_type::exact;
  if ( solver_name == "fmm" )
    return force_solver_type::fmm;

  throw std::runtime_error(
      "Invalid force solver '" + solver_name +
      "'. Expected 'p3m', 'exact', or 'fmm'." );
}

template <class AoSoAType, class Field>
class RuntimeForceSolver
{
  private:
  class ForceSolverInterface
  {
    public:
    virtual ~ForceSolverInterface() noexcept = default;

    virtual void setup_subcycle( AoSoAType& aosoa_device, const float c,
                                 const float cm_size, const float min_pos,
                                 const float max_pos, const float rmax2,
                                 const float rsm2 ) = 0;
    virtual void updateVel( std::shared_ptr<AoSoAType> aosoa_device ) = 0;
  };

  template <class SolverImpl>
  class ForceSolverModel : public ForceSolverInterface
  {
    public:
    ~ForceSolverModel() noexcept override = default;

    void setup_subcycle( AoSoAType& aosoa_device, const float c,
                         const float cm_size, const float min_pos,
                         const float max_pos, const float rmax2,
                         const float rsm2 ) override
    {
      _solver.setup_subcycle( aosoa_device, c, cm_size, min_pos, max_pos,
                              rmax2, rsm2 );
    }

    void updateVel( std::shared_ptr<AoSoAType> aosoa_device ) override
    {
      _solver.updateVel( aosoa_device );
    }

    SolverImpl& implementation()
    {
      return _solver;
    }

    private:
    SolverImpl _solver;
  };

  public:
  RuntimeForceSolver()
      : _solver_type( force_solver_type::p3m )
  {
    initializeSolver();
  }

  explicit RuntimeForceSolver( const force_solver_type solver_type )
      : _solver_type( solver_type )
  {
    initializeSolver();
  }

  void setForceSolverType( const force_solver_type solver_type )
  {
    if ( _solver_type == solver_type )
      return;

    _solver_type = solver_type;
    initializeSolver();
  }

  force_solver_type getForceSolverType() const
  {
    return _solver_type;
  }

  void setup_subcycle( AoSoAType& aosoa_device, const float c,
                       const float cm_size, const float min_pos,
                       const float max_pos, const float rmax2,
                       const float rsm2 )
  {
    _solver->setup_subcycle( aosoa_device, c, cm_size, min_pos, max_pos,
                             rmax2, rsm2 );
  }

  void updateVel( std::shared_ptr<AoSoAType> aosoa_device )
  {
    _solver->updateVel( aosoa_device );
  }

  void setLeafTiles( const int leaf_tiles )
  {
    _leaf_tiles = leaf_tiles;
    if ( _solver_type == force_solver_type::fmm )
      initializeSolver();
  }

  void setReductionFactor( const int reduction_factor )
  {
    _reduction_factor = reduction_factor;
    if ( _solver_type == force_solver_type::fmm )
      initializeSolver();
  }

  private:
  void initializeSolver()
  {
    switch ( _solver_type )
    {
      case force_solver_type::p3m:
        _solver = std::make_unique<ForceSolverModel<P3MForceSolver<AoSoAType, Field>>>();
        return;
      case force_solver_type::exact:
        _solver = std::make_unique<ForceSolverModel<ExactForceSolver<AoSoAType, Field>>>();
        return;
      case force_solver_type::fmm:
#ifdef HACCabana_ENABLE_CANOPY
        {
          auto solver = std::make_unique<
              ForceSolverModel<CanopyForceSolver<AoSoAType, Field>>>();
          solver->implementation().setLeafTiles( _leaf_tiles );
          solver->implementation().setReductionFactor( _reduction_factor );
          _solver = std::move( solver );
        }
        return;
#else
        throw std::runtime_error(
            "FMM solver requested, but HACCabana was built without Canopy support." );
#endif
    }

    throw std::runtime_error( "Unknown force solver type." );
  }

  force_solver_type _solver_type;
  int _leaf_tiles = 16;
  int _reduction_factor = 2;
  std::unique_ptr<ForceSolverInterface> _solver;
};

} // end namespace HACCabana

#endif
