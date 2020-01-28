#pragma once

#include "gradient.hpp"
#include <vector>

namespace timemachine {

template<typename RealType, int D>
class Nonbonded : public Gradient<RealType, D> {

private:

    int *d_charge_param_idxs_;
    int *d_lj_param_idxs_;
    int *d_exclusion_idxs_; // [E,2]
    int *d_charge_scale_idxs_; // [E]
    int *d_lj_scale_idxs_; // [E]

    double cutoff_;

    // these buffers can be in RealType as well
    RealType *d_block_bounds_ctr_;
    RealType *d_block_bounds_ext_;

    const int E_;
    const int N_;

public:

    Nonbonded(
        const std::vector<int> &charge_param_idxs, // [N]
        const std::vector<int> &lj_param_idxs, // [N]
        const std::vector<int> &exclusion_idxs, // [E,2]
        const std::vector<int> &charge_scale_idxs, // [E]
        const std::vector<int> &lj_scale_idxs, // [E]
        double cutoff);

    ~Nonbonded();

    /*
    Execute the force computation, the semantics are:

    1. If d_coords_tangents == null, then out_coords != null, out_coords_tangent == null, out_params_tangents == null
    2. If d_coords_tangents != null, then out_coords == null, out_coords_tangent != null, out_params_tangents != null

    */
    virtual void execute_device(
        const int N,
        const int P,
        const RealType *d_coords,
        const RealType *d_coords_tangents,
        const RealType *d_params,
        unsigned long long *out_coords,
        RealType *out_coords_tangents,
        RealType *out_params_tangents
    ) override;


};


}