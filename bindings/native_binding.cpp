#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <turboreg/sim3_native.hpp>
namespace py=pybind11;
using namespace turboreg::native;
PYBIND11_MODULE(turboreg_native,m) {
    m.doc()="Deterministic C++/Eigen CPU implementation of the TurboReg Sim3 scale graph pipeline";
    m.def("register_sim3",[](py::array_t<double,py::array::c_style|py::array::forcecast> a,
                              py::array_t<double,py::array::c_style|py::array::forcecast> b,py::dict parameters) {
        if(a.ndim()!=2||b.ndim()!=2||a.shape(1)!=3||b.shape(1)!=3||a.shape(0)!=b.shape(0))
            throw std::invalid_argument("Expected matching (N,3) arrays");
        Options o;
        for(auto entry:parameters) {
            const auto key=py::cast<std::string>(entry.first);const auto value=entry.second;
#define OPT(name) if(key==#name) {o.name=py::cast<decltype(o.name)>(value);continue;}
            OPT(max_points) OPT(num_pivots) OPT(k_neighbors) OPT(graph_iterations) OPT(refinement_iterations)
            OPT(epsilon_scale) OPT(nms_radius) OPT(inlier_threshold) OPT(min_pair_distance)
            OPT(scale_min) OPT(scale_max) OPT(log_scale_bin_width) OPT(metric) OPT(local_scale_mode)
            OPT(use_tsc) OPT(use_second_order) OPT(reject_degenerate) OPT(uniform_refinement)
#undef OPT
            throw std::invalid_argument("Unknown native option: "+key);
        }
        Result r;
        {py::gil_scoped_release release;
         Eigen::Map<const Points> x(a.data(),a.shape(0),3),y(b.data(),b.shape(0),3);
         r=register_sim3(x,y,o);}
        py::array_t<double> transform({4,4});auto t=transform.mutable_unchecked<2>();
        for(int i=0;i<4;++i) for(int j=0;j<4;++j) t(i,j)=r.transformation(i,j);
        py::array_t<bool> mask(r.num_input);auto ptr=mask.mutable_data();
        for(int i=0;i<r.num_input;++i) ptr[i]=r.inlier_mask[i];
        py::dict out;out["transformation"]=transform;out["inlier_mask"]=mask;out["scale"]=r.scale;
        out["num_input"]=r.num_input;out["num_used"]=r.num_used;out["num_inliers"]=r.num_inliers;
        return out;
    },py::arg("src"),py::arg("dst"),py::arg("options")=py::dict());
}
