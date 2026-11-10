#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <iostream>
#include <memory>
#include <turboreg/turboreg.hpp>
#include <turboreg/scale_turboreg.hpp>
#include <turboreg/sim3.hpp>

namespace py = pybind11;
using namespace turboreg;

void bind_turboreg_gpu(py::module &m)
{
     py::class_<TurboRegGPU>(m, "TurboRegGPU")
         .def(py::init<int, float, int, float, float, const std::string &>(),
              py::arg("max_N"),
              py::arg("tau_length_consis"),
              py::arg("num_pivot"),
              py::arg("radiu_nms"),
              py::arg("tau_inlier"),
              py::arg("metric_str"))
         .def("run_reg", &TurboRegGPU::runRegCXXReturnTensor,
              py::arg("kpts_src_all"), py::arg("kpts_dst_all"),
              "Run rigid registration.");

     py::class_<ScaleTurboRegGPU>(m, "ScaleTurboRegGPU")
         .def(py::init([](int max_N, float epsilon_scale, int num_pivot,
                          float radiu_nms, float tau_inlier,
                          const std::string &metric_str,
                          int k_neighbors, int num_iterations, int graph_k,
                          float trim_ratio, float min_pair_dist,
                          float local_scale_min, float local_scale_max,
                          float local_scale_diff,
                          const std::string &local_scale_mode,
                          bool use_tsc,
                          bool use_distance_diff_consistency,
                          float distance_diff_thresh,
                          bool use_second_scoring,
                          bool use_degeneracy_filter) {
                  (void)graph_k;
                  return std::make_unique<ScaleTurboRegGPU>(
                      max_N,
                      epsilon_scale,
                      num_pivot,
                      radiu_nms,
                      tau_inlier,
                      metric_str,
                      k_neighbors,
                      num_iterations,
                      trim_ratio,
                      min_pair_dist,
                      local_scale_min,
                      local_scale_max,
                      local_scale_diff,
                      local_scale_mode,
                      use_tsc,
                      use_distance_diff_consistency,
                      distance_diff_thresh,
                      use_second_scoring,
                      use_degeneracy_filter);
              }),
              py::arg("max_N"),
              py::arg("epsilon_scale"),
              py::arg("num_pivot"),
              py::arg("radiu_nms"),
              py::arg("tau_inlier"),
              py::arg("metric_str"),
              py::arg("k_neighbors") = 20,
              py::arg("num_iterations") = 2,
              py::arg("graph_k") = 20,

              py::arg("trim_ratio") = 0.1f,
              py::arg("min_pair_dist") = 1e-3f,
              py::arg("local_scale_min") = 0.1f,
              py::arg("local_scale_max") = 10.0f,
              py::arg("local_scale_diff") = -1.0f,
              py::arg("local_scale_mode") = "hist",
              py::arg("use_tsc") = true,
              py::arg("use_distance_diff_consistency") = false,
              py::arg("distance_diff_thresh") = -1.0f,
              py::arg("use_second_scoring") = true,
              py::arg("use_degeneracy_filter") = true)
         .def("run_reg", &ScaleTurboRegGPU::runRegCXXReturnTensor,
              py::arg("kpts_src_all"), py::arg("kpts_dst_all"),
              "Run Sim(3) registration.");


}

PYBIND11_MODULE(turboreg_gpu, m)
{
     m.doc() = "TurboReg bindings";
     bind_turboreg_gpu(m);
     py::class_<Sim3Options>(m, "Sim3Options")
         .def(py::init<>())
         .def_readwrite("max_points", &Sim3Options::max_points)
         .def_readwrite("epsilon_scale", &Sim3Options::epsilon_scale)
         .def_readwrite("num_pivots", &Sim3Options::num_pivots)
         .def_readwrite("nms_radius", &Sim3Options::nms_radius)
         .def_readwrite("inlier_threshold", &Sim3Options::inlier_threshold)
         .def_readwrite("metric", &Sim3Options::metric)
         .def_readwrite("k_neighbors", &Sim3Options::k_neighbors)
         .def_readwrite("graph_iterations", &Sim3Options::graph_iterations)
         .def_readwrite("refinement_iterations", &Sim3Options::refinement_iterations)
         .def_readwrite("min_pair_distance", &Sim3Options::min_pair_distance)
         .def_readwrite("scale_min", &Sim3Options::scale_min)
         .def_readwrite("scale_max", &Sim3Options::scale_max)
         .def_readwrite("log_scale_bin_width", &Sim3Options::log_scale_bin_width)
         .def_readwrite("local_scale_mode", &Sim3Options::local_scale_mode)
         .def_readwrite("use_tsc", &Sim3Options::use_tsc)
         .def_readwrite("use_second_order", &Sim3Options::use_second_order)
         .def_readwrite("reject_degenerate", &Sim3Options::reject_degenerate);
     py::class_<Sim3Result>(m, "Sim3Result")
         .def_readonly("transformation", &Sim3Result::transformation)
         .def_readonly("inlier_mask", &Sim3Result::inlier_mask)
         .def_readonly("num_input", &Sim3Result::num_input)
         .def_readonly("num_used", &Sim3Result::num_used)
         .def_readonly("num_inliers", &Sim3Result::num_inliers)
         .def_readonly("scale", &Sim3Result::scale);
     m.def("register_sim3", &register_sim3, py::arg("src"), py::arg("dst"),
           py::arg("options") = Sim3Options{}, py::call_guard<py::gil_scoped_release>());
     m.def("ransac_sim3", &ransac_sim3, py::arg("src"), py::arg("dst"),
           py::arg("options") = Sim3Options{}, py::arg("iterations") = 10000,
           py::arg("seed") = 42, py::call_guard<py::gil_scoped_release>());

}
