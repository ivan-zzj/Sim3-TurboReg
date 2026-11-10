#include <turboreg/sim3.hpp>
#include <turboreg/scale_turboreg.hpp>
#include <iostream>
#include <functional>
#include <cmath>

void check(bool condition, const char* message) {
    if(!condition) throw std::runtime_error(message);
}
void rejects(const std::function<void()>& f) {
    bool threw=false;
    try { f(); } catch(const std::exception&) { threw=true; }
    check(threw,"Expected invalid input or degenerate problem to fail.");
}
void run(torch::Device device) {
    torch::manual_seed(42);
    auto src=torch::randn({100,3}).to(device);
    auto original=src.clone();
    auto R=torch::tensor({{0.f,-1.f,0.f},{1.f,0.f,0.f},{0.f,0.f,1.f}}).to(device);
    auto t=torch::tensor({0.3f,-0.2f,0.7f}).to(device);
    turboreg::Sim3Options o;
    o.max_points=100; o.num_pivots=128; o.inlier_threshold=0.025f; o.nms_radius=0;
    for(float s:{0.05f,0.5f,1.f,2.f,12.f}) {
        o.scale_min=0.01f; o.scale_max=20.f;
        auto dst=s*src.matmul(R.t())+t;
        auto r=turboreg::register_sim3(src,dst,o);
        check(std::abs(r.scale-s)<1e-4,"Scale recovery failed (including outside legacy clamp).");
        check(torch::allclose(r.transformation.slice(0,0,3).slice(1,0,3),s*R,1e-4,1e-4),"Rotation recovery failed.");
        check(torch::allclose(r.transformation.index({torch::indexing::Slice(0,3),3}),t,1e-4,1e-4),"Translation recovery failed.");
        check(r.num_inliers==100,"Exact alignment inliers incorrect.");
    }
    auto dst=1.5f*src.matmul(R.t())+t;
    dst.slice(0,60,100).copy_(8*torch::randn({40,3}).to(device));
    auto robust=turboreg::register_sim3(src,dst,o);
    check(std::abs(robust.scale-1.5)<1e-4 && robust.num_inliers==60,"Outlier recovery failed.");
    auto ransac=turboreg::ransac_sim3(src,dst,o,1024,42);
    check(std::abs(ransac.scale-1.5)<1e-4 && ransac.num_inliers==60,"RANSAC recovery failed.");
    check(torch::equal(src,original),"Registration mutated caller input.");
    check(turboreg::register_sim3(src.to(torch::kFloat64),1.5*src.to(torch::kFloat64),o).num_inliers==100,"Float64 input failed.");
    auto triangle=torch::tensor({{0.f,0.f,0.f},{1.f,0.f,0.f},{0.f,1.f,0.f}}).to(device);
    check(turboreg::register_sim3(triangle,2*triangle,o).num_inliers==3,"Minimal three-point problem failed.");
    auto duplicate=torch::cat({triangle,triangle},0);
    check(turboreg::register_sim3(duplicate,2*duplicate,o).num_inliers==6,"Duplicate point handling failed.");
    rejects([&]{turboreg::register_sim3(src,src.slice(0,0,99),o);});
    rejects([&]{turboreg::register_sim3(src.slice(0,0,2),src.slice(0,0,2),o);});
    rejects([&]{turboreg::register_sim3(torch::zeros({5,3}).to(device),torch::zeros({5,3}).to(device),o);});
    auto line=torch::zeros({6,3}).to(device); line.select(1,0).copy_(torch::arange(6).to(device));
    rejects([&]{turboreg::register_sim3(line,2*line,o);});
    auto nan=src.clone(); nan[0][0]=NAN;
    rejects([&]{turboreg::register_sim3(nan,src,o);});
    o.k_neighbors=0;
    rejects([&]{turboreg::register_sim3(src,src,o);});
}
int main() {
    try {
        torch::set_num_threads(2);
        run(torch::kCPU);
        if(torch::cuda::is_available()) run(torch::kCUDA);
        else std::cout<<"CUDA unavailable: GPU checks skipped.\n";
        std::cout<<"Sim3 contract checks passed.\n";
        return 0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
