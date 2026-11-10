#include <turboreg/sim3.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>

int main(int argc, char** argv) {
    if (argc < 3 || argc > 4) {
        std::cerr << "Usage: turboreg_sim3 CORRESPONDENCES.txt TRANSFORM.txt [cpu|cuda:0]\n"
                  << "Each row: src_x src_y src_z dst_x dst_y dst_z\n";
        return 2;
    }
    try {
        std::ifstream file(argv[1]);
        if (!file) throw std::runtime_error("Cannot open correspondence file.");
        std::vector<float> values;
        std::string line;
        while(std::getline(file,line)) {
            if(line.find_first_not_of(" \t\r")==std::string::npos) continue;
            std::istringstream row(line);
            for(int i=0;i<6;++i) {
                float v;
                if(!(row>>v)) throw std::runtime_error("Expected six numbers per row.");
                values.push_back(v);
            }
            std::string extra;
            if(row>>extra) throw std::runtime_error("Expected exactly six columns.");
        }
        auto points=torch::from_blob(values.data(),{static_cast<int64_t>(values.size()/6),6},torch::kFloat32).clone();
        auto device=torch::Device(argc==4?argv[3]:"cpu");
        auto r=turboreg::register_sim3(points.slice(1,0,3).to(device),points.slice(1,3,6).to(device));
        auto T=r.transformation.cpu();
        auto a=T.accessor<float,2>();
        std::ofstream output(argv[2]);
        if(!output) throw std::runtime_error("Cannot open output file.");
        output<<std::setprecision(10);
        for(int i=0;i<4;++i) { for(int j=0;j<4;++j) output<<a[i][j]<<(j==3?'\n':' '); }
        if(!output) throw std::runtime_error("Failed writing output.");
        std::cout<<"scale="<<r.scale<<" inliers="<<r.num_inliers<<" input="<<r.num_input<<" used="<<r.num_used<<"\n";
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
