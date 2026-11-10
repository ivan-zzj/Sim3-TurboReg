// Standalone comparison solvers; no ground-truth inputs or TurboReg dependencies.
// FGR-style Geman--McClure continuation follows Zhou et al. (ECCV 2016).
// Its Sim(3) step uses weighted Horn alignment, following the scale extension
// described in TEASER, Sec. XI-B, footnote 8. This is a reimplementation,
// not the unavailable source used for the published TEASER figure.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Eigen/Dense>
#include <chrono>
#include <random>
#include <limits>
#include <cmath>

namespace py = pybind11;
using Points = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
using Clock = std::chrono::steady_clock;
struct Fit { double s=1; Eigen::Matrix3d R=Eigen::Matrix3d::Identity(); Eigen::Vector3d t=Eigen::Vector3d::Zero(); };

Fit alignment(const Points& a, const Points& b, const Eigen::VectorXd& w, bool horn) {
    const double sum=w.sum();
    if (!(sum>1e-15)) throw std::runtime_error("No weighted support");
    const Eigen::Vector3d ca=a.transpose()*w/sum, cb=b.transpose()*w/sum;
    Eigen::Matrix3d cov=Eigen::Matrix3d::Zero(); double va=0,vb=0;
    for (int i=0;i<a.rows();++i) {
        const Eigen::Vector3d x=a.row(i).transpose()-ca, y=b.row(i).transpose()-cb;
        cov.noalias()+=w[i]*y*x.transpose(); va+=w[i]*x.squaredNorm(); vb+=w[i]*y.squaredNorm();
    }
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov,Eigen::ComputeFullU|Eigen::ComputeFullV);
    if (!(va>1e-15) || svd.singularValues()[1] <= 1e-12*svd.singularValues()[0])
        throw std::runtime_error("Degenerate weighted point set");
    Eigen::Vector3d signs=Eigen::Vector3d::Ones();
    signs[2]=(svd.matrixU()*svd.matrixV().transpose()).determinant()<0?-1.:1.;
    Fit f; f.R=svd.matrixU()*signs.asDiagonal()*svd.matrixV().transpose();
    f.s=horn?std::sqrt(vb/va):svd.singularValues().dot(signs)/va;
    f.t=cb-f.s*f.R*ca;
    if (!(f.s>0) || !std::isfinite(f.s) || !f.t.allFinite()) throw std::runtime_error("Invalid similarity");
    return f;
}

std::pair<int,double> score(const Points& a,const Points& b,const Fit& f,double threshold) {
    int count=0; double loss=0, bound=threshold*threshold;
    for(int i=0;i<a.rows();++i) {
        double e=(f.s*f.R*a.row(i).transpose()+f.t-b.row(i).transpose()).squaredNorm();
        count+=e<=bound; loss+=std::min(e,bound);
    }
    return {count,loss};
}

py::dict output(const Fit& f,int iterations,const std::string& stop,int inliers) {
    py::array_t<double> matrix({4,4}); auto x=matrix.mutable_unchecked<2>();
    for(int i=0;i<4;++i) for(int j=0;j<4;++j) x(i,j)=i==j?1.:0.;
    for(int i=0;i<3;++i) { for(int j=0;j<3;++j) x(i,j)=f.s*f.R(i,j); x(i,3)=f.t[i]; }
    py::dict result; result["transformation"]=matrix; result["iterations"]=iterations;
    result["stop_reason"]=stop; result["num_inliers"]=inliers; return result;
}

py::dict ransac(const Points& a,const Points& b,double threshold,uint64_t seed,int cap,double timeout) {
    const auto start=Clock::now(); const int n=a.rows();
    std::mt19937_64 generator(seed); std::uniform_int_distribution<int> index(0,n-1);
    Fit best; int best_count=-1, iterations=0; double best_loss=std::numeric_limits<double>::infinity();
    double required=std::numeric_limits<double>::infinity(); std::string stop="iteration_limit";
    while(iterations<cap) {
        if(std::chrono::duration<double>(Clock::now()-start).count()>=timeout) {stop="time_limit";break;}
        if(iterations>=required) {stop="confidence";break;}
        int i=index(generator),j=index(generator),k=index(generator);
        while(j==i) j=index(generator);
        while(k==i||k==j) k=index(generator);
        Points x(3,3),y(3,3); x<<a.row(i),a.row(j),a.row(k); y<<b.row(i),b.row(j),b.row(k);
        ++iterations;
        try {
            Fit f=alignment(x,y,Eigen::VectorXd::Ones(3),false);
            auto [count,loss]=score(a,b,f,threshold);
            if(count>best_count || (count==best_count&&loss<best_loss)) {
                best=f; best_count=count; best_loss=loss;
                // Three DISTINCT indices per sample; confidence is based on
                // estimated consensus, never on the known synthetic outlier rate.
                const double p=count>=3?double(count)*(count-1)*(count-2)/(double(n)*(n-1)*(n-2)):0.;
                required=p>=1?1:p>0?std::ceil(std::log(.01)/std::log1p(-p)):std::numeric_limits<double>::infinity();
            }
        } catch(const std::runtime_error&) { /* Rejected degenerate hypothesis counts as an iteration. */ }
    }
    if(best_count<3) throw std::runtime_error("No nondegenerate RANSAC consensus");
    // Standard final consensus refit, outside the hypothesis budget but included
    // in API timing. No iterative local optimization or oracle inlier labels.
    Eigen::VectorXd weights(n);
    for(int i=0;i<n;++i) weights[i]=(best.s*best.R*a.row(i).transpose()+best.t-b.row(i).transpose()).norm()<=threshold?1.:0.;
    Fit refined=alignment(a,b,weights,false);
    auto [count,loss]=score(a,b,refined,threshold);
    if(count>best_count || (count==best_count&&loss<best_loss)) {best=refined;best_count=count;}
    return output(best,iterations,stop,best_count);
}

py::dict fgr(const Points& a,const Points& b,double threshold) {
    // Shared normalization preserves unknown relative scale. All supplied
    // correspondences enter the robust objective; no rigid tuple filtering.
    const Eigen::RowVector3d ca=a.colwise().mean(),cb=b.colwise().mean();
    Points x=a.rowwise()-ca,y=b.rowwise()-cb;
    double radius=std::max(x.rowwise().norm().maxCoeff(),y.rowwise().norm().maxCoeff());
    if(!(radius>1e-12)) throw std::runtime_error("Degenerate normalization");
    x/=radius; y/=radius;
    const double lower=std::pow(threshold/radius,2);
    double mu=1.; Fit model; Eigen::VectorXd weights(a.rows());
    // 64 continuation stages, factor 1.4. Each stage updates line-process
    // weights and then a closed-form weighted similarity (Horn scale).
    for(int iteration=0;iteration<64;++iteration) {
        for(int i=0;i<a.rows();++i) {
            double residual=(model.s*model.R*x.row(i).transpose()+model.t-y.row(i).transpose()).squaredNorm();
            double line=mu/(mu+residual); weights[i]=line*line;
        }
        model=alignment(x,y,weights,true);
        mu=std::max(lower,mu/1.4);
    }
    model.t=radius*model.t+cb.transpose()-model.s*model.R*ca.transpose();
    return output(model,64,"fixed_iterations",score(a,b,model,threshold).first);
}

PYBIND11_MODULE(sim3_comparison,m) {
    m.def("solve",[](py::array_t<double,py::array::c_style|py::array::forcecast> a,
                     py::array_t<double,py::array::c_style|py::array::forcecast> b,
                     const std::string& method,double threshold,uint64_t seed,double timeout) {
        if(a.ndim()!=2||b.ndim()!=2||a.shape(1)!=3||b.shape(1)!=3||a.shape(0)!=b.shape(0)||a.shape(0)<3)
            throw std::invalid_argument("Expected matching N x 3 inputs, N >= 3");
        Eigen::Map<const Points> x(a.data(),a.shape(0),3), y(b.data(),b.shape(0),3);
        if(!x.allFinite()||!y.allFinite()||!(threshold>0)||!(timeout>0)) throw std::invalid_argument("Invalid input/options");
        if(method=="fgr_sim3") return fgr(x,y,threshold);
        if(method=="ransac_1k"||method=="ransac_1min")
            return ransac(x,y,threshold,seed,method=="ransac_1k"?1000:std::numeric_limits<int>::max(),timeout);
        throw std::invalid_argument("Unknown comparison solver");
    },py::arg("src"),py::arg("dst"),py::arg("method"),py::arg("threshold"),py::arg("seed")=1,py::arg("timeout")=60.);
}
