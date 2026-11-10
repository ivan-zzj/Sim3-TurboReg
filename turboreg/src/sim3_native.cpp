#include <turboreg/sim3_native.hpp>
#include <Eigen/SVD>
#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>

namespace turboreg::native {
namespace {
using V3=Eigen::Vector3d;
using M3=Eigen::Matrix3d;
struct Transform { M3 sr; V3 t; };
void check(bool condition,const char* message) { if(!condition) throw std::invalid_argument(message); }
void validate(const Options& o) {
    check(o.max_points>=3 && o.num_pivots>0 && o.k_neighbors>0 && o.graph_iterations>0 && o.refinement_iterations>=0,"Invalid integer options");
    for(double v:{o.epsilon_scale,o.inlier_threshold,o.min_pair_distance,o.scale_min,o.scale_max,o.log_scale_bin_width})
        check(std::isfinite(v)&&v>0,"Options must be finite and positive");
    check(std::isfinite(o.nms_radius)&&o.nms_radius>=0 && o.scale_max>o.scale_min,"Invalid distance or scale range");
    check(o.metric=="IN"||o.metric=="MAE"||o.metric=="MSE","Unknown scoring metric");
    check(o.local_scale_mode=="hist"||o.local_scale_mode=="median"||o.local_scale_mode=="union_median","Unknown local scale mode");
    check((std::log(o.scale_max)-std::log(o.scale_min))/o.log_scale_bin_width<1000000,"Histogram too large");
}
double median(std::vector<double> values) {
    if(values.empty()) return 1.;
    const auto mid=values.begin()+values.size()/2;
    std::nth_element(values.begin(),mid,values.end());
    return values.size()%2 ? *mid : .5*(*mid+*std::max_element(values.begin(),mid));
}
bool fit(const Points& a,const Points& b,const std::vector<int>& ids,const std::vector<double>* weights,Transform& out) {
    if(ids.size()<3) return false;
    double sum=0; V3 ca=V3::Zero(),cb=V3::Zero();
    for(size_t k=0;k<ids.size();++k) {
        const double w=weights?(*weights)[k]:1.; sum+=w;
        ca+=w*a.row(ids[k]).transpose(); cb+=w*b.row(ids[k]).transpose();
    }
    if(sum<=0) return false;
    ca/=sum;cb/=sum;
    M3 H=M3::Zero(); double variance=0;
    for(size_t k=0;k<ids.size();++k) {
        const double w=weights?(*weights)[k]:1.;
        const V3 x=a.row(ids[k]).transpose()-ca,y=b.row(ids[k]).transpose()-cb;
        H.noalias()+=w*x*y.transpose(); variance+=w*x.squaredNorm();
    }
    if(variance<=1e-20) return false;
    Eigen::JacobiSVD<M3> svd(H,Eigen::ComputeFullU|Eigen::ComputeFullV);
    const V3 singular=svd.singularValues();
    // A line cannot define a unique rotation; fail explicitly even without area filtering.
    if(singular[1]<=std::max(1e-20,singular[0]*1e-14)) return false;
    V3 sign=V3::Ones();
    if((svd.matrixV()*svd.matrixU().transpose()).determinant()<0) sign[2]=-1.;
    const M3 rotation=svd.matrixV()*sign.asDiagonal()*svd.matrixU().transpose();
    const double scale=singular.dot(sign)/variance;
    if(!(scale>0) || !std::isfinite(scale)) return false;
    out.sr=scale*rotation; out.t=cb-out.sr*ca;
    return out.sr.allFinite()&&out.t.allFinite();
}
bool area_valid(const Points& p,const std::array<int,3>& ids) {
    const V3 a=p.row(ids[1])-p.row(ids[0]),b=p.row(ids[2])-p.row(ids[0]),c=p.row(ids[2])-p.row(ids[1]);
    return a.cross(b).norm()>1e-4*std::max({a.squaredNorm(),b.squaredNorm(),c.squaredNorm()});
}
double residual(const Transform& T,const Points& a,const Points& b,int i) {
    return (T.sr*a.row(i).transpose()+T.t-b.row(i).transpose()).norm();
}
Transform graph_solve(const Points& a,const Points& b,const Options& o) {
    const int n=a.rows(),k=std::min(o.k_neighbors,n-1);
    Eigen::MatrixXd da(n,n),db(n,n);
    for(int i=0;i<n;++i) {
        da(i,i)=db(i,i)=0;
        for(int j=i+1;j<n;++j) {
            da(i,j)=da(j,i)=(a.row(i)-a.row(j)).norm();
            db(i,j)=db(j,i)=(b.row(i)-b.row(j)).norm();
        }
    }
    check(da.allFinite()&&db.allFinite(),"Pairwise distances overflow");
    const double lo=std::log(o.scale_min),hi=std::log(o.scale_max),width=std::max(o.log_scale_bin_width,.001);
    const int nb=static_cast<int>(std::floor((hi-lo)/width))+1;
    std::vector<double> local(n),sums(nb);
    std::vector<int> counts(nb),indices(n),src_k(k),dst_k(k),seen(n);
    for(int i=0;i<n;++i) {
        auto nearest=[&](const Eigen::MatrixXd& d,std::vector<int>& neighbors) {
            std::iota(indices.begin(),indices.end(),0);
            const auto cmp=[&](int x,int y) {
                const double dx=x==i?std::numeric_limits<double>::infinity():d(i,x);
                const double dy=y==i?std::numeric_limits<double>::infinity():d(i,y);
                return dx==dy?x<y:dx<dy;
            };
            std::partial_sort(indices.begin(),indices.begin()+k,indices.end(),cmp);
            std::copy_n(indices.begin(),k,neighbors.begin());
        };
        nearest(da,src_k);nearest(db,dst_k);
        if(o.local_scale_mode=="median") {
            std::vector<double> sa,sb;
            for(int j:src_k) if(da(i,j)>o.min_pair_distance) sa.push_back(da(i,j));
            for(int j:dst_k) if(db(i,j)>o.min_pair_distance) sb.push_back(db(i,j));
            local[i]=!sa.empty()&&!sb.empty()?median(sb)/(median(sa)+1e-8):1.;
            continue;
        }
        std::fill(seen.begin(),seen.end(),0);
        std::fill(counts.begin(),counts.end(),0);std::fill(sums.begin(),sums.end(),0.);
        std::vector<double> scales;
        auto accumulate=[&](int j) {
            if(seen[j]) return;
            seen[j]=1;
            if(da(i,j)<=o.min_pair_distance||db(i,j)<=o.min_pair_distance) return;
            const double scale=db(i,j)/(da(i,j)+1e-8);
            if(!std::isfinite(scale)||scale<=o.scale_min||scale>=o.scale_max) return;
            if(o.local_scale_mode=="union_median") {scales.push_back(scale);return;}
            const double log=std::log(scale);
            const int bin=static_cast<int>(std::floor((log-lo)/width));
            if(bin>=0&&bin<nb) {++counts[bin];sums[bin]+=log;}
        };
        for(int j:src_k) accumulate(j);
        for(int j:dst_k) accumulate(j);
        if(o.local_scale_mode=="union_median") local[i]=median(scales);
        else {
            const int best=std::max_element(counts.begin(),counts.end())-counts.begin();
            local[i]=counts[best]?std::exp(sums[best]/counts[best]):1.;
        }
    }
    const double tolerance_ratio=std::exp(o.epsilon_scale);
    Eigen::MatrixXd adjacency=Eigen::MatrixXd::Zero(n,n);
    for(int i=0;i<n;++i) for(int j=i+1;j<n;++j) {
        if(da(i,j)<=1e-8||da(i,j)+db(i,j)<=o.nms_radius) continue;
        if(o.use_tsc) {
            const double edge=db(i,j)/(da(i,j)+1e-8);
            if(edge<=o.scale_min||edge>=o.scale_max||local[i]<=o.scale_min||local[i]>=o.scale_max||local[j]<=o.scale_min||local[j]>=o.scale_max) continue;
            // log(max)-log(min)<epsilon, evaluated without per-edge logarithms.
            const double li=local[i]+1e-8,lj=local[j]+1e-8,le=edge+1e-8;
            if(std::max({li,lj,le})>=std::min({li,lj,le})*tolerance_ratio) continue;
        }
        adjacency(i,j)=adjacency(j,i)=1.;
    }
    Eigen::MatrixXd score=adjacency;
    if(o.use_second_order && n<=128) {
        // For a binary graph, (A*A)[i,j] counts common neighbors exactly.
        std::vector<std::array<uint64_t,2>> bits(n,{0,0});
        for(int i=0;i<n;++i) for(int j=0;j<n;++j)
            if(adjacency(i,j)) bits[i][j/64]|=uint64_t(1)<<(j%64);
        for(int i=0;i<n;++i) for(int j=i+1;j<n;++j) if(adjacency(i,j)) {
            const int common=__builtin_popcountll(bits[i][0]&bits[j][0])+__builtin_popcountll(bits[i][1]&bits[j][1]);
            score(i,j)=score(j,i)=common;
        }
    } else if(o.use_second_order) score=(adjacency*adjacency).cwiseProduct(adjacency);
    struct Edge {int i,j;double score;};
    std::vector<Edge> edges;
    for(int i=0;i<n;++i) for(int j=i+1;j<n;++j) if(score(i,j)>0) edges.push_back({i,j,score(i,j)});
    const auto edge_cmp=[](const Edge& x,const Edge& y) {
        if(x.score!=y.score) return x.score>y.score;
        return std::pair<int,int>(x.i,x.j)<std::pair<int,int>(y.i,y.j);
    };
    const size_t budget=std::min<size_t>(o.num_pivots,edges.size());
    std::partial_sort(edges.begin(),edges.begin()+budget,edges.end(),edge_cmp);
    Transform best;double best_score=-1;
    std::set<std::array<int,3>> visited;
    for(size_t e=0;e<budget;++e) {
        const auto& edge=edges[e];
        std::array<int,2> third={-1,-1};std::array<double,2> sc={-1,-1};
        for(int t=edge.j+1;t<n;++t) {
            if(!adjacency(edge.i,t)||!adjacency(edge.j,t)) continue;
            const double value=edge.score+score(edge.i,t)+score(edge.j,t);
            if(value>sc[0]) {third[1]=third[0];sc[1]=sc[0];third[0]=t;sc[0]=value;}
            else if(value>sc[1]) {third[1]=t;sc[1]=value;}
        }
        for(int t:third) {
            if(t<0) continue;
            const std::array<int,3> tri={edge.i,edge.j,t};
            if(!visited.insert(tri).second) continue;
            if(o.reject_degenerate&&(!area_valid(a,tri)||!area_valid(b,tri))) continue;
            Transform estimate;
            const std::vector<int> ids(tri.begin(),tri.end());
            if(!fit(a,b,ids,nullptr,estimate)) continue;
            double value=0;
            for(int i=0;i<n;++i) {
                const double r=residual(estimate,a,b,i);
                if(o.metric=="IN") value+=r<o.inlier_threshold;
                else {const double w=std::max(0.,1.-r/o.inlier_threshold);value+=o.metric=="MSE"?w*w:w;}
            }
            if(value>best_score) {best=estimate;best_score=value;}
        }
    }
    if(best_score<0) throw std::runtime_error("No nondegenerate Sim3 hypothesis");
    return best;
}
}
Result register_sim3(const Eigen::Ref<const Points>& src,const Eigen::Ref<const Points>& dst,const Options& o) {
    validate(o);
    check(src.rows()>=3&&src.rows()==dst.rows(),"Expected matching N>=3 points");
    check(src.allFinite()&&dst.allFinite(),"Inputs must be finite");
    const int n=std::min<int>(src.rows(),o.max_points);
    const Points a=src.topRows(n),b=dst.topRows(n);
    Points ai=a,bi=b; Transform T;
    for(int iter=0;iter<o.graph_iterations;++iter) {
        try { T=graph_solve(ai,bi,o); }
        catch(const std::runtime_error&) {
            if(iter==0) throw;
            break; // Preserve the previous valid estimate if a later graph has no clique.
        }
        if(iter+1==o.graph_iterations) break;
        std::vector<int> ids;
        for(int i=0;i<ai.rows();++i) if(residual(T,ai,bi,i)<o.inlier_threshold) ids.push_back(i);
        if(ids.size()<10) break;
        Points an(ids.size(),3),bn(ids.size(),3);
        for(size_t i=0;i<ids.size();++i) {an.row(i)=ai.row(ids[i]);bn.row(i)=bi.row(ids[i]);}
        ai=std::move(an);bi=std::move(bn);
    }
    std::vector<int> previous;
    for(int iter=0;iter<o.refinement_iterations;++iter) {
        std::vector<int> ids;std::vector<double> weights;
        for(int i=0;i<n;++i) {
            const double r=residual(T,a,b,i);
            if(r<o.inlier_threshold) {ids.push_back(i);const double q=r/o.inlier_threshold;weights.push_back(o.uniform_refinement?1.:1./(1.+q*q));}
        }
        if(ids.size()<3||ids==previous) break;
        previous=ids;Transform refined;
        if(!fit(a,b,ids,&weights,refined)) break;
        T=refined;
    }
    Result r;r.transformation=Eigen::Matrix4d::Identity();
    r.transformation.topLeftCorner<3,3>()=T.sr;r.transformation.topRightCorner<3,1>()=T.t;
    r.scale=T.sr.col(0).norm();r.num_input=src.rows();r.num_used=n;
    r.inlier_mask.resize(src.rows());
    for(int i=0;i<src.rows();++i) {
        const bool inside=(T.sr*src.row(i).transpose()+T.t-dst.row(i).transpose()).norm()<o.inlier_threshold;
        r.inlier_mask[i]=inside;r.num_inliers+=inside;
    }
    if(r.num_inliers<3) throw std::runtime_error("No Sim3 consensus with at least three correspondences");
    return r;
}
}
