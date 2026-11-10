#include <turboreg/detail/sim3_cuda.hpp>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>

namespace turboreg::detail {
namespace {
__device__ float median(float* values, int count) {
    for (int i=1;i<count;++i) {
        float v=values[i]; int j=i;
        while (j>0 && values[j-1]>v) { values[j]=values[j-1];--j; }
        values[j]=v;
    }
    return count%2 ? values[count/2] : 0.5f*(values[count/2]+values[count/2-1]);
}

__global__ void local_kernel(const float* ds,const float* dd,const int64_t* si,
                             const int64_t* di,float* output,int64_t n,int k,int mode,
                             float min_dist,float smin,float smax,float log_min,float width) {
    int64_t row=blockIdx.x*blockDim.x+threadIdx.x;
    if (row>=n) return;
    float values[256]; int ids[256]; int count=0;
    if (mode==0) {
        float target[128]; int target_count=0;
        for (int j=0;j<k;++j) {
            float a=ds[row*n+si[row*k+j]],b=dd[row*n+di[row*k+j]];
            if (a>min_dist) values[count++]=a;
            if (b>min_dist) target[target_count++]=b;
        }
        output[row]=(count && target_count) ? median(target,target_count)/(median(values,count)+1e-8f) : 1.f;
        return;
    }
    for (int j=0;j<2*k;++j) {
        int nbr=j<k ? si[row*k+j] : di[row*k+j-k];
        float a=ds[row*n+nbr],b=dd[row*n+nbr];
        if (!(a>min_dist && b>min_dist)) continue;
        float s=b/(a+1e-8f);
        if (!(s>smin && s<smax)) continue;
        bool seen=false;
        for (int t=0;t<count;++t) if (ids[t]==nbr) {seen=true;break;}
        if (!seen) {ids[count]=nbr;values[count++]=s;}
    }
    if (!count) {output[row]=1.f;return;}
    if (mode==2) {output[row]=median(values,count);return;}
    int bins[256];
    for (int j=0;j<count;++j) {
        values[j]=logf(values[j]);
        bins[j]=static_cast<int>(floorf((values[j]-log_min)/width));
    }
    int best_count=-1,best_bin=2147483647; double best_sum=0;
    for (int j=0;j<count;++j) {
        int votes=0;double sum=0;
        for (int t=0;t<count;++t) if (bins[t]==bins[j]) {++votes;sum+=static_cast<double>(values[t]);}
        if (votes>best_count || (votes==best_count && bins[j]<best_bin)) {
            best_count=votes;best_bin=bins[j];best_sum=sum;
        }
    }
    output[row]=static_cast<float>(exp(best_sum/best_count));
}

__global__ void graph_kernel(const float* ds,const float* dd,const float* scales,
                             float* output,int64_t n,float smin,float smax,
                             float epsilon,float radius,bool tsc,bool diff,float threshold) {
    int64_t i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=n*n) return;
    float a=ds[i],b=dd[i];
    bool valid=a>1e-8f && !(a+b<=radius);
    if (diff || tsc) {
        float edge=b/(a+1e-8f),s1=scales[i/n],s2=scales[i%n];
        valid=valid && edge>smin && edge<smax && s1>smin && s1<smax && s2>smin && s2<smax;
        if (diff) valid=valid && fabsf(a-b)<threshold;
        else {
            float x=logf(s1+1e-8f),y=logf(s2+1e-8f),z=logf(edge+1e-8f);
            valid=valid && fmaxf(fmaxf(x,y),z)-fminf(fminf(x,y),z)<epsilon;
        }
    }
    output[i]=valid?1.f:0.f;
}

// Small fixed-size statistics: avoid dozens of elementwise/reduction launches.
// Keep the SVD in LibTorch/cuSOLVER, including its convergence checks.
__global__ void covariance_kernel(const float* a,const float* b,const float* weights,
                                  float* h,float* stats,int batch,int n,float threshold) {
    int model=blockIdx.x*blockDim.x+threadIdx.x;
    if(model>=batch)return;
    a+=model*n*3;b+=model*n*3;
    if(weights)weights+=model*n;
    float ca[3]={0,0,0},cb[3]={0,0,0};float sum=0;
    for(int i=0;i<n;++i) {
        float w=weights?weights[i]:1.f;if(w<threshold)w=0;
        sum+=w;
        for(int j=0;j<3;++j){ca[j]+=a[3*i+j]*w;cb[j]+=b[3*i+j]*w;}
    }
    sum+=1e-8f;
    for(int j=0;j<3;++j){ca[j]/=sum;cb[j]/=sum;}
    float cov[9]={0,0,0,0,0,0,0,0,0},var=0;
    for(int i=0;i<n;++i) {
        float w=weights?weights[i]:1.f;if(w<threshold)w=0;
        float x[3],y[3],norm=0;
        for(int j=0;j<3;++j){x[j]=a[3*i+j]-ca[j];y[j]=b[3*i+j]-cb[j];norm+=x[j]*x[j];}
        var+=norm*w;
        for(int j=0;j<3;++j)for(int k=0;k<3;++k)cov[3*j+k]+=(x[j]*w)*y[k];
    }
    for(int j=0;j<9;++j)h[model*9+j]=cov[j]/sum;
    for(int j=0;j<3;++j){stats[model*7+j]=ca[j];stats[model*7+3+j]=cb[j];}
    stats[model*7+6]=var/sum;
}

__device__ float det3(const float* x) {
    return x[0]*(x[4]*x[8]-x[5]*x[7])-x[1]*(x[3]*x[8]-x[5]*x[6])+x[2]*(x[3]*x[7]-x[4]*x[6]);
}

__global__ void compose_kernel(const float* up,const float* sp,const float* vp,const float* stats,
                               float* out,int batch,int64_t u0,int64_t u1,int64_t u2,
                               int64_t v0,int64_t v1,int64_t v2,int64_t s0,int64_t s1) {
    int m=blockIdx.x*blockDim.x+threadIdx.x;if(m>=batch)return;
    float u[9],v[9],r[9];
    for(int j=0;j<3;++j)for(int k=0;k<3;++k){u[3*j+k]=up[m*u0+j*u1+k*u2];v[3*j+k]=vp[m*v0+k*v1+j*v2];}
    for(int j=0;j<3;++j)for(int k=0;k<3;++k){float sum=0;for(int z=0;z<3;++z)sum+=v[3*j+z]*u[3*k+z];r[3*j+k]=sum;}
    float det=det3(r);float sign=(det>0)?1.f:((det<0)?-1.f:0.f);
    float scale=(sp[m*s0]+sp[m*s0+s1]+sign*sp[m*s0+2*s1])/fmaxf(stats[m*7+6],1e-12f);
    for(int j=0;j<3;++j)for(int k=0;k<3;++k){float sum=0;for(int z=0;z<3;++z)sum+=(v[3*j+z]*(z==2?sign:1.f))*u[3*k+z];r[3*j+k]=sum;out[m*16+j*4+k]=scale*sum;}
    for(int j=0;j<3;++j){float sum=0;for(int k=0;k<3;++k)sum+=r[3*j+k]*stats[m*7+k];out[m*16+j*4+3]=stats[m*7+3+j]-scale*sum;}
    out[m*16+12]=out[m*16+13]=out[m*16+14]=0;out[m*16+15]=1;
}
}

at::Tensor local_scales_cuda(const at::Tensor& ds,const at::Tensor& dd,
                            const at::Tensor& src_ids,const at::Tensor& dst_ids,
                            int mode,float min_distance,float scale_min,float scale_max,float bin_width) {
    const c10::cuda::CUDAGuard guard(ds.device());
    auto result=at::empty({ds.size(0)},ds.options());
    float smin=std::max(scale_min,1e-6f),smax=std::max(scale_max,smin*1.01f);
    local_kernel<<<(ds.size(0)+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(
        ds.data_ptr<float>(),dd.data_ptr<float>(),src_ids.data_ptr<int64_t>(),dst_ids.data_ptr<int64_t>(),
        result.data_ptr<float>(),ds.size(0),src_ids.size(1),mode,min_distance,smin,smax,std::log(smin),std::max(bin_width,1e-3f));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}

at::Tensor compatibility_cuda(const at::Tensor& ds,const at::Tensor& dd,const at::Tensor& scales,
                             float scale_min,float scale_max,float epsilon,float radius,
                             bool tsc,bool distance_diff,float distance_threshold) {
    const c10::cuda::CUDAGuard guard(ds.device());
    auto result=at::empty_like(ds);
    graph_kernel<<<(ds.numel()+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(
        ds.data_ptr<float>(),dd.data_ptr<float>(),scales.data_ptr<float>(),result.data_ptr<float>(),
        ds.size(0),scale_min,scale_max,epsilon,radius,tsc,distance_diff,distance_threshold);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}

at::Tensor umeyama_cuda(const at::Tensor& a,const at::Tensor& b,const at::Tensor& weights,float threshold) {
    const c10::cuda::CUDAGuard guard(a.device());
    auto ac=a.contiguous(),bc=b.contiguous();
    at::Tensor w;
    if(weights.defined() && weights.numel()) {
        w=weights.dim()==1?weights.unsqueeze(0).expand({a.size(0),a.size(1)}).contiguous():weights.contiguous();
        TORCH_CHECK(w.scalar_type()==at::kFloat && w.device()==a.device() && w.dim()==2 &&
                    w.size(0)==a.size(0) && w.size(1)==a.size(1),"Invalid Umeyama weights.");
    }
    int batch=a.size(0),n=a.size(1);
    auto h=at::empty({batch,3,3},a.options()),stats=at::empty({batch,7},a.options());
    covariance_kernel<<<(batch+63)/64,64,0,at::cuda::getCurrentCUDAStream()>>>(
        ac.data_ptr<float>(),bc.data_ptr<float>(),w.defined()?w.data_ptr<float>():nullptr,
        h.data_ptr<float>(),stats.data_ptr<float>(),batch,n,threshold);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    auto [u,s,vh]=at::linalg_svd(h,false);
    auto out=at::empty({batch,4,4},a.options());
    compose_kernel<<<(batch+63)/64,64,0,at::cuda::getCurrentCUDAStream()>>>(
        u.data_ptr<float>(),s.data_ptr<float>(),vh.data_ptr<float>(),stats.data_ptr<float>(),out.data_ptr<float>(),
        batch,u.stride(0),u.stride(1),u.stride(2),vh.stride(0),vh.stride(1),vh.stride(2),s.stride(0),s.stride(1));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
}
