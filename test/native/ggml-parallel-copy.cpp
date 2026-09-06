#include "ggml-cpu-impl.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <omp.h>
#include "baseline-copy.h"
#include "candidate-copy.h"

static ggml_tensor tensor(ggml_type type, std::array<int64_t,4> dims, int pad, int layout) {
    ggml_tensor t{}; t.type=type;
    std::copy(dims.begin(), dims.end(), t.ne);
    t.nb[0]=ggml_type_size(type);
    const size_t row=ggml_row_size(type,t.ne[0]);
    t.nb[1]=row+pad; t.nb[2]=t.nb[1]*t.ne[1]+pad; t.nb[3]=t.nb[2]*t.ne[2]+pad;
    if(layout==1) { t.nb[2]=row+pad; t.nb[1]=t.nb[2]*t.ne[2]+pad; t.nb[3]=t.nb[1]*t.ne[1]+pad; }
    if(layout==2) t.nb[1]=0;
    return t;
}

static void run(ggml_tensor *dst, int nth, bool candidate) {
    #pragma omp parallel num_threads(nth)
    {
        ggml_compute_params p{}; p.ith=omp_get_thread_num();p.nth=omp_get_num_threads();
        if(!candidate || !ggml_compute_forward_dup_bytes_parallel(&p,dst)) ggml_compute_forward_dup_bytes(&p,dst);
    }
}

static int correctness() {
    int cases=0;
    for(auto ty:{GGML_TYPE_F32,GGML_TYPE_F16,GGML_TYPE_Q8_0})
    for(int n0:{32,512,32768}) for(int n1:{1,3,9}) for(int n2:{1,5})
    for(int n3:{1,2}) for(int sl:{0,1,2}) for(int dl:{0,1}) {
        auto src=tensor(ty,{n0,n1,n2,n3},36,sl), dst=tensor(ty,{n0,n1,n2,n3},68,dl);
        std::vector<unsigned char> input(ggml_nbytes(&src)+128), want(ggml_nbytes(&dst)+128,0xa5), got(want);
        for(size_t i=0;i<input.size();i++) input[i]=(i*173+(i>>8)*19+11)%251;
        src.data=input.data()+64;dst.src[0]=&src;dst.data=want.data()+64;
        run(&dst,1,false);
        for(int nth:{1,2,3,4,7}) {
            std::fill(got.begin(),got.end(),0xa5);dst.data=got.data()+64;
            run(&dst,nth,true);
            if(got!=want) {fprintf(stderr,"mismatch type=%d dims=%d,%d,%d,%d layout=%d,%d nth=%d\n",ty,n0,n1,n2,n3,sl,dl,nth);return 1;}
            ++cases;
        }
    }
    // Aliasing, overlapping destination rows and small copies must not enter
    // the new path. Check rejection without invoking memcpy on an overlap.
    auto src=tensor(GGML_TYPE_F32,{32768,2,2,1},64,0),dst=src;
    std::vector<unsigned char> data(2*ggml_nbytes(&src)+4096,0xa5);
    src.data=data.data();dst.src[0]=&src;dst.data=data.data()+64;
    ggml_compute_params p{};p.nth=4;p.ith=0;
    if(ggml_compute_forward_dup_bytes_parallel(&p,&dst)) return 2;
    dst.data=data.data()+ggml_nbytes(&src)+1024;dst.nb[1]=0;
    if(ggml_compute_forward_dup_bytes_parallel(&p,&dst)) return 3;
    auto tiny=tensor(GGML_TYPE_F32,{32,1,1,1},0,0);tiny.src[0]=&tiny;tiny.data=data.data();
    if(ggml_compute_forward_dup_bytes_parallel(&p,&tiny)) return 4;
    printf("{\"correctness_cases\":%d,\"alias_and_layout_guards\":true}\n",cases);
    return 0;
}

static double time_copies(ggml_tensor *dst,int n,bool candidate) {
    auto start=std::chrono::steady_clock::now();
    for(int i=0;i<n;i++)run(dst,4,candidate);
    return std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/n;
}

int main(int argc,char**) {
    omp_set_dynamic(0);
    if(argc==1)return correctness();
    for(auto shape:{std::array<int64_t,4>{262144,1,32,1},{4096,1,1024,1},{5120,4,64,1},{1024,32,64,1}}) {
        auto src=tensor(GGML_TYPE_F32,shape,64,0),dst=tensor(GGML_TYPE_F32,shape,128,0);
        std::vector<unsigned char> input(ggml_nbytes(&src),0x3b),output(ggml_nbytes(&dst),0);
        src.data=input.data();dst.data=output.data();dst.src[0]=&src;
        std::vector<double> old,newer;
        time_copies(&dst,20,false);time_copies(&dst,20,true);
        for(int round=0;round<7;round++) {
            if(round%2) {newer.push_back(time_copies(&dst,100,true));old.push_back(time_copies(&dst,100,false));}
            else {old.push_back(time_copies(&dst,100,false));newer.push_back(time_copies(&dst,100,true));}
        }
        std::sort(old.begin(),old.end());std::sort(newer.begin(),newer.end());
        printf("{\"shape\":[%ld,%ld,%ld,%ld],\"bytes\":%zu,\"baseline_us\":%.3f,\"candidate_us\":%.3f,\"speedup\":%.3f}\n",shape[0],shape[1],shape[2],shape[3],ggml_nbytes(&dst),old[3]*1e6,newer[3]*1e6,old[3]/newer[3]);fflush(stdout);
    }
}
