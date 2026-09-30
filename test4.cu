#include<assert.h>
#include<cooperative_groups.h>
#include<cooperative_groups/memcpy_async.h>
#include<cooperative_groups/reduce.h>
#include<cooperative_groups/scan.h>
#include<cstdio>
#include<cuda_fp16.h>
#include<curand.h>
#include<stdint.h>
#include<type_traits>
#define LEN(a)sizeof(a)/sizeof(*a)




template<bool PAK,bool IF,bool TR,class ST,class LD>
__global__ void __maxnreg__(40)sort84_ker(LD*A,unsigned X){
  list<PAK,IF,TR,unsigned,16>c;
  for(unsigned x=0;x<X;x++){
    LD a;
    a=A[x];
    for(unsigned v=0;v<16;v++){
      const uint8_t k=((uint8_t*)&a)[v]&0xf;
      c(k,c[k]+1);
   }
 }
  ST a;
  unsigned x=0,i=0;
  for(uint8_t k=0;k<16;k++){
    const uint16_t k2=(uint16_t)k<<8 | k;
    const unsigned k3=(unsigned)k2<<16 | k2;
    const uint64_t k4=(uint64_t)k3<<32 | k3;
    unsigned v=c[k];
    if(i){
      const unsigned i_=16-i;
      if(i_&1)
       ((uint8_t*)&a)[i]=k;
      if(i_&2)
       ((uint16_t*)&a)[i/2]=k2;
      if(i_&4)
       ((unsigned*)&a)[i/4]=k3;
      if(i==8)
       ((uint64_t*)&a)[1]=k4;
      if(v>=i_){
       ((ST*)A)[x++]=a;
        i=v-i_;
     }else{
        i+=v;
        continue;
     }
   }else
      i=v;
    if constexpr(sizeof(ST::x)==4)
      a=make_int4(k3,k3,k3,k3);
    else
      a=make_long2(k4,k4);
    for(;i>=16;i-=16)
     ((ST*)A)[x++]=a;
 }
}
template<bool PAK,bool IF,bool TR,class ST,class LD>
__global__ void __maxnreg__(40)sort4_ker(LD*A,unsigned X){
  list<PAK,IF,TR,unsigned,16>c;
  for(unsigned x=0;x<X;x++){
    LD a;
    a=A[x];
    for (unsigned v = 0; v < 16; v++) {
      const uint8_t k = ((uint8_t*)&a)[v];
      c(k & 0xf, c[k & 0xf] + 1);
      c(k >> 4, c[k >> 4] + 1);
    }
 }
  ST a;
  unsigned x=0,i=0;
  for(uint8_t k=0;k<16;k++){
    const uint8_t k2=k<<4 | k;
    const uint16_t k4=(uint16_t)k2<<8 | k2;
    const unsigned k8=(unsigned)k4<<16 | k4;
    const uint64_t k16=(uint64_t)k8<<32 | k8;
    unsigned v=c[k];
    if(i){
      const unsigned i_=32-i;
      if(i_&1)
       ((uint8_t*)&a)[i]=((uint8_t*)&a)[i]&0xf0 | k;
      if(i_&2)
       ((uint8_t*)&a)[i/2]=k2;
      if(i_&4)
       ((uint16_t*)&a)[i/4]=k4;
      if(i_&8)
       ((unsigned*)&a)[i/8]=k8;
      if(i==16)
       ((uint64_t*)&a)[1]=k16;
      if(v>=i_){
       ((ST*)A)[x++]=a;
        i=v-i_;
     }else{
        i+=v;
        continue;
     }
   }else
      i=v;
    if constexpr(sizeof(ST::x)==4)
      a=make_int4(k8,k8,k8,k8);
    else
      a=make_long2(k16,k16);
    for(;i>=32;i-=32)
     ((ST*)A)[x++]=a;
 }
}
template<bool PAK,bool IF,bool TR,class ST,class LD>
__global__ void __maxnreg__(40)sort4_ker(LD*A,uint16_t X){
  list<PAK,IF,TR,uint16_t,16>c;
  for(uint16_t x=0;x<X;x++){
    LD a;
    a=A[x];
    for(uint8_t i=0;i<16;i++){
      const uint8_t k=((uint8_t*)&a)[i];
      c(k&0xf,c[k&0xf]+1);
      c(k>>4,c[k>>4]+1);
   }
 }
  ST a;
  uint16_t x=0,i=0;
  for(uint8_t k=0;k<16;k++){
    const uint8_t k2=k<<4 | k;
    const uint16_t k4=(uint16_t)k2<<8 | k2;
    const uint16_t k8=(uint16_t)k4<<16 | k4;
    const uint64_t k16=(uint64_t)k8<<32 | k8;
    uint16_t v=c[k];
    if(i){
      const uint16_t i_=32-i;
      if(i_&1)
       ((uint8_t*)&a)[i]=((uint8_t*)&a)[i]&0xf0 | k;
      if(i_&2)
       ((uint8_t*)&a)[i/2]=k2;
      if(i_&4)
       ((uint16_t*)&a)[i/4]=k4;
      if(i_&8)
       ((uint16_t*)&a)[i/8]=k8;
      if(i==16)
       ((uint64_t*)&a)[1]=k16;
      if(v>=i_){
       ((ST*)A)[x++]=a;
        i=v-i_;
     }else{
        i+=v;
        continue;
     }
   }else
      i=v;
    if constexpr(sizeof(ST::x)==4)
      a=make_int4(k8,k8,k8,k8);
    else
      a=make_long2(k16,k16);
    for(;i>=32;i-=32)
     ((ST*)A)[x++]=a;
 }
}
__global__ void sum_ker(const int4*__restrict__ A,unsigned X,unsigned*__restrict__ O){
  unsigned o=0;
  for(unsigned x=0;x<X;x++){
    int4 a=A[x];
    o+=a.x+a.y+a.z+a.w;
 }
  asm volatile("{.shared .b64 share;.reg .b32 r;.reg .pred p;"
     "redux.sync.add.u32 %0,%0,-1;"
     "mov.b32 r,%laneid;"
     "setp.ne.b32 p,r,0;"
     "@p exit;"
     "mov.b32 r,%tid.x;"
     "setp.ne.b32 p,r,0;"
     "@!p st.shared.b32[share],0;"
     "bar.sync 0;"
     "red.relaxed.cta.shared.add.u32[share],%0;"
     "bar.sync 0;"
     "@p exit;"
     "ld.relaxed.cta.shared.b32 %0,[share];"
     "red.relaxed.gpu.global.add.u32[%1],%0;}"::"r"(o),"l"(O));
}

int main(){
  int4*A;
  cudaMalloc(&A,500000*16);
  unsigned*global;
  cudaMalloc(&global,4);
  cudaMemset(global,0,1);

  sum_ker<<<2048,768>>>(A,500000,global);

  cudaDeviceSynchronize();
  printf(cudaGetErrorString(cudaGetLastError()));
}