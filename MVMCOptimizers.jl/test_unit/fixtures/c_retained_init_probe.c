/* InitParameter: mVMC 1.3.0 src/mVMC/parameter.c, unchanged definition.
 * Copyright (C) 2016 The University of Tokyo. GPL-3.0-or-later.
 * Upstream: https://github.com/issp-center-dev/mVMC
 * Extraction/source hashes and reproduction: c_retained_init.md.
 */
#include <stdio.h>
#include <stdint.h>
#include <complex.h>
#include <math.h>
extern void init_gen_rand(uint32_t);
extern double genrand_real2(void);
extern uint32_t gen_rand32(void);
int NProj=2, FlagRBM=1, AllComplexFlag=0, NRBM=3, Nneuron=1, NSlater=3, NOptTrans=0;
int OptFlag[16]={0,0,0,0,1,0,0,0,1,0,1,0,0,0,1,0};
double complex Proj[2], RBM[3], Slater[3], OptTrans[1], ParaQPOptTrans[1];
void InitParameter() {
  int i;
  //printf("AllComplexFlag=%d \n",AllComplexFlag);
  #pragma omp parallel for default(shared) private(i)
  for(i=0;i<NProj;i++) Proj[i] = 0.0;

  if (FlagRBM) {
    if(AllComplexFlag==0){
      for(i=0;i<NRBM;i++){
        if(OptFlag[2*i+2*NProj] > 0){ //TBC
          RBM[i]  =   0.01*(genrand_real2() -0.5)/(double)Nneuron; /* uniform distribution [0,1) */
        } else {
          RBM[i] = 0.0;
        }
      }
    }else{
      for(i=0;i<NRBM;i++){
        if(OptFlag[2*i+2*NProj] > 0){ //TBC
          RBM[i]  =   1e-2*genrand_real2()*cexp(2.0*I*M_PI*genrand_real2());
          //RBM[i]  =   1e-2*genrand_real2()*cexp(2.0*I*M_PI*genrand_real2())/(double)Nneuron;
        } else {
          RBM[i] = 0.0;
        }
      }
    }
  }

  if(AllComplexFlag==0){
    for(i=0;i<NSlater;i++){
      if(OptFlag[2*i+2*NProj + 2*FlagRBM*NRBM] > 0){ //TBC
        Slater[i] =  2*(genrand_real2()-0.5); /* uniform distribution [-1,1) */
        //Slater[i] =  1*genrand_real2(); /* uniform distribution [0,1) */
        //Slater[i] += 1*I*genrand_real2(); /* uniform distribution [0,1) */
        //printf("DEBUG: i=%d slater=%lf %lf \n",i,creal(Slater[i]),cimag(Slater[i]));
      } else {
        Slater[i] = 0.0;
      }
    }
  }
  else{
    for(i=0;i<NSlater;i++){
      if(OptFlag[2*i+2*NProj + 2*FlagRBM*NRBM] > 0){ //TBC
        Slater[i] =  2*(genrand_real2()-0.5); /* uniform distribution [-1,1) */
        Slater[i] += 2*I*(genrand_real2()-0.5); /* uniform distribution [-1,1) */
        Slater[i] /=sqrt(2.0);
        //printf("i=%d: %lf %lf \n",i,creal(Slater[i]),cimag(Slater[i]));
      } else {
        Slater[i] = 0.0;
      }
    }
  }

  for(i=0;i<NOptTrans;i++){
    OptTrans[i] = ParaQPOptTrans[i];
  }

  return;
}


int main(void){init_gen_rand(11272); InitParameter(); for(int i=0;i<8;i++){ double complex v=i<2?Proj[i]:i<5?RBM[i-2]:Slater[i-5]; printf("%.17g %.17g\n",creal(v),cimag(v)); } for(int i=0;i<624;i++) printf("%u\n",gen_rand32()); return 0;}
