/* Original optional driver. children.inc contains verbatim GPL-3.0-or-later
 * mVMC matrix.c children; unchanged native MPL-2.0 utu2 routines are linked.
 * No Cargo dependency. Synthetic fixed operands, not a full VMC/MPI runner. */
#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
extern const char *openblas_get_config(void);
int Nsite=2, Nsite2=4, Nsize=4;
double real_slater[64], real_inverse[32], real_pf[2];
double complex complex_slater[64], complex_inverse[32], complex_pf[2];
double *SlaterElm_real=real_slater,*InvM_real=real_inverse,*PfM_real=real_pf;
double complex *SlaterElm=complex_slater,*InvM=complex_inverse,*PfM=complex_pf;
static int factor_info, inverse_calls;
static double complex computed_pf;
extern void dsktrf_(const char*,const char*,const int*,double*,const int*,int*,double*,const int*,int*);
extern void zsktrf_(const char*,const char*,const int*,double complex*,const int*,int*,double complex*,const int*,int*);
extern void dscal_(const int*,const double*,double*,const int*);
extern void zscal_(const int*,const double complex*,double complex*,const int*);
extern void utu2pfa_d(int,double*,int,int*,double*);
extern void utu2pfa_z(int,double complex*,int,int*,double complex*);
extern void utu2inv_d(int,double*,int,int*,double*,double*,int);
extern void utu2inv_z(int,double complex*,int,int*,double complex*,double complex*,int);
static void observed_pfa_d(int n,double*a,int lda,int*p,double*pf) {
  utu2pfa_d(n,a,lda,p,pf); computed_pf=*pf;
}
static void observed_pfa_z(int n,double complex*a,int lda,int*p,double complex*pf) {
  utu2pfa_z(n,a,lda,p,pf); computed_pf=*pf;
}
#define M_DSKTRF(...) do { dsktrf_(__VA_ARGS__); factor_info=info; } while(0)
#define M_ZSKTRF(...) do { zsktrf_(__VA_ARGS__); factor_info=info; } while(0)
#define M_DSCAL dscal_
#define M_ZSCAL zscal_
static void observed_inv_d(int n,double*a,int lda,int*p,double*v,double*m,int ldm) {
  ++inverse_calls; utu2inv_d(n,a,lda,p,v,m,ldm);
}
static void observed_inv_z(int n,double complex*a,int lda,int*p,double complex*v,double complex*m,int ldm) {
  ++inverse_calls; utu2inv_z(n,a,lda,p,v,m,ldm);
}
#define utu2inv_d observed_inv_d
#define utu2inv_z observed_inv_z
#define utu2pfa_d observed_pfa_d
#define utu2pfa_z observed_pfa_z
#include "children.inc"
#undef utu2inv_d
#undef utu2inv_z
#undef utu2pfa_d
#undef utu2pfa_z
int main(void) {
  fprintf(stderr,"BLAS=%s; threads requested via OPENBLAS_NUM_THREADS=1\n",openblas_get_config());
  const char *cases[]={"regular","zero","infinite","nan","offset_regular","offset_infinite","complex_regular","sum_overflow"};
  const int idx[4]={0,1,0,1}, spins[4]={0,0,1,1};
  for(int c=0;c<8;++c) for(int mode=0;mode<2;++mode) {
    memset(real_slater,0,sizeof real_slater); memset(complex_slater,0,sizeof complex_slater);
    for(int i=0;i<32;++i) {real_inverse[i]=23; complex_inverse[i]=37+11*I;}
    for(int i=0;i<2;++i) {real_pf[i]=23; complex_pf[i]=37+11*I;}
    int start=(c==4 || c==5)?1:0, local=(c==4 || c==5)?1:0;
    int offset=(start+local)*16;
    double a=c==1?0:c==2 || c==5?INFINITY:c==3?NAN:c==7?1e308:0.5;
    double complex za=c==6?0.5+0.25*I:c==7?a+a*I:a;
    real_slater[offset+1]=a; real_slater[offset+4]=-a;
    real_slater[offset+11]=c==1?0:1.25; real_slater[offset+14]=-real_slater[offset+11];
    complex_slater[offset+1]=za; complex_slater[offset+4]=-za;
    complex_slater[offset+11]=real_slater[offset+11]; complex_slater[offset+14]=-complex_slater[offset+11];
    if(c!=1) {
      const int rows[4]={0,0,1,1}, cols[4]={2,3,2,3};
      const double values[4]={0.125,0.25,0.375,0.625};
      for(int k=0;k<4;++k) {
        int upper=offset+rows[k]*4+cols[k], lower=offset+cols[k]*4+rows[k];
        real_slater[upper]=values[k]; real_slater[lower]=-values[k];
        complex_slater[upper]=values[k]; complex_slater[lower]=-values[k];
      }
    }
    double buf[16]={0},work[16]={0},rwork[16]={0};
    double complex zbuf[16]={0},zwork[16]={0}; int piv[4]={0};
    factor_info=-999; inverse_calls=0; computed_pf=NAN;
    int status=mode?calculateMAll_child_fsz(idx,spins,start,start+2,local,zbuf,piv,zwork,16,rwork):
      calculateMAll_child_fsz_real(idx,spins,start,start+2,local,buf,piv,work,16);
    if(c==7) fprintf(stderr,"sum_overflow %s computedPF=(%.17g,%.17g) componentsFinite=%d sumFinite=%d\n",mode?"complex":"real",creal(computed_pf),cimag(computed_pf),isfinite(creal(computed_pf))&&isfinite(cimag(computed_pf)),isfinite(creal(computed_pf)+cimag(computed_pf)));
    double complex pf=mode?complex_pf[local]:real_pf[local];
    printf("%s %s %d %d %d %d %d %.17g %.17g",mode?"complex":"real",cases[c],start,local,status,factor_info,inverse_calls,creal(pf),cimag(pf));
    for(int i=0;i<16;++i) {
      double complex z=mode?complex_inverse[local*16+i]:real_inverse[local*16+i];
      printf(" %.17g %.17g",creal(z),cimag(z));
    }
    putchar('\n');
  }
}
