/* Optional native acquisition adapter, not a test-time oracle.
 * Extracted GPL-3.0 mVMC functions retain their upstream license headers.
 * Driver glue is original; original complex initializer/kernel are unmodified.
 */
#include <assert.h>
#include <complex.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <setjmp.h>
#include <math.h>
#define gen_rand32 native_gen_rand32
#define init_gen_rand native_init_gen_rand
#include "SFMT.c"
#undef gen_rand32
#undef init_gen_rand
static uint64_t draws;
extern const char *openblas_get_config(void);
extern int openblas_get_num_threads(void);
uint32_t gen_rand32(void) { ++draws; return native_gen_rand32(); }
int Nsite=128,Nsite2=256,Nsize=2,Ne=1,TwoSz=-1,LapackLWork=4;
int NProj=0,NGutzwillerIdx=0,NJastrowIdx=0,NDoublonHolon2siteIdx=0,NDoublonHolon4siteIdx=0;
int *GutzwillerIdx,**JastrowIdx,**DoublonHolon2siteIdx,**DoublonHolon4siteIdx,*LocSpn;
double complex *SlaterElm,*InvM,*PfM;
static int *iw;
static double *dw;
static double complex *cw;
static int ccursor;
void RequestWorkSpaceThreadInt(int n) { iw=malloc(n*sizeof(*iw)); assert(iw); }
int *GetWorkSpaceThreadInt(int n) { (void)n; return iw; }
void ReleaseWorkSpaceThreadInt(void) { free(iw); }
void RequestWorkSpaceThreadDouble(int n) { dw=malloc(n*sizeof(*dw)); assert(dw); }
double *GetWorkSpaceThreadDouble(int n) { (void)n; return dw; }
void ReleaseWorkSpaceThreadDouble(void) { free(dw); }
void RequestWorkSpaceThreadComplex(int n) { cw=malloc(n*sizeof(*cw)); assert(cw); ccursor=0; }
double complex *GetWorkSpaceThreadComplex(int n) { double complex*p=cw+ccursor; ccursor+=n; return p; }
void ReleaseWorkSpaceThreadComplex(void) { free(cw); }
extern void zsktrf_(const char*,const char*,const int*,double complex*,const int*,int*,double complex*,const int*,int*);
extern void zscal_(const int*,const double complex*,double complex*,const int*);
extern void utu2pfa_z(int,double complex*,int,int*,double complex*);
extern void utu2inv_z(int,double complex*,int,int*,double complex*,double complex*,int);
#define M_ZSKTRF zsktrf_
#define M_ZSCAL zscal_
int calculateMAll_child_fsz(const int*,const int*,int,int,int,double complex*,int*,double complex*,int,double*);
#include "matrix.inc"
#include "projection.inc"
static int attempts,last_status,aborted;
static int sites[2],spins[2],cfg[256],num[256];
static jmp_buf abort_target;
static int observed_calculate(const int*a,const int*b,int first,int end) {
    ++attempts;
    last_status=CalculateMAll_fsz(a,b,first,end);
    printf("kernel %d %d\n",attempts,last_status);
    return last_status;
}
typedef int MPI_Comm;
#define MPI_COMM_WORLD 0
#define MPI_INT 0
#define MPI_MAX 0
int MPI_Comm_size(MPI_Comm c,int*v) { (void)c; *v=1; return 0; }
int MPI_Comm_rank(MPI_Comm c,int*v) { (void)c; *v=0; return 0; }
int MPI_Allreduce(const int*a,int*b,int n,int t,int op,MPI_Comm c) { (void)n;(void)t;(void)op;(void)c; *b=*a; return 0; }
int observed_abort(MPI_Comm c,int status) { (void)c; aborted=status; longjmp(abort_target,1); }
#define MPI_Abort observed_abort
#define CalculateMAll_fsz observed_calculate
#include "initializer.inc"
#undef CalculateMAll_fsz
static void integers(const int*a,int n) { for(int i=0;i<n;i++)printf("%s%d",i?" ":"",a[i]); puts(""); }
int main(void) {
    fprintf(stderr,"BLAS=%s; actual_threads=%d; serial adapter\n",openblas_get_config(),openblas_get_num_threads());
    LocSpn=calloc(128,sizeof(int)); SlaterElm=calloc(65536,sizeof(*SlaterElm));
    InvM=calloc(4,sizeof(*InvM)); PfM=calloc(1,sizeof(*PfM));
    assert(LocSpn&&SlaterElm&&InvM&&PfM);
    SlaterElm[83*256+220]=1; SlaterElm[220*256+83]=-1;
    native_init_gen_rand(1);
    if(setjmp(abort_target)==0) {
        int projection=777;
        int result=makeInitialSample_fsz(sites,cfg,num,&projection,spins,0,1,0);
        printf("unexpected_return %d\n",result);
        return 2;
    }
    assert(attempts==101 && last_status==0 && aborted==1 && draws==202 && idx==202);
    assert(creal(PfM[0])==1 && cimag(PfM[0])==0);
    puts("fixture_begin");
    printf("%d %d %d %d %llu\n",attempts,last_status,aborted,idx,(unsigned long long)draws);
    integers(sites,2); integers(cfg,256); integers(num,256); integers(spins,2);
    for(int i=0;i<624;i++)printf("%s%u",i?" ":"",psfmt32[i]); puts("");
    for(int i=0;i<624;i++)printf("%s%u",i?" ":"",native_gen_rand32()); puts("");
    return 0;
}
