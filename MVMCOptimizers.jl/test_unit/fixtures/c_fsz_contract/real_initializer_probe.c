/* Original optional acquisition driver. See README for native source/license
 * boundaries. No Cargo oracle dependency; single-rank serial adapter only. */
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <setjmp.h>
extern const char *openblas_get_config(void);
extern int openblas_get_num_threads(void);
extern int openblas_get_parallel(void);
#define gen_rand32 native_gen_rand32
#define init_gen_rand native_init_gen_rand
#include "SFMT.c"
#undef gen_rand32
#undef init_gen_rand
static uint64_t draw_count;
uint32_t gen_rand32(void) { ++draw_count; return native_gen_rand32(); }
void init_gen_rand(uint32_t seed) { native_init_gen_rand(seed); draw_count=0; }

int Nsite,Nsite2,Nsize,Ne,TwoSz,LapackLWork;
int NProj=0,NGutzwillerIdx=0,NJastrowIdx=0,NDoublonHolon2siteIdx=0,NDoublonHolon4siteIdx=0;
int *GutzwillerIdx,**JastrowIdx,**DoublonHolon2siteIdx,**DoublonHolon4siteIdx,*LocSpn;
double *SlaterElm_real,*InvM_real,*PfM_real;
static int *integer_work;
static double *double_work;
static size_t double_cursor,double_capacity;
void RequestWorkSpaceThreadInt(int n) { integer_work=malloc((size_t)n*sizeof(int)); assert(integer_work); }
int *GetWorkSpaceThreadInt(int n) { (void)n; return integer_work; }
void ReleaseWorkSpaceThreadInt(void) { free(integer_work); }
void RequestWorkSpaceThreadDouble(int n) { double_capacity=(size_t)n; double_cursor=0; double_work=malloc(double_capacity*sizeof(double)); assert(double_work); }
double *GetWorkSpaceThreadDouble(int n) { assert(double_cursor+(size_t)n<=double_capacity); double *p=double_work+double_cursor; double_cursor+=(size_t)n; return p; }
void ReleaseWorkSpaceThreadDouble(void) { free(double_work); }
extern void dsktrf_(const char*,const char*,const int*,double*,const int*,int*,double*,const int*,int*);
extern void dscal_(const int*,const double*,double*,const int*);
extern void utu2pfa_d(int,double*,int,int*,double*);
extern void utu2inv_d(int,double*,int,int*,double*,double*,int);
static int factor_info, inverse_calls;
#define M_DSKTRF(...) do { dsktrf_(__VA_ARGS__); factor_info=info; } while(0)
#define M_DSCAL dscal_
static void observed_inverse(int n,double*a,int ld,int*p,double*v,double*m,int ldm) { ++inverse_calls; utu2inv_d(n,a,ld,p,v,m,ldm); }
#define utu2inv_d observed_inverse
int calculateMAll_child_fsz_real(const int*,const int*,int,int,int,double*,int*,double*,int);
#include "matrix.inc"
#undef utu2inv_d
#include "projection.inc"

static char label[128];
static int attempts,last_status,aborted;
static int *ele_idx,*ele_cfg,*ele_num,*ele_spn;
static jmp_buf abort_target;
static void integers(const char *name,const int *values,int length) {
  printf("%s",name); for(int i=0;i<length;++i) printf(" %d",values[i]); putchar('\n');
}
static void checkpoint(const char *phase) {
  printf("checkpoint %s %s %d %d %d %d %llu\n",label,phase,attempts,last_status,aborted,idx,(unsigned long long)draw_count);
  integers("tmp_idx",ele_idx,Nsize); integers("tmp_cfg",ele_cfg,Nsite2);
  integers("tmp_num",ele_num,Nsite2); integers("tmp_spn",ele_spn,Nsize);
  puts("tmp_proj");
  printf("raw624"); for(int i=0;i<624;++i) printf(" %u",psfmt32[i]); putchar('\n');
  uint32_t saved[624]; memcpy(saved,psfmt32,sizeof saved); int saved_cursor=idx;
  printf("next624"); for(int i=0;i<624;++i) printf(" %u",native_gen_rand32()); putchar('\n');
  memcpy(psfmt32,saved,sizeof saved); idx=saved_cursor;
  assert(memcmp(psfmt32,saved,sizeof saved)==0 && idx==saved_cursor);
}
static int observed_calculate(const int *sites,const int *spins,int first,int end) {
  ++attempts; factor_info=-999; inverse_calls=0;
  last_status=CalculateMAll_fsz_real(sites,spins,first,end);
  printf("kernel %s %d %d %d %d\n",label,attempts,last_status,factor_info,inverse_calls);
#ifndef INITIALIZER_NO_ATTEMPT_CHECKPOINT
  checkpoint("attempt");
#endif
  return last_status;
}
typedef int MPI_Comm;
#define MPI_COMM_WORLD 0
static int MPI_Comm_size(MPI_Comm c,int *v) { (void)c; *v=1; return 0; }
static int MPI_Comm_rank(MPI_Comm c,int *v) { (void)c; *v=0; return 0; }
#define MPI_INT 0
#define MPI_MAX 0
static int MPI_Allreduce(const int *a,int*b,int n,int type,int op,MPI_Comm c) { (void)n;(void)type;(void)op;(void)c; *b=*a; return 0; }
static int observed_abort(MPI_Comm c,int status) { (void)c; aborted=status; longjmp(abort_target,1); }
#define MPI_Abort observed_abort
#define CalculateMAll_fsz_real observed_calculate
#include "initializer.inc"
#undef CalculateMAll_fsz_real

static void allocate(int ns,int ne,int qp) {
  Nsite=ns; Nsite2=2*ns; Ne=ne; Nsize=2*ne; LapackLWork=Nsize*Nsize;
  LocSpn=calloc((size_t)ns,sizeof(int));
  SlaterElm_real=malloc((size_t)qp*Nsite2*Nsite2*sizeof(double));
  InvM_real=malloc((size_t)qp*Nsize*Nsize*sizeof(double)); PfM_real=malloc((size_t)qp*sizeof(double));
  ele_idx=malloc((size_t)Nsize*sizeof(int)); ele_spn=malloc((size_t)Nsize*sizeof(int));
  ele_cfg=malloc((size_t)Nsite2*sizeof(int)); ele_num=malloc((size_t)Nsite2*sizeof(int));
  assert(LocSpn&&SlaterElm_real&&InvM_real&&PfM_real&&ele_idx&&ele_spn&&ele_cfg&&ele_num);
  for(int i=0;i<qp*Nsize*Nsize;++i) InvM_real[i]=23;
  for(int i=0;i<qp;++i) PfM_real[i]=23;
}
static void release(void) { free(LocSpn);free(SlaterElm_real);free(InvM_real);free(PfM_real);free(ele_idx);free(ele_spn);free(ele_cfg);free(ele_num); }
int main(int argc,char **argv) {
  fprintf(stderr,"BLAS=%s; actual_threads=%d; parallel_backend=%d; OPENBLAS_NUM_THREADS=1 requested; serial single-rank adapter\n",openblas_get_config(),openblas_get_num_threads(),openblas_get_parallel());
  if(argc==2 && strcmp(argv[1],"--boundary-pair")==0) {
    native_init_gen_rand(1); int pairs[101][2];
    for(int i=0;i<101;++i) { pairs[i][0]=(int)(native_gen_rand32()%128); pairs[i][1]=(int)(native_gen_rand32()%128); }
    for(int i=0;i<100;++i) assert(pairs[i][0]!=pairs[100][0] || pairs[i][1]!=pairs[100][1]);
    printf("%d %d\n",pairs[100][0],pairs[100][1]); return 0;
  }
  if(argc!=2) return 2;
  FILE *input=fopen(argv[1],"r"); if(!input) return 3;
  int operation,ns,ne,qp,first,end; unsigned seed;
  while(fscanf(input,"%127s %d %d %d %d %d %d %d %u",label,&operation,&ns,&ne,&qp,&first,&end,&TwoSz,&seed)==9) {
    assert(ns>0 && ns<=128 && ne>0 && 2*ne<=2*ns && qp>0 && qp<=4 && first>=0 && first<=end && end<=qp);
    allocate(ns,ne,qp); attempts=0;last_status=0;aborted=0;factor_info=-999;inverse_calls=0;
    for(int i=0;i<ns;++i) assert(fscanf(input,"%d",&LocSpn[i])==1);
    for(int i=0;i<Nsize;++i) assert(fscanf(input,"%d",&ele_idx[i])==1);
    for(int i=0;i<Nsize;++i) assert(fscanf(input,"%d",&ele_spn[i])==1);
    for(int i=0;i<qp*Nsite2*Nsite2;++i) assert(fscanf(input,"%lf",&SlaterElm_real[i])==1);
    memset(ele_cfg,0,(size_t)Nsite2*sizeof(int));memset(ele_num,0,(size_t)Nsite2*sizeof(int));
    init_gen_rand(seed);
    printf("input %s op=%d nsite=%d nelec=%d qp=%d range=%d:%d twoSz=%d seed=%u\n",label,operation,ns,ne,qp,first,end,TwoSz,seed);
    if(operation==0) {
      last_status=CalculateMAll_fsz_real(ele_idx,ele_spn,first,end);
      printf("kernel %s 0 %d %d %d\n",label,last_status,factor_info,inverse_calls);
    } else if(setjmp(abort_target)==0) {
      int projection_sentinel=777;
      int returned=makeInitialSample_fsz_real(ele_idx,ele_cfg,ele_num,&projection_sentinel,ele_spn,first,end,0);
      printf("returned %s %d\n",label,returned);
      assert(projection_sentinel==777);
    }
    checkpoint("final");
    printf("pf");for(int i=0;i<qp;++i)printf(" %.17g",PfM_real[i]);putchar('\n');
    printf("inverse");for(int i=0;i<qp*Nsize*Nsize;++i)printf(" %.17g",InvM_real[i]);putchar('\n');
    release();
  }
  fclose(input); return 0;
}
