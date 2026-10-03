/*
mVMC - A numerical solver package for a wide range of quantum lattice models based on many-variable Variational Monte Carlo method
Copyright (C) 2016 The University of Tokyo, All rights reserved.

This program is developed based on the mVMC-mini program
(https://github.com/fiber-miniapp/mVMC-mini)
which follows "The BSD 3-Clause License".

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details. 

You should have received a copy of the GNU General Public License 
along with this program. If not, see http://www.gnu.org/licenses/. 
*/
/* MakeRBMCnt below is verbatim upstream rbm.c:187-288; see README.md. */

#include <complex.h>
#include <stdio.h>
int Nsite=2,Nsite2=4,Nneuron=1,NneuronCharge,NneuronSpin,NneuronGeneral;
int NRBM_PhysLayerIdx,NRBM_HiddenLayerIdx=1;
int NChargeRBM_PhysLayerIdx,NSpinRBM_PhysLayerIdx,NGeneralRBM_PhysLayerIdx;
int NChargeRBM_HiddenLayerIdx,NSpinRBM_HiddenLayerIdx,NGeneralRBM_HiddenLayerIdx;
int NChargeRBM_PhysHiddenIdx,NSpinRBM_PhysHiddenIdx,NGeneralRBM_PhysHiddenIdx;
int layers[4]={0};
int *ChargeRBM_PhysLayerIdx=layers,*SpinRBM_PhysLayerIdx=layers,*GeneralRBM_PhysLayerIdx=layers;
int *ChargeRBM_HiddenLayerIdx=layers,*SpinRBM_HiddenLayerIdx=layers,*GeneralRBM_HiddenLayerIdx=layers;
int indices[4]={0,1,2,3};
int *rows[4]={indices,indices+1,indices+2,indices+3};
int **ChargeRBM_PhysHiddenIdx=rows,**SpinRBM_PhysHiddenIdx=rows,**GeneralRBM_PhysHiddenIdx=rows;
double complex parameters[5],*RBM=parameters;
void MakeRBMCnt(double complex *rbmCnt, const int *eleNum) {
  const int *n=eleNum;
  const int *n0=eleNum; //up-spin
  const int *n1=eleNum+Nsite; //down-spin
  int idx;
  int ri,hi,hidx,xi;
  double complex ctmp;
  /* optimization for Kei */
  const int nRBM=NRBM_PhysLayerIdx + Nneuron;
  //const int nRBM=NRBM;
  const int nSite=Nsite;
  const int nSite2=Nsite2;
  const int nNeuronGeneral=NneuronGeneral;
  const int nNeuronCharge=NneuronCharge;
  const int nNeuronSpin=NneuronSpin;
  const double complex *RBM_Hidden     = RBM+NRBM_PhysLayerIdx;
  const double complex *RBM_PhysHidden = RBM+NRBM_PhysLayerIdx+NRBM_HiddenLayerIdx;
  int offset,offset2;

  /* initialization */
  for(idx=0;idx<nRBM;idx++) rbmCnt[idx] = 0.0;

  /* Potential on Physical Layer */
  if(NChargeRBM_PhysLayerIdx>0) {
    for(ri=0;ri<nSite;ri++) {
      rbmCnt[ ChargeRBM_PhysLayerIdx[ri] ] += n0[ri]+n1[ri]-1;
    }
  }
  if(NSpinRBM_PhysLayerIdx>0) {
    offset = NChargeRBM_PhysLayerIdx;
    for(ri=0;ri<nSite;ri++) {
      rbmCnt[ SpinRBM_PhysLayerIdx[ri] + offset] += n0[ri]-n1[ri];
    }
  }
  if(NGeneralRBM_PhysLayerIdx>0) {
    offset = NChargeRBM_PhysLayerIdx + NSpinRBM_PhysLayerIdx;
    for(ri=0;ri<nSite2;ri++) {
      rbmCnt[ GeneralRBM_PhysLayerIdx[ri] + offset] += 2*n0[ri]-1;
    }
  }

  /* Potential on Hidden Layer */
  if(NChargeRBM_HiddenLayerIdx>0) {
    for(hi=0;hi<nNeuronCharge;hi++) {
      hidx = ChargeRBM_HiddenLayerIdx[hi];
      rbmCnt[hi+NRBM_PhysLayerIdx] += RBM_Hidden[hidx];
    }
  }
  if(NSpinRBM_HiddenLayerIdx>0) {
    for(hi=0;hi<nNeuronSpin;hi++) {
      hidx = SpinRBM_HiddenLayerIdx[hi];
      rbmCnt[hi+NRBM_PhysLayerIdx + NneuronCharge] += RBM_Hidden[hidx + NChargeRBM_HiddenLayerIdx];
    }
  }
  if(NGeneralRBM_HiddenLayerIdx>0) {
    offset  = NRBM_PhysLayerIdx + NneuronCharge + NneuronSpin;
    offset2 = NChargeRBM_HiddenLayerIdx + NSpinRBM_HiddenLayerIdx;
    for(hi=0;hi<nNeuronGeneral;hi++) {
      hidx = GeneralRBM_HiddenLayerIdx[hi];
      rbmCnt[hi+offset] += RBM_Hidden[hidx + offset2];
    }
  }

  /* Coupling between Phys-Hidden Layers */
  if(NChargeRBM_PhysHiddenIdx>0) {
    for(hi=0;hi<nNeuronCharge;hi++) {
      ctmp = 0.0;
      for(ri=0;ri<nSite;ri++) {
        xi = n0[ri]+n1[ri]-1;
        hidx = ChargeRBM_PhysHiddenIdx[ri][hi];
        ctmp += RBM_PhysHidden[hidx]*xi;
      }
      rbmCnt[hi+NRBM_PhysLayerIdx] += ctmp;
    }
  }
  if(NSpinRBM_PhysHiddenIdx>0) {
    for(hi=0;hi<nNeuronSpin;hi++) {
      ctmp = 0.0;
      for(ri=0;ri<nSite;ri++) {
        xi = n0[ri]-n1[ri];
        hidx = SpinRBM_PhysHiddenIdx[ri][hi];
        ctmp += RBM_PhysHidden[hidx + NChargeRBM_PhysHiddenIdx]*xi;
      }
      rbmCnt[hi+NRBM_PhysLayerIdx + NneuronCharge] += ctmp;
    }
  }
  if(NGeneralRBM_PhysHiddenIdx>0) {
    offset  = NRBM_PhysLayerIdx + NneuronCharge + NneuronSpin;
    offset2 = NChargeRBM_PhysHiddenIdx+NSpinRBM_PhysHiddenIdx;
    for(hi=0;hi<nNeuronGeneral;hi++) {
      ctmp = 0.0;
      for(ri=0;ri<nSite2;ri++) {
        xi = 2*n[ri]-1;
        hidx = GeneralRBM_PhysHiddenIdx[ri][hi];
        ctmp += RBM_PhysHidden[hidx + offset2]*xi;
      }
      rbmCnt[hi+offset] += ctmp;
    }
  }
 
  return;
}

int main(void) {
  for(int family=0;family<3;family++) for(int complex_mode=0;complex_mode<2;complex_mode++) {
    NneuronCharge=family==0; NneuronSpin=family==1; NneuronGeneral=family==2;
    NChargeRBM_HiddenLayerIdx=family==0; NSpinRBM_HiddenLayerIdx=family==1; NGeneralRBM_HiddenLayerIdx=family==2;
    NChargeRBM_PhysHiddenIdx=family==0 ? 2:0; NSpinRBM_PhysHiddenIdx=family==1 ? 2:0; NGeneralRBM_PhysHiddenIdx=family==2 ? 4:0;
    parameters[0]=1.0+(complex_mode ? 2.0:0.0)*I;
    parameters[1]=0x1p54+(complex_mode ? 0x1p55:0.0)*I;
    parameters[2]=-parameters[1]; parameters[3]=parameters[4]=0;
    int occupations[4]={1,1,family==0,family==0}; double complex counter[1];
    MakeRBMCnt(counter,occupations);
    printf("family=%d complex=%d counter=%a,%a\n",family,complex_mode,creal(counter[0]),cimag(counter[0]));
    if(counter[0]!=parameters[0]) return 1;
  }
  return 0;
}
