	// Author: Ce Liu (c) Dec, 2009; celiu@mit.edu
// Modified By: Deepak Pathak (c) 2016; pathak@berkeley.edu

#include "OpticalFlow.h"
#include "ImageProcessing.h"
#include "GaussianPyramid.h"
#include <cstdlib>
#include <iostream>
//Elijah
#include <chrono>
#include <map>
#include <bits/stdc++.h>
using namespace std::chrono;
using namespace std;

#include <omp.h>

#ifndef _MATLAB
	bool OpticalFlow::IsDisplay=true;
#else
	bool OpticalFlow::IsDisplay=false;
#endif

// PAP global Variable for OpenMP
int GLOBAL_nThreads;
map<string, string>* GLOBAL_timingMap;
double timer()
{
	return omp_get_wtime();
}

//OpticalFlow::InterpolationMethod OpticalFlow::interpolation = OpticalFlow::Bicubic;
OpticalFlow::InterpolationMethod OpticalFlow::interpolation = OpticalFlow::Bilinear;
OpticalFlow::NoiseModel OpticalFlow::noiseModel = OpticalFlow::Lap;
GaussianMixture OpticalFlow::GMPara;
Vector<double> OpticalFlow::LapPara;

// Global Phase parameters:
double 	TotalExecution=0.0,
				Construction=0.0,
				Allocation=0.0,
				Phase1_Generate=0.0,
				Phase2_Derivatives=0.0,
				Phase3_PsiData=0.0,
				Phase4_LinearSystem=0.0,
				Phase5_SOR=0.0,
				Phase6_Update=0.0,
				PostProcessing=0.0;


// Parallel Specific timed parameters:
double 	TotalGetDxs=0.0,
				GeneratePyramidLevels=0.0,
				total_im2feature=0.0,
				total_Multiplywith=0.0,
				total_warpImageBicubicRef=0.0,
				total_threshold=0.0,
				total_genInImageMask=0.0,
				total_Laplacian=0.0,
				total_dx=0.0,
				total_dy=0.0,
				total_add=0.0,
				total_subtract=0.0,
				total_estLaplacianNoise=0.0;

// PAP Relevant Variables:
double GLOBAL_NTHREADS;

OpticalFlow::OpticalFlow(void)
{
}

OpticalFlow::~OpticalFlow(void)
{
}

//--------------------------------------------------------------------------------------------------------
//  function to compute dx, dy and dt for motion estimation
//--------------------------------------------------------------------------------------------------------
double OpticalFlow::getDxs(DImage &imdx, DImage &imdy, DImage &imdt, const DImage &im1, const DImage &im2)
{
	double start=timer();

	double gfilter[5]={0.02,0.11,0.74,0.11,0.02};
	if(1)
	{
		DImage Im1,Im2,Im;

		im1.imfilter_hv(Im1,gfilter,2,gfilter,2);
		im2.imfilter_hv(Im2,gfilter,2,gfilter,2);
		Im.copyData(Im1);
		total_Multiplywith+=Im.Multiplywith(0.4);
		total_add+=Im.Add(Im2,0.6);

		total_dx+=Im.dx(imdx,true);
		total_dx+=Im.dy(imdy,true);
		total_subtract+=imdt.Subtract(Im2,Im1);
	}
	else
	{
		// Im1 and Im2 are the smoothed version of im1 and im2
		DImage Im1,Im2;

		im1.imfilter_hv(Im1,gfilter,2,gfilter,2);
		im2.imfilter_hv(Im2,gfilter,2,gfilter,2);

		//Im1.copyData(im1);
		//Im2.copyData(im2);

		total_dx+=Im2.dx(imdx,true);
		total_dy+=Im2.dy(imdy,true);
		total_subtract+=imdt.Subtract(Im2,Im1);
	}


	imdx.setDerivative();
	imdy.setDerivative();
	imdt.setDerivative();

	double end=timer();
	return end-start;
}

//--------------------------------------------------------------------------------------------------------
// function to do sanity check: imdx*du+imdy*dy+imdt=0
//--------------------------------------------------------------------------------------------------------
void OpticalFlow::SanityCheck(const DImage &imdx, const DImage &imdy, const DImage &imdt, double du, double dv)
{
	if(imdx.matchDimension(imdy)==false || imdx.matchDimension(imdt)==false)
	{
		cout<<"The dimensions of the derivatives don't match!"<<endl;
		return;
	}
	const _FlowPrecision* pImDx,*pImDy,*pImDt;
	pImDx=imdx.data();
	pImDy=imdy.data();
	pImDt=imdt.data();
	double error=0;
	for(int i=0;i<imdx.height();i++)
		for(int j=0;j<imdx.width();j++)
			for(int k=0;k<imdx.nchannels();k++)
			{
				int offset=(i*imdx.width()+j)*imdx.nchannels()+k;
				double temp=pImDx[offset]*du+pImDy[offset]*dv+pImDt[offset];
				error+=fabs(temp);
			}
	error/=imdx.nelements();
	cout<<"The mean error of |dx*u+dy*v+dt| is "<<error<<endl;
}

//--------------------------------------------------------------------------------------------------------
// function to warp image based on the flow field
//--------------------------------------------------------------------------------------------------------
void OpticalFlow::warpFL(DImage &warpIm2, const DImage &Im1, const DImage &Im2, const DImage &vx, const DImage &vy)
{
	if(warpIm2.matchDimension(Im2)==false)
		warpIm2.allocate(Im2.width(),Im2.height(),Im2.nchannels());
	ImageProcessing::warpImage(warpIm2.data(),Im1.data(),Im2.data(),vx.data(),vy.data(),Im2.width(),Im2.height(),Im2.nchannels());
}

void OpticalFlow::warpFL(DImage &warpIm2, const DImage &Im1, const DImage &Im2, const DImage &Flow)
{
	if(warpIm2.matchDimension(Im2)==false)
		warpIm2.allocate(Im2.width(),Im2.height(),Im2.nchannels());
	ImageProcessing::warpImageFlow(warpIm2.data(),Im1.data(),Im2.data(),Flow.data(),Im2.width(),Im2.height(),Im2.nchannels());
}


//--------------------------------------------------------------------------------------------------------
// function to generate mask of the pixels that move inside the image boundary
//--------------------------------------------------------------------------------------------------------
double OpticalFlow::genInImageMask(DImage &mask, const DImage &vx, const DImage &vy,int interval)
{
	double start=timer();

	int imWidth,imHeight;
	imWidth=vx.width();
	imHeight=vx.height();
	if(mask.matchDimension(vx)==false)
		mask.allocate(imWidth,imHeight);
	const _FlowPrecision *pVx,*pVy;
	_FlowPrecision *pMask;
	pVx=vx.data();
	pVy=vy.data();
	mask.reset();
	pMask=mask.data();
	double x,y;

	#pragma omp parallel num_threads(GLOBAL_nThreads)
	{
		#pragma omp for
		for(int i=0;i<imHeight;i++)
			for(int j=0;j<imWidth;j++)
			{
				int offset=i*imWidth+j;
				y=i+pVx[offset];
				x=j+pVy[offset];
				if(x<interval  || x>imWidth-1-interval || y<interval || y>imHeight-1-interval)
					continue;
				pMask[offset]=1;
			}
	}
	double end=timer();
	return end-start;
}

void OpticalFlow::genInImageMask(DImage &mask, const DImage &flow,int interval)
{
	int imWidth,imHeight;
	imWidth=flow.width();
	imHeight=flow.height();
	if(mask.matchDimension(flow.width(),flow.height(),1)==false)
		mask.allocate(imWidth,imHeight);
	else
		mask.reset();

	const _FlowPrecision *pFlow;
	_FlowPrecision *pMask;
	pFlow = flow.data();;
	pMask=mask.data();
	double x,y;
	for(int i=0;i<imHeight;i++)
		for(int j=0;j<imWidth;j++)
		{
			int offset=i*imWidth+j;
			y=i+pFlow[offset*2+1];
			x=j+pFlow[offset*2];
			if(x<interval  || x>imWidth-1-interval || y<interval || y>imHeight-1-interval)
				continue;
			pMask[offset]=1;
		}
}

//--------------------------------------------------------------------------------------------------------
// function to compute optical flow field using two fixed point iterations
// Input arguments:
//     Im1, Im2:						frame 1 and frame 2
//	warpIm2:						the warped frame 2 according to the current flow field u and v
//	u,v:									the current flow field, NOTICE that they are also output arguments
//
//--------------------------------------------------------------------------------------------------------
void OpticalFlow::SmoothFlowSOR(const DImage &Im1, const DImage &Im2, DImage &warpIm2, DImage &u, DImage &v,
																    double alpha, int nOuterFPIterations, int nInnerFPIterations, int nSORIterations, int nCores)
{
	DImage mask,imdx,imdy,imdt;
	int imWidth,imHeight,nChannels,nPixels;
	imWidth=Im1.width();
	imHeight=Im1.height();
	nChannels=Im1.nchannels();
	nPixels=imWidth*imHeight;

	DImage du(imWidth,imHeight),dv(imWidth,imHeight);
	DImage uu(imWidth,imHeight),vv(imWidth,imHeight);
	DImage ux(imWidth,imHeight),uy(imWidth,imHeight);
	DImage vx(imWidth,imHeight),vy(imWidth,imHeight);
	DImage Phi_1st(imWidth,imHeight);
	DImage Psi_1st(imWidth,imHeight,nChannels);

	DImage imdxy,imdx2,imdy2,imdtdx,imdtdy;
	DImage ImDxy,ImDx2,ImDy2,ImDtDx,ImDtDy;
	DImage foo1,foo2;

	double prob1,prob2,prob11,prob22;

	double varepsilon_phi=pow(0.001,2);
	double varepsilon_psi=pow(0.001,2);

	double start_time;

	//--------------------------------------------------------------------------
	// the outer fixed point iteration
	//--------------------------------------------------------------------------
	for(int count=0;count<nOuterFPIterations;count++)
	{
		//PHASE: Phase1_generate
		start_time=timer();
		// Compute the gradient
		TotalGetDxs=timer();
		getDxs(imdx,imdy,imdt,Im1,warpIm2);
		TotalGetDxs=timer()-TotalGetDxs;
		// Generate the mask to set the weight of the pxiels moving outside of the image boundary to be zero
		total_genInImageMask+=genInImageMask(mask,u,v);
		Phase1_Generate+=timer()-start_time;
		//


		// set the derivative of the flow field to be zero
		du.reset();
		dv.reset();

		//--------------------------------------------------------------------------
		// the inner fixed point iteration
		//--------------------------------------------------------------------------
		for(int hh=0;hh<nInnerFPIterations;hh++)
		{
			//PHASE: Phase2_Derivatives
			start_time=timer();
			// compute the derivatives of the current flow field
			if(hh==0)
			{
				uu.copyData(u);
				vv.copyData(v);
			}
			else
			{
				total_add+=uu.Add(u,du);
				total_add+=vv.Add(v,dv);
			}
			total_dx+=uu.dx(ux);
			total_dy+=uu.dy(uy);
			total_dx+=vv.dx(vx);
			total_dy+=vv.dy(vy);
			Phase2_Derivatives+=timer()-start_time;
			//


			// compute the weight of phi
			Phi_1st.reset();
			_FlowPrecision* phiData=Phi_1st.data();
			double temp;
			const _FlowPrecision *uxData,*uyData,*vxData,*vyData;
			uxData=ux.data();
			uyData=uy.data();
			vxData=vx.data();
			vyData=vy.data();
			//double power_alpha = 0.5;

			#pragma omp parallel num_threads(GLOBAL_nThreads)
			{
				#pragma omp for
				for(int i=0;i<nPixels;i++)
				{
					temp=uxData[i]*uxData[i]+uyData[i]*uyData[i]+vxData[i]*vxData[i]+vyData[i]*vyData[i];
					//phiData[i]=power_alpha*pow(temp+varepsilon_phi,power_alpha-1);
					phiData[i] = 0.5/sqrt(temp+varepsilon_phi);
					//phiData[i] = 1/(power_alpha+temp);
				}
			}

			// compute the nonlinear term of psi
			Psi_1st.reset();
			_FlowPrecision* psiData=Psi_1st.data();
			const _FlowPrecision *imdxData,*imdyData,*imdtData;
			const _FlowPrecision *duData,*dvData;
			imdxData=imdx.data();
			imdyData=imdy.data();
			imdtData=imdt.data();
			duData=du.data();
			dvData=dv.data();


			//PHASE: Phase3_PsiData
			start_time=timer();
			//double _a  = 10000, _b = 0.1;
			if(nChannels==1)
			{
				for(int i=0;i<nPixels;i++)
				{
					temp=imdtData[i]+imdxData[i]*duData[i]+imdyData[i]*dvData[i];
					//if(temp*temp<0.04)
					// psiData[i]=1/(2*sqrt(temp*temp+varepsilon_psi));
					//psiData[i] = _a*_b/(1+_a*temp*temp);

					// the following code is for log Gaussian mixture probability model
					temp *= temp;
					switch(noiseModel)
					{
						case GMixture:
							prob1 = GMPara.Gaussian(temp,0,0)*GMPara.alpha[0];
							prob2 = GMPara.Gaussian(temp,1,0)*(1-GMPara.alpha[0]);
							prob11 = prob1/(2*GMPara.sigma_square[0]);
							prob22 = prob2/(2*GMPara.beta_square[0]);
							psiData[i] = (prob11+prob22)/(prob1+prob2);
							break;
						case Lap:
							if(LapPara[0]<1E-20)
								continue;
							//psiData[i]=1/(2*sqrt(temp+varepsilon_psi)*LapPara[0]);
	                        psiData[i]=1/(2*sqrt(temp+varepsilon_psi));
							break;
					}
				}
			}
			else // Multiple Channels
			{
				#pragma omp parallel num_threads(GLOBAL_nThreads)
				{
					#pragma omp for
					for(int i=0;i<nPixels;i++)
						for(int k=0;k<nChannels;k++)
						{
							// Each pixel is nChannels, this channel is pixel*channels_per_pixel + channel #
							int offset=i*nChannels+k;
							temp=imdtData[offset]+imdxData[offset]*duData[i]+imdyData[offset]*dvData[i];
							//if(temp*temp<0.04)
							 // psiData[offset]=1/(2*sqrt(temp*temp+varepsilon_psi));
							//psiData[offset] =  _a*_b/(1+_a*temp*temp);
							temp *= temp;
							switch(noiseModel)
							{
							case GMixture:
								prob1 = GMPara.Gaussian(temp,0,k)*GMPara.alpha[k];
								prob2 = GMPara.Gaussian(temp,1,k)*(1-GMPara.alpha[k]);
								prob11 = prob1/(2*GMPara.sigma_square[k]);
								prob22 = prob2/(2*GMPara.beta_square[k]);
								psiData[offset] = (prob11+prob22)/(prob1+prob2);
								break;
							case Lap:
								if(LapPara[k]<1E-20)
									continue;
								//psiData[offset]=1/(2*sqrt(temp+varepsilon_psi)*LapPara[k]);
	                            psiData[offset]=1/(2*sqrt(temp+varepsilon_psi));
								break;
							}
						}
				}//end parallel
			}
			Phase3_PsiData+=timer()-start_time;
			//


			//PHASE: Phase4_LinearSystem
			start_time=timer();
			// prepare the components of the large linear system
			ImDxy.Multiply(Psi_1st,imdx,imdy);
			ImDx2.Multiply(Psi_1st,imdx,imdx);
			ImDy2.Multiply(Psi_1st,imdy,imdy);
			ImDtDx.Multiply(Psi_1st,imdx,imdt);
			ImDtDy.Multiply(Psi_1st,imdy,imdt);

			if(nChannels>1)
			{
				ImDxy.collapse(imdxy);
				ImDx2.collapse(imdx2);
				ImDy2.collapse(imdy2);
				ImDtDx.collapse(imdtdx);
				ImDtDy.collapse(imdtdy);
			}
			else
			{
				imdxy.copyData(ImDxy);
				imdx2.copyData(ImDx2);
				imdy2.copyData(ImDy2);
				imdtdx.copyData(ImDtDx);
				imdtdy.copyData(ImDtDy);
			}
			// laplacian filtering of the current flow field
		  total_Laplacian+=  Laplacian(foo1,u,Phi_1st);
			total_Laplacian+=  Laplacian(foo2,v,Phi_1st);

			Phase4_LinearSystem+=timer()-start_time;
			//


			for(int i=0;i<nPixels;i++)
			{
				imdtdx.data()[i] = -imdtdx.data()[i]-alpha*foo1.data()[i];
				imdtdy.data()[i] = -imdtdy.data()[i]-alpha*foo2.data()[i];
			}

			// set omega
			double omega = 1.8;
			du.reset();
			dv.reset();


			//PHASE: Phase5_SOR
			start_time=timer();
			#pragma omp parallel num_threads(nCores)
			{
				#pragma omp for
				for(int k = 0; k<nSORIterations; k++)
						for(int i = 0; i<imHeight; i++)
							for(int j = 0; j<imWidth; j++)
							{
								// Assert: offset is unique to each loop, no race conditions
								int offset = i * imWidth+j;
								double sigma1 = 0, sigma2 = 0, coeff = 0;
		                        double _weight;


								if(j>0)
								{
		                            _weight = phiData[offset-1];
									sigma1  += _weight*du.data()[offset-1];
									sigma2  += _weight*dv.data()[offset-1];
									coeff   += _weight;
								}
								if(j<imWidth-1)
								{
		                            _weight = phiData[offset];
									sigma1 += _weight*du.data()[offset+1];
									sigma2 += _weight*dv.data()[offset+1];
									coeff   += _weight;
								}
								if(i>0)
								{
		                            _weight = phiData[offset-imWidth];
									sigma1 += _weight*du.data()[offset-imWidth];
									sigma2 += _weight*dv.data()[offset-imWidth];
									coeff   += _weight;
								}
								if(i<imHeight-1)
								{
		                            _weight = phiData[offset];
									sigma1  += _weight*du.data()[offset+imWidth];
									sigma2  += _weight*dv.data()[offset+imWidth];
									coeff   += _weight;
								}
								sigma1 *= -alpha;
								sigma2 *= -alpha;
								coeff *= alpha;
								 // compute du
								sigma1 += imdxy.data()[offset]*dv.data()[offset];
								du.data()[offset] = (1-omega)*du.data()[offset] + omega/(imdx2.data()[offset] + alpha*0.05 + coeff)*(imdtdx.data()[offset] - sigma1);
								// compute dv
								sigma2 += imdxy.data()[offset]*du.data()[offset];
								dv.data()[offset] = (1-omega)*dv.data()[offset] + omega/(imdy2.data()[offset] + alpha*0.05 + coeff)*(imdtdy.data()[offset] - sigma2);
							} // end SOR loops
				} // End parallel
		} // End InnerFPIterations
		Phase5_SOR+=timer()-start_time;
		//


		//PHASE: Phase6_Update
		start_time=timer();
		total_add+=u.Add(du);
		total_add+=v.Add(dv);
		if(interpolation == Bilinear)
			warpFL(warpIm2,Im1,Im2,u,v);
		else
		{
			total_warpImageBicubicRef+= Im2.warpImageBicubicRef(Im1,warpIm2,u,v);
			total_threshold+=warpIm2.threshold();
		}

		// estimate noise level
		switch(noiseModel)
		{
		case GMixture:
			estGaussianMixture(Im1,warpIm2,GMPara);
			break;
		case Lap:
			total_estLaplacianNoise+=estLaplacianNoise(Im1,warpIm2,LapPara);
		}
		Phase6_Update+=timer()-start_time;
		//

	} // End OuterFPIterations
}	// End SmoothFlowSOR


void OpticalFlow::estGaussianMixture(const DImage& Im1,const DImage& Im2,GaussianMixture& para,double prior)
{
	int nIterations = 3, nChannels = Im1.nchannels();
	DImage weight1(Im1),weight2(Im1);
	double *total1,*total2;
	total1 = new double[nChannels];
	total2 = new double[nChannels];
	for(int count = 0; count<nIterations; count++)
	{
		double temp;
		memset(total1,0,sizeof(double)*nChannels);
		memset(total2,0,sizeof(double)*nChannels);

		// E step
		for(int i = 0;i<weight1.npixels();i++)
			for(int k=0;k<nChannels;k++)
			{
				int offset = i*weight1.nchannels()+k;
				temp = Im1[offset]-Im2[offset];
				temp *= temp;
				weight1[offset] = para.Gaussian(temp,0,k)*para.alpha[k];
				weight2[offset] = para.Gaussian(temp,1,k)*(1-para.alpha[k]);
				temp = weight1[offset]+weight2[offset];
				weight1[offset]/=temp;
				weight2[offset]/=temp;
				total1[k] += weight1[offset];
				total2[k] += weight2[offset];
			}

		// M step
		para.reset();


		for(int i = 0;i<weight1.npixels();i++)
			for(int k =0;k<nChannels;k++)
			{
				int offset = i*weight1.nchannels()+k;
				temp = Im1[offset]-Im2[offset];
				temp *= temp;
				para.sigma[k]+= weight1[offset]*temp;
				para.beta[k] += weight2[offset]*temp;
			}

		for(int k =0;k<nChannels;k++)
		{
			para.alpha[k] = total1[k]/(total1[k]+total2[k])*(1-prior)+0.95*prior; // regularize alpha
			para.sigma[k] = sqrt(para.sigma[k]/total1[k]);
			para.beta[k]   = sqrt(para.beta[k]/total2[k])*(1-prior)+0.3*prior; // regularize beta
		}
		para.square();
		count = count;
	}
}


double OpticalFlow::estLaplacianNoise(const DImage& Im1,const DImage& Im2,Vector<double>& para)
{
	double start=timer();

	int nChannels = Im1.nchannels();
	if(para.dim()!=nChannels)
		para.allocate(nChannels);
	else
		para.reset();
	double temp;
	Vector<double> total(nChannels);
	for(int k = 0;k<nChannels;k++)
		total[k] = 0;

	for(int i =0;i<Im1.npixels();i++)
		for(int k = 0;k<nChannels;k++)
		{
			// offset = this channel
			int offset = i*nChannels+k;
			temp= fabs(Im1.data()[offset]-Im2.data()[offset]);
			if(temp>0 && temp<1000000)
			{
				// k={0,1,2} so these will need to be Pragma ATOMIC
				para[k] += temp;

				total[k]++;
			}
		}


	for(int k = 0;k<nChannels;k++)
	{
		if(total[k]==0)
		{
			//cout<<"All the pixels are invalid in estimation Laplacian noise!!!"<<endl;
			//cout<<"Something severely wrong happened!!!"<<endl;
			cout << "total[" << k << "]=" << total[k] << endl;
			para[k] = 0.001;
		}
		else
			para[k]/=total[k];
	}

	double end=timer();
	return end-start;
}

/*
double OpticalFlow::Laplacian(DImage &output, const DImage &input, const DImage& weight)
{
	double start=timer();

	if(output.matchDimension(input)==false)
		output.allocate(input);
	output.reset();

	if(input.matchDimension(weight)==false)
	{
		cout<<"Error in image dimension matching OpticalFlow::Laplacian()!"<<endl;
		return 0.0;
	}

	const _FlowPrecision *inputData=input.data(),*weightData=weight.data();
	int width=input.width(),height=input.height();
	DImage foo(width,height);
	_FlowPrecision *fooData=foo.data(),*outputData=output.data();


	// horizontal filtering
	#pragma omp parallel num_threads(GLOBAL_nThreads)
	{
		#pragma omp for
		for(int i=0;i<height;i++)
			for(int j=0;j<width-1;j++)
			{
				int offset=i*width+j;
				fooData[offset]=(inputData[offset+1]-inputData[offset])*weightData[offset];
			}
		#pragma omp for
		for(int i=0;i<height;i++)
			for(int j=0;j<width;j++)
			{
				int offset=i*width+j;
				if(j<width-1)
					outputData[offset]-=fooData[offset];
				if(j>0)
					outputData[offset]+=fooData[offset-1];
			}

		#pragma omp barrier
		foo.reset();

		#pragma omp for
		for(int i=0;i<height-1;i++)
			for(int j=0;j<width;j++)
			{
				int offset=i*width+j;
				fooData[offset]=(inputData[offset+width]-inputData[offset])*weightData[offset];
			}
		#pragma omp for
		for(int i=0;i<height;i++)
			for(int j=0;j<width;j++)
			{
				int offset=i*width+j;
				if(i<height-1)
					outputData[offset]-=fooData[offset];
				if(i>0)
					outputData[offset]+=fooData[offset-width];
			}
	}

	double end=timer();
	return end-start;
}
*/

double OpticalFlow::Laplacian(DImage &output, const DImage &input, const DImage& weight)
{
	double start=timer();

	if(output.matchDimension(input)==false)
		output.allocate(input);
	output.reset();

	if(input.matchDimension(weight)==false)
	{
		cout<<"Error in image dimension matching OpticalFlow::Laplacian()!"<<endl;
		return 0.0;
	}

	const _FlowPrecision *inputData=input.data(),*weightData=weight.data();
	int width=input.width(),height=input.height();
	DImage foo(width,height);
	_FlowPrecision *fooData=foo.data(),*outputData=output.data();


	// horizontal filtering
	#pragma omp parallel num_threads(GLOBAL_nThreads)
	{
		#pragma omp for
		for(int i=0;i<height;i++)
			for(int j=0;j<width-1;j++)
			{
				int offset=i*width+j;
				fooData[offset]=(inputData[offset+1]-inputData[offset])*weightData[offset];
				if(j<width-1)
					outputData[offset]-=fooData[offset];
				if(j>0)
					outputData[offset]+=fooData[offset-1];
			}


		foo.reset();

		#pragma omp for
		for(int i=0;i<height-1;i++)
			for(int j=0;j<width;j++)
			{
				int offset=i*width+j;
				fooData[offset]=(inputData[offset+width]-inputData[offset])*weightData[offset];
				if(i<height-1)
					outputData[offset]-=fooData[offset];
				if(i>0)
					outputData[offset]+=fooData[offset-width];
			}
	}

	double end=timer();
	return end-start;
}


void OpticalFlow::testLaplacian(int dim)
{
	// generate the random weight
	DImage weight(dim,dim);
	for(int i=0;i<dim;i++)
		for(int j=0;j<dim;j++)
			//weight.data()[i*dim+j]=(double)rand()/RAND_MAX+1;
			weight.data()[i*dim+j]=1;
	// go through the linear system;
	DImage sysMatrix(dim*dim,dim*dim);
	DImage u(dim,dim),du(dim,dim);
	for(int i=0;i<dim*dim;i++)
	{
		u.reset();
		u.data()[i]=1;
		Laplacian(du,u,weight);
		for(int j=0;j<dim*dim;j++)
			sysMatrix.data()[j*dim*dim+i]=du.data()[j];
	}
	// test whether the matrix is symmetric
	for(int i=0;i<dim*dim;i++)
	{
		for(int j=0;j<dim*dim;j++)
		{
			if(sysMatrix.data()[i*dim*dim+j]>=0)
				printf(" ");
			printf(" %1.0f ",sysMatrix.data()[i*dim*dim+j]);
		}
		printf("\n");
	}
}


//-------------------------------------------------------
// PAP_Team version of Coarse2FineFlow
//		-This function is the highest level of Optical Flow
//					calculation in the cpp files,
//		-All of the diagnostic timing will be performed here
//					and passed back to Python using the TIMING_PROFILE map
//
//-------------------------------------------------------
void myfunc(){cout<<"heloo";}
void OpticalFlow::Coarse2FineFlow(map<string,string>* TIMING_PROFILE, DImage &vx, DImage &vy, DImage &warpI2,const DImage &Im1, const DImage &Im2, int pyramidLevels, int nCores)
{
	// ASSERT: Coarse2FineFlow will always execute before Image.h > Global Variables will always be defined
	// === Pass global variables for use amongst scripts
	GLOBAL_nThreads=nCores;
	GLOBAL_timingMap=TIMING_PROFILE;
	TotalExecution=timer();
	double time;

	//PHASE: Construction
	Construction=timer();

	// Hardcoded values
  double alpha = 0.012;
  double ratio = 0.75;
  int nOuterFPIterations = 7;
  int nInnerFPIterations = 1;
  int nCGIterations = 30;
	bool IsDisplay=true;
	// === SetUp: Calculate Pyramid Set-up with GPyramids
	GeneratePyramidLevels=timer();
	GaussianPyramid GPyramid1;
	GaussianPyramid GPyramid2;
	GPyramid1.ConstructPyramidLevels(Im1,ratio,pyramidLevels);
	GPyramid2.ConstructPyramidLevels(Im2,ratio,pyramidLevels);
	GeneratePyramidLevels=timer()-GeneratePyramidLevels;
	// Accumulators
	double duration_total_flow=0.0;
	double debug_timer=0.0;
	// Image Processing Data Structures
	DImage Image1,Image2,WarpImage2;


	// === Input: initialize noise
	switch(noiseModel){
	case GMixture:
		GMPara.reset(Im1.nchannels()+2);
		break;
	case Lap:
		LapPara.allocate(Im1.nchannels()+2);
		for(int i = 0;i<LapPara.dim();i++)
			LapPara[i] = 0.02;
		break;
	}
	Construction=timer()-Construction;
	//


	// P Y R A M I D
	double image_flow_begin, image_flow_end,duration_image_flow;
	for(int k=GPyramid1.nlevels()-1;k>=0;k--)
	{
		//=== [P] DEBUG: Pyramid Level
		if(IsDisplay)
			cout<<"P["<<to_string(k)<<"]"<<flush;

		//PHASE: Allocation
		debug_timer=timer();
		time=timer();

		//=== [P]: Allocate Pyramid Images
		int width=GPyramid1.Image(k).width();
		int height=GPyramid1.Image(k).height();
		total_im2feature+=im2feature(Image1,GPyramid1.Image(k));
		total_im2feature+=im2feature(Image2,GPyramid2.Image(k));

		//=== [P]: Allocate Derivative Images
		if(k==GPyramid1.nlevels()-1) 		// if at the top level
		{
			vx.allocate(width,height);
			vy.allocate(width,height);
			WarpImage2.copyData(Image2);
		}
		else														// not at top level (Interpolation=Bilinear)
		{
			vx.imresize(width,height);
			total_Multiplywith+= vx.Multiplywith(1/ratio);
			vy.imresize(width,height);
			total_Multiplywith+= vy.Multiplywith(1/ratio);
			if(interpolation == Bilinear)
				warpFL(WarpImage2,Image1,Image2,vx,vy);
			else
				total_warpImageBicubicRef+= Image2.warpImageBicubicRef(Image1,WarpImage2,vx,vy);
		}
		Allocation+= timer()-time;
		//

		//=== [P]: Calculate Optical Flow
		image_flow_begin=timer();
		SmoothFlowSOR(Image1,Image2,WarpImage2,vx,vy,alpha,nOuterFPIterations+k,nInnerFPIterations,nCGIterations+k*3, nCores);
		image_flow_end=timer();

		//=== [P]: Timing
		duration_image_flow=image_flow_end-image_flow_begin;
		duration_total_flow+=duration_image_flow;

		//=== [P] DEBUG: print elapsed time for this pyramid level
		debug_timer=timer()-debug_timer;
		cout<<"("<< fixed<<setprecision(1)<<debug_timer <<"s) "<<flush;
	}
	printf("! \n");
	cout<<flush;
	// End Pyramid


	//PHASE: PostProcessing
	PostProcessing=timer();
	total_warpImageBicubicRef+= Im2.warpImageBicubicRef(Im1,warpI2,vx,vy);
	total_threshold+=warpI2.threshold();
	PostProcessing=timer()-PostProcessing;
	//


	TotalExecution=timer()-TotalExecution;

	// === Output: Store the values
	GLOBAL_timingMap->insert( make_pair("Total C++ Execution",to_string( TotalExecution )) );
					//GLOBAL_timingMap->insert( make_pair("Total Flow Calculation",to_string( duration_total_flow )) );
	GLOBAL_timingMap->insert( make_pair("Construction",to_string( Construction )) );
	GLOBAL_timingMap->insert( make_pair("Allocation",to_string( Allocation )) );
	GLOBAL_timingMap->insert( make_pair("Phase1_Generate",to_string( Phase1_Generate )) );
	GLOBAL_timingMap->insert( make_pair("Phase2_Derivatives",to_string( Phase2_Derivatives )) );
	GLOBAL_timingMap->insert( make_pair("Phase3_PsiData",to_string( Phase3_PsiData )) );
	GLOBAL_timingMap->insert( make_pair("Phase4_LinearSystem",to_string( Phase4_LinearSystem )) );
	GLOBAL_timingMap->insert( make_pair("Phase5_SOR",to_string( Phase5_SOR )) );
	GLOBAL_timingMap->insert( make_pair("Phase6_Update",to_string( Phase6_Update )) );
	GLOBAL_timingMap->insert( make_pair("PostProcessing",to_string( PostProcessing )) );


	// GLOBAL_timingMap->insert( make_pair("_Total getDxs",to_string( TotalGetDxs )) );
	// GLOBAL_timingMap->insert( make_pair("Generate Pyramid Levels",to_string( GeneratePyramidLevels )) );
	// GLOBAL_timingMap->insert( make_pair("im2feature",to_string( total_im2feature )) );
	// GLOBAL_timingMap->insert( make_pair("multiplyWith",to_string( total_Multiplywith )) );
	// GLOBAL_timingMap->insert( make_pair("dx",to_string( total_dx )) );
	// GLOBAL_timingMap->insert( make_pair("dy",to_string( total_dy )) );
	// GLOBAL_timingMap->insert( make_pair("add",to_string( total_add )) );
	// GLOBAL_timingMap->insert( make_pair("subtract",to_string( total_subtract )) );
	// GLOBAL_timingMap->insert( make_pair("warpImageBicubicRef",to_string( total_warpImageBicubicRef )) );
	// GLOBAL_timingMap->insert( make_pair("threshold",to_string( total_threshold )) );
	// GLOBAL_timingMap->insert( make_pair("genInImageMask",to_string( total_genInImageMask )) );
	// GLOBAL_timingMap->insert( make_pair("Laplacian",to_string( total_Laplacian )) );
	// GLOBAL_timingMap->insert( make_pair("estLaplacianNoise",to_string( total_estLaplacianNoise )) );


	// Resetting the global variables. "Dont try global variables, kids"
	TotalExecution=0.0;
	Construction=0.0;
	Allocation=0.0;
	Phase1_Generate=0.0;
	Phase2_Derivatives=0.0;
	Phase3_PsiData=0.0;
	Phase4_LinearSystem=0.0;
	Phase5_SOR=0.0;
	Phase6_Update=0.0;
	PostProcessing=0.0;

	TotalGetDxs=0.0;
	GeneratePyramidLevels=0.0;
	total_im2feature=0.0;
	total_Multiplywith=0.0;
	total_warpImageBicubicRef=0.0;
	total_threshold=0.0;
	total_genInImageMask=0.0;
	total_Laplacian=0.0;
	total_dx=0.0;
	total_dy=0.0;
	total_add=0.0;
	total_subtract=0.0;
	total_estLaplacianNoise=0.0;
}
//-------------------------
// End PAP_Team optical flow call
//-------------------------

//---------------------------------------------------------------------------------------
// function to convert image to feature image
//---------------------------------------------------------------------------------------
double OpticalFlow::im2feature(DImage &imfeature, const DImage &im)
{
	double start=timer();

	int width=im.width();
	int height=im.height();
	int nchannels=im.nchannels();
	if(nchannels==1)
	{
		imfeature.allocate(im.width(),im.height(),3);
		DImage imdx,imdy;
		total_dx+=im.dx(imdx,true);
		total_dy+=im.dy(imdy,true);
		_FlowPrecision* data=imfeature.data();
		for(int i=0;i<height;i++)
			for(int j=0;j<width;j++)
			{
				int offset=i*width+j;
				data[offset*3]=im.data()[offset];
				data[offset*3+1]=imdx.data()[offset];
				data[offset*3+2]=imdy.data()[offset];
			}
	}
	else if(nchannels==3)
	{
		DImage grayImage;
		im.desaturate(grayImage);

		imfeature.allocate(im.width(),im.height(),5);
		DImage imdx,imdy;
		total_dx+=grayImage.dx(imdx,true);
		total_dy+=grayImage.dy(imdy,true);
		_FlowPrecision* data=imfeature.data();
		#pragma omp parallel num_threads(GLOBAL_nThreads)
		{
			#pragma omp for
			for(int i=0;i<height;i++)
				for(int j=0;j<width;j++)
				{
					int offset=i*width+j;
					data[offset*5]=grayImage.data()[offset];
					data[offset*5+1]=imdx.data()[offset];
					data[offset*5+2]=imdy.data()[offset];
					data[offset*5+3]=im.data()[offset*3+1]-im.data()[offset*3];
					data[offset*5+4]=im.data()[offset*3+1]-im.data()[offset*3+2];
				}
		}// end parallel
	}
	else
		imfeature.copyData(im);

	double end=timer();
	return end-start;
}

bool OpticalFlow::LoadOpticalFlow(const char* filename,DImage &flow)
{
	Image<unsigned short int> foo;
	if(foo.loadImage(filename) == false)
		return false;
	if(!flow.matchDimension(foo))
		flow.allocate(foo);
	for(int  i = 0;i<flow.npixels();i++)
	{
		flow.data()[i*2] = (double)foo.data()[i*2]/160-200;
		flow.data()[i*2+1] = (double)foo.data()[i*2+1]/160-200;
	}
	return true;
}

bool OpticalFlow::LoadOpticalFlow(ifstream& myfile,DImage& flow)
{
	Image<unsigned short int> foo;
	if(foo.loadImage(myfile) == false)
		return false;
	if(!flow.matchDimension(foo))
		flow.allocate(foo);
	for(int  i = 0;i<flow.npixels();i++)
	{
		flow.data()[i*2] = (double)foo.data()[i*2]/160-200;
		flow.data()[i*2+1] = (double)foo.data()[i*2+1]/160-200;
	}
	return true;
}

bool OpticalFlow::SaveOpticalFlow(const DImage& flow, const char* filename)
{
	Image<unsigned short int> foo;
	foo.allocate(flow);
	for(int i =0;i<flow.npixels();i++)
	{
		foo.data()[i*2] = (__min(__max(flow.data()[i*2],-200),200)+200)*160;
		foo.data()[i*2+1] = (__min(__max(flow.data()[i*2+1],-200),200)+200)*160;
	}
	return foo.saveImage(filename);
}

bool OpticalFlow::SaveOpticalFlow(const DImage& flow,ofstream& myfile)
{
	Image<unsigned short int> foo;
	foo.allocate(flow);
	for(int i =0;i<flow.npixels();i++)
	{
		foo.data()[i*2] = (__min(__max(flow.data()[i*2],-200),200)+200)*160;
		foo.data()[i*2+1] = (__min(__max(flow.data()[i*2+1],-200),200)+200)*160;
	}
	return foo.saveImage(myfile);
}

bool OpticalFlow::showFlow(const DImage& flow,const char* filename)
{
	if(flow.nchannels()!=1)
	{
		cout<<"The flow must be a single channel image!"<<endl;
		return false;
	}
	Image<unsigned char> foo;
	foo.allocate(flow.width(),flow.height());
	double Max = flow.max();
	double Min = flow.min();
	for(int i = 0;i<flow.npixels(); i++)
		foo[i] = (flow[i]-Min)/(Max-Min)*255;
  // opencv support disabled. Can no longer write images.
	// foo.imwrite(filename);
  return false;
}
