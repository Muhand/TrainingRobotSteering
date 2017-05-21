/* ========================================== *
* Filename:	bpnet.h                       *
* Author:		James Matthews.               *
*											  *
* Description:								  *
* This is a tiny neural network that uses	  *
* back propagation for weight adjustment.	  *
* ========================================== */
#ifndef CBPNET_H
#define CBPNET_H

#include <math.h>
#include <stdlib.h>
#include <time.h>

#define BP_LEARNING	(float)(0.5)	// The learning coefficient.

class CBPNet {
public:
	CBPNet();
	~CBPNet() {};

	float* Train(float, float, float, float);
	float* Run(float, float);

private:
	float m_fWeights[3][4];		// Weights for the 3 neurons.

	float Sigmoid(float);		// The sigmoid function.
};

#endif