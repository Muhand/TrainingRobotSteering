#include "CBPNet.h"

CBPNet::CBPNet() {
	srand((unsigned)(time(NULL)));

	for (int i = 0; i<3; i++) {
		for (int j = 0; j<4; j++) {
			// For some reason, the Microsoft rand() function
			// generates a random integer. So, I divide by the
			// number by MAXINT/2, to get a num between 0 and 2,
			// the subtract one to get a num between -1 and 1.
			m_fWeights[i][j] = (float)(rand()) / (32767 / 2) - 1;
		}
	}
}

float* CBPNet::Train(float i1, float i2, float d1, float d2) {
	// These are all the main variables used in the 
	// routine. Seems easier to group them all here.
	float net1, net2, i3, i4;//, out1, out2;
	float *out = new float[2];

	// Calculate the net values for the hidden layer neurons.
	net1 = 1 * m_fWeights[0][0] + i1 * m_fWeights[1][0] +
		i2 * m_fWeights[2][0];
	net2 = 1 * m_fWeights[0][1] + i1 * m_fWeights[1][1] +
		i2 * m_fWeights[2][1];

	// Use the hardlimiter function - the Sigmoid.
	i3 = Sigmoid(net1);
	i4 = Sigmoid(net2);

	// Now, calculate the net for the final output layer.
	net1 = 1 * m_fWeights[0][2] + i3 * m_fWeights[1][2] +
		i4 * m_fWeights[2][2];
	net2 = 1 * m_fWeights[0][3] + i3 * m_fWeights[1][3] +
		i4 * m_fWeights[2][3];

	out[0] = Sigmoid(net1);
	out[1] = Sigmoid(net2);

	// We have to calculate the deltas for the two layers.
	// Remember, we have to calculate the errors backwards
	// from the output layer to the hidden layer (thus the
	// name 'BACK-propagation').
	float deltas1[3];
	float deltas2[3];

	deltas1[2] = out[0]*(1 - out[0])*(d1 - out[0]);
	deltas1[1] = i4*(1 - i4)*(m_fWeights[2][2])*(deltas1[2]);
	deltas1[0] = i3*(1 - i3)*(m_fWeights[1][2])*(deltas1[2]);

	deltas2[2] = out[1]*(1 - out[1])*(d2 - out[1]);
	deltas2[1] = i4*(1 - i4)*(m_fWeights[2][3])*(deltas2[2]);
	deltas2[0] = i3*(1 - i3)*(m_fWeights[1][3])*(deltas2[2]);



	// Now, alter the weights accordingly.
	float v1 = i1, v2 = i2;
	for (int i = 0; i<3; i++) {
		// Change the values for the output layer, if necessary.
		if (i == 2) {
			v1 = i3;
			v2 = i4;
		}

		m_fWeights[0][i] += BP_LEARNING * 1 * deltas1[i];
		m_fWeights[1][i] += BP_LEARNING * v1 *deltas1[i];
		m_fWeights[2][i] += BP_LEARNING * v2 *deltas1[i];
	}

	
	v1 = i1, v2 = i2;
	for (int i = 0; i<4; i++) {
		// Change the values for the output layer, if necessary.
		if (i == 3) {
			v1 = i3;
			v2 = i4;

			m_fWeights[0][i] += BP_LEARNING * 1 * deltas2[2];
			m_fWeights[1][i] += BP_LEARNING * v1 *deltas2[2];
			m_fWeights[2][i] += BP_LEARNING * v2 *deltas2[2];
		}
		else
		{
			m_fWeights[0][i] += BP_LEARNING * 1 * deltas2[i];
			m_fWeights[1][i] += BP_LEARNING * v1 *deltas2[i];
			m_fWeights[2][i] += BP_LEARNING * v2 *deltas2[i];
		}

	}

	return out;
}

float CBPNet::Sigmoid(float num) {
	return (float)(1 / (1 + exp(-num)));
}

float* CBPNet::Run(float i1, float i2) {
	// I just copied and pasted the code from the Train() function,
	// so see there for the necessary documentation.

	float net1, net2, i3, i4;
	float *out = new float[2];

	net1 = 1 * m_fWeights[0][0] + i1 * m_fWeights[1][0] +
		i2 * m_fWeights[2][0];
	net2 = 1 * m_fWeights[0][1] + i1 * m_fWeights[1][1] +
		i2 * m_fWeights[2][1];

	i3 = Sigmoid(net1);
	i4 = Sigmoid(net2);

	net1 = 1 * m_fWeights[0][2] + i3 * m_fWeights[1][2] +
		i4 * m_fWeights[2][2];
	net2 = 1 * m_fWeights[0][3] + i3 * m_fWeights[1][3] +
		i4 * m_fWeights[2][3];


	out[0] = Sigmoid(net1);
	out[1] = Sigmoid(net2);

	return out;
}
