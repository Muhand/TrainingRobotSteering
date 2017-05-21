#include <iostream>
#include "CBPNet.h"

using namespace std;

#define BPM_ITER	20000

void main() {
	CBPNet bp;

	for (int i = 0; i<BPM_ITER; i++) {
		bp.Train(200, 500, 100,20);
		bp.Train(0, 1, 0,1);
		bp.Train(1, 0, 1,0);
		bp.Train(1, 1, 1,1);
	}

	float *a = bp.Run(0, 0);
	float *b = bp.Run(0, 1);
	float *c = bp.Run(1, 0);
	float *d = bp.Run(1, 1);

	cout << "0,0 = " << a[0] << " "<< a[1] << endl;
	cout << "0,1 = " << b[0]<< " "<< b[1] << endl;
	cout << "1,0 = " << c[0]<< " "<< c[1] << endl;
	cout << "1,1 = " << d[0]<< " "<< d[1] << endl;
}