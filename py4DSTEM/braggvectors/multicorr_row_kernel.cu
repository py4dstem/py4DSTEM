#include <cupy/complex.cuh>
#define PI 3.14159265359
extern "C" __global__
void multicorr_row_kernel(
	complex<float> *ar,
	const float *xyShifts,
	const long long N_pts,
	const long long image_size_x,
	const long long image_size_y,
	const long long upsample_factor) {
	/*
	Fill in the entries of the multicorr row kernel.
	Inputs (C++ type/Python type):
		ar (complex<float>* / cp.complex64):	Array of size N_pts x kernel_size x image_size[0]
				to hold the row kernels
		xyShifts (const float* / cp.float32): (N_pts x 2) array of center points to build kernels for
		N_pts (const long long/int) number of center points we are
				building kernels for
		image_size_x (const long long/int): x size of the correlation image
		image_size_y (const long long/int): y size of correlation image
		upsample_factor (const long long/int): note, kernel_width = ceil(1.5*upsample_factor)
	*/
	int kernel_size = ceil(1.5 * upsample_factor);

	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	// Using strides to compute indices:
	int stride_0 = image_size_x * kernel_size; // Stride along 0-th dimension of ar
	int stride_1 = image_size_x; // Stride along 1-th dimension of ar

	// Which kernel in the stack (first index of ar)
	int kernel_idx = tid / stride_0;
	// Which row in the kernel (second index of ar)
	int row_idx = (tid % stride_0) / stride_1;
	// Which column in the kernel (last index of ar)
	int col_idx = (tid % stride_0) % stride_1;

	complex<float> prefactor = complex<float>(0,-2.0 * PI) / float(image_size_x * upsample_factor);

	// Now do the actual calculation
	if (tid < N_pts * image_size_x * kernel_size) {
		// np.arange(numColumns) - xyShift[idx,0]
		float columnEntry = (float)row_idx - xyShifts[kernel_idx*2];

		// np.fft.ifftshift(np.arange(imageSize[0])) - np.floor(imageSize[0]/2)
		// modresult is necessary to get the Pythonic behavior of mod of negative numbers
		int modresult = int(col_idx - ceil((float)image_size_x / 2.)) % image_size_x;
		modresult = modresult < 0 ? modresult + image_size_x : modresult;
		float rowEntry = float(modresult) - floor((float)image_size_x/2.) ; 

		ar[tid] = exp(prefactor * columnEntry * rowEntry);

		// Use these for testing the indexing:
		//ar[tid] = complex<float>(0,(float)tid);
		//ar[tid] = complex<float>(0,(float)kernel_idx);
		//ar[tid] = complex<float>(0,(float)row_idx);
		//ar[tid] = complex<float>(0,(float)col_idx);
	}

}