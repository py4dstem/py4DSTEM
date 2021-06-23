import cupy as cp

__all__ = ['kernels']

kernels = {}


############################# get_maximal_points ################################
maximal_pts_float32 = r'''
extern "C" __global__
void maximal_pts(const float *ar, bool *out, const long long sizex, const long long sizey, const long long N){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int x = tid % sizex;
    int y = tid / sizey; // Floor divide
    bool res = false;
    if (tid < N && x>0 && x<(sizex-1) && y>0 && y<(sizey-1)) {
        float val = ar[tid];
        
        out[tid] = ( val > ar[tid + sizey]) &
                    (val > ar[tid - sizey]) &
                    (val > ar[tid + 1]) &
                    (val > ar[tid - 1]) &
                    (val > ar[tid - sizey - 1]) &
                    (val > ar[tid - sizey + 1]) &
                    (val > ar[tid + sizey - 1]) &
                    (val > ar[tid+sizey + 1]);
    }
}
'''

kernels['maximal_pts_float32'] = cp.RawKernel(maximal_pts_float32,'maximal_pts')

maximal_pts_float64 = r'''
extern "C" __global__
void maximal_pts(const double *ar, bool *out, const long long sizex, const long long sizey, const long long N){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int x = tid % sizex;
    int y = tid / sizey; // Floor divide
    bool res = false;
    if (tid < N && x>0 && x<(sizex-1) && y>0 && y<(sizey-1)) {
        double val = ar[tid];
        
        out[tid] = ( val > ar[tid + sizey]) &
                    (val > ar[tid - sizey]) &
                    (val > ar[tid + 1]) &
                    (val > ar[tid - 1]) &
                    (val > ar[tid - sizey - 1]) &
                    (val > ar[tid - sizey + 1]) &
                    (val > ar[tid + sizey - 1]) &
                    (val > ar[tid+sizey + 1]);
    }
}
'''

kernels['maximal_pts_float64'] = cp.RawKernel(maximal_pts_float64,'maximal_pts')