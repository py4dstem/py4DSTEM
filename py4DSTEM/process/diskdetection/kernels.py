import cupy as cp

__all__ = ['kernels']

kernels = {}


############################# get_maximal_points ################################

"""
These kernels are approximately 50x faster than the np.roll approach used in the CPU version,
per my testing with 1024x1024 pixels and float64 on a Jetson Xavier NX.
The boundary conditions are slightly different in this version, in that pixels on the edge
of the frame are always false. This simplifies the indexing, and since in the Braggdisk 
detection application an edgeBoundary is always applied in the case of subpixel detection, 
this is not considered a problem. 
"""

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



################################ edge_boundary ######################################

edge_boundary = r'''
extern "C" __global__
void edge_boundary(bool *ar, const long long edgeBoundary, 
                const long long sizex, const long long sizey, const long long N){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int x = tid % sizex;
    int y = tid / sizey; // Floor divide
    if (tid < N) {
        if (x<edgeBoundary || x>(sizex-1-edgeBoundary) || y<edgeBoundary || y>(sizey-1-edgeBoundary)){
            ar[tid] = false;
        }
    }
}
'''

kernels['edge_boundary'] = cp.RawKernel(edge_boundary,'edge_boundary')