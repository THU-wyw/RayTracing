#ifndef TRAVERSAL_H
#define TRAVERSAL_H

#include <cuda_runtime.h>

struct Model;
struct Bvh;
struct ScreenParams;
extern cudaError_t GetColorAllPixels(const ScreenParams& params,
                                     const Model& model,
                                     const Bvh& bvh,
                                     bool *vhit,
                                     float *vcolor);
#endif