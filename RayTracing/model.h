#ifndef READ_MODEL_H
#define READ_MODEL_H

#include "geometry.h"
struct Model
{
    float3* points;
    Triangle* triangles;
    int triangle_number;
    int point_number;
    int leaf_number;
    float3 global_min;
    float3 global_max;
};

extern Model ReadModel(const char* filename);

extern Model GetDeviceCopy(const Model& host_model);
#endif