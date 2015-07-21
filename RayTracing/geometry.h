#ifndef GEOMETRY_H
#define GEOMETRY_H
#include "vector_types_math.h"

struct Triangle
{
    float3 v1;
    float3 v2;
    float3 v3;
};

struct Ray
{
    float3 origin;
    float3 direction;
};

#endif