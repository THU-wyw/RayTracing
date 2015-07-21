#ifndef BVH_H
#define BVH_H
#include "vector_types_math.h"

struct Model;
struct Bvh
{
    struct MortonCode
    {
        uint code;
        int index;
    };

    struct InnerNode
    {
        int father_index;
        int low_index;
        int high_index;
        bool left_is_leaf;
        bool right_is_leaf;
        int left_index;
        int right_index;
        float3 min_bound;
        float3 max_bound;
    };

    struct LeafNode
    {
        int father_index;
        int low_index;
        int high_index;
        float3 min_bound;
        float3 max_bound;
    };

    InnerNode* inner_nodes;
    LeafNode* leaf_nodes;
    int* original_index;
    int leaf_number;
};

extern Bvh BuildDeviceBvh(const Model& device_model);
#endif