#include "test.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include "bvh.h"
#include "screen.h"
#include "geometry.h"
#include <cmath>
#include <cstdint>
using namespace std;

void test_bvh(const Bvh& device_bvh)
{
    Bvh bvh;
    bvh.leaf_number = device_bvh.leaf_number;
    bvh.inner_nodes = new Bvh::InnerNode[bvh.leaf_number - 1];
    bvh.leaf_nodes = new Bvh::LeafNode[bvh.leaf_number - 1];
    cudaMemcpy(bvh.inner_nodes, device_bvh.inner_nodes, sizeof(Bvh::InnerNode) * (bvh.leaf_number - 1), cudaMemcpyDeviceToHost);
    cudaMemcpy(bvh.leaf_nodes, device_bvh.leaf_nodes, sizeof(Bvh::LeafNode) * (bvh.leaf_number), cudaMemcpyDeviceToHost);
    int* height = new int[bvh.leaf_number];
    int max_height = 0;
    for (int i = 0; i < bvh.leaf_number; ++i)
    {
        height[i] = 0;
        int current = bvh.leaf_nodes[i].father_index;
        while (current != -1)
        {
            current = bvh.inner_nodes[current].father_index;
            height[i]++;
        }
        if (height[i] > max_height) max_height = height[i];
    }
    int * distribution = new int[max_height + 1];
    for (int i = 0; i <= max_height; ++i)
        distribution[i] = 0;
    for (int i = 0; i < bvh.leaf_number; ++i)
        distribution[height[i]]++;
    std::ofstream output("bvh.txt");
    for (int i = 0; i <= max_height; ++i)
        output << i << '\t' << distribution[i] << std::endl;
    output.close();
    delete[] height;
    delete[] distribution;
}

void test_contract_rays(const Ray* rays, int number, const ScreenParams& params)
{
    std::vector<uint> hash_values(number);
    const float CELL_FRACTION = 0.004;
    const float MAX_SIZE = 256;
    const float PI = 3.14159265f;
    float3 tmp1 = params.length_screen_start;
    float3 tmp2 = params.length_screen_start +
        ScrHeight / 80 * params.length_up +
        ScrWidth / 80 * params.length_right;
    float3 low = fminf(tmp1, tmp2);
    float3 high = fmaxf(tmp1, tmp2);
    float cell_size = CELL_FRACTION * length(high - low);
    for (int i = 0; i < number; i++)
    {
        float3 relative_origin = rays[i].origin - low;
        uint x = relative_origin.x / cell_size;
        uint y = relative_origin.y / cell_size;
        uint z = relative_origin.z / cell_size;
        float theta = acos(rays[i].direction.z / length(rays[i].direction));
        float phi = atan2(rays[i].direction.y, rays[i].direction.x);
        uint a1 = theta * 16 / PI;
        uint a2 = theta * 16 / (2 * PI);
        hash_values[i] = (x << 24) + (y << 16) + (z << 8) + ((a1 & 0xF) << 4) + a2;
    }
    std::sort(hash_values.begin(), hash_values.end());
    int count = 0;
    for (int i = 1; i < number; i++)
    {
        if (hash_values[i] == hash_values[i - 1]) count++;
    }
    std::cout << count << std::endl;
}