#include "test.h"
#include <iostream>
#include <fstream>
#include "bvh.h"

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