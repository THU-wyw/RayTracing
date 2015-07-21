#include "bvh.h"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/device_malloc.h>
#include "model.h"
#include "cuda_runtime.h"
#include <thrust/device_free.h>

class CreateMortonCodesFunction
{
private:
    const Model model_;
    __inline__ __host__ __device__ uint ExpandBits(uint v)
    {
        v = (v * 0x00010001u) & 0xFF0000FFu;
        v = (v * 0x00000101u) & 0x0F00F00Fu;
        v = (v * 0x00000011u) & 0xC30C30C3u;
        v = (v * 0x00000005u) & 0x49249249u;
        return v;
    }
public:
    CreateMortonCodesFunction(const Model& model):model_(model) {}
    __host__ __device__ uint operator()(int i)
    {
        float3 center = 0.5 * (fminf(
            fminf(model_.triangles[i].v1,
            model_.triangles[i].v2),
            model_.triangles[i].v3)
            + fmaxf(
            fmaxf(model_.triangles[i].v1,
            model_.triangles[i].v2),
            model_.triangles[i].v3));
        float3 p = (center - model_.global_min) / (model_.global_max - model_.global_min);
        p = clamp(p * 1024.0f, 0.0f, 1023.0f);
        uint xx = ExpandBits((uint)p.x);
        uint yy = ExpandBits((uint)p.y);
        uint zz = ExpandBits((uint)p.z);
        return xx * 4 + yy * 2 + zz;
    }
};

struct MortonCodeComparation
{
    inline __host__ __device__ bool operator() (const Bvh::MortonCode& a,
                                         const Bvh::MortonCode& b)
    {
        return a.code < b.code;
    }
};

class CreateLeavesFunction
{
private:
    int triangle_number_;
    int leaf_number_;
    int* last_index_;
public:
    CreateLeavesFunction(int triangle_number, int leaf_number, int* last_index)
        : triangle_number_(triangle_number),
        leaf_number_(leaf_number),
        last_index_(last_index) {}
    __host__ __device__ Bvh::LeafNode operator()(int index)
    {
        Bvh::LeafNode node;
        if (index > 0 && index < leaf_number_)
        {
            node.low_index = last_index_[index - 1] + 1;
            node.high_index = last_index_[index];
        }
        else if (index == 0)
        {
            node.low_index = 0;
            node.high_index = last_index_[0];
        }
        else
        {
            node.low_index = last_index_[index - 1] + 1;
            node.high_index = triangle_number_ - 1;
        }
        return node;
    }
};

struct MortonIsSingleFunction
{
    inline __host__ __device__ int operator() (uint& a, uint& b)
    {
        return a < b ? 1 : 0;
    }
};

__global__ void CreateInnerNodes(int leaf_number,
                                 const uint* morton_codes,
                                 const int* last_id,
                                 Bvh::InnerNode* inner_nodes,
                                 Bvh::LeafNode* leaf_nodes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= leaf_number - 1) return;
    auto delta = [leaf_number, morton_codes, last_id](int i, int j) -> int
    {
        if (j < 0 || j >= leaf_number) return -1;
        return __clz(morton_codes[last_id[i]]
            ^ morton_codes[last_id[j]]);
    };
    int x = delta(idx, idx - 1);
    int y = delta(idx, idx + 1);
    int d = (x < y) ? 1 : -1;
    // Compute upper bound for the length of the range
    int tmin = min(x,y);
    //we have found that it is beneficial to start from a larger number, e.g. 128, 
    //and multiply the value by 4 instead of 2, after each iteration to reduce the total amount of work.
    int lmax = 2;
    int c = delta(idx, idx + lmax * d);
    while (c > tmin)
    {
        lmax = lmax * 2;
        c = delta(idx, idx + lmax * d);
    }
    // Find the other end using binary search
    int l = 0;
    int k = lmax / 2;
    int e = 0;
    while (k > 0)
    {
        e = delta(idx, idx + (l + k) * d);
        if (e > tmin)
            l = l + k;
        k = k / 2;
    }
    int j = idx + l*d;
    // Find the split position using binary search
    uint tnode = delta(idx, j);
    uint mid, start, end;
    if (d == 1)
    {
        start = idx;
        end = j;
        while (start <= end)
        {
            mid=(start + end)/2;
            if (start == end)
            {
                if(delta(idx, mid) > tnode)
                    start = mid+1;
                else
                    end = --mid;
            }
            else
            {
                if(delta(idx, mid) > tnode)
                {
                    if (start < mid)
                        start = mid;
                    else
                        start = mid+1;
                }
                else
                    end = mid-1;
            }
        }
    }
    else
    {
        start = j;
        end = idx;
        while (start <= end)
        {
            mid = (start + end + 1) / 2;
            if (start == end)
                start = mid + 1;
            else
            {				
                if(delta(mid, idx) > tnode)
                    end = mid - 1;
                else
                    start = mid;
            }
        }
    }
    int split = mid;	
    inner_nodes[idx].left_index = split;
    inner_nodes[idx].right_index = split + 1;
    inner_nodes[idx].left_is_leaf = min(idx, j) == split;
    inner_nodes[idx].right_is_leaf = max(idx, j) == split+1;
    if (inner_nodes[idx].left_is_leaf)
    {
        leaf_nodes[split].father_index = idx;
    }
    else
    {
        inner_nodes[split].father_index = idx;
    }
    if (inner_nodes[idx].right_is_leaf)
    {
        leaf_nodes[split + 1].father_index = idx;
    }
    else
    {
        inner_nodes[split + 1].father_index = idx;
    }

    if (idx == 0)
    {
        inner_nodes[0].father_index = -1;
    }
}

class CreateBoundBoxesFunction
{
private:
    Bvh bvh_;
    Model model_;
    int* visited_;
public:
    CreateBoundBoxesFunction(const Bvh& bvh, const Model& model, int* visited)
        : bvh_(bvh), model_(model), visited_(visited) {}
    __device__ void operator() (int index)
    {
        //从叶子节点开始
        bvh_.leaf_nodes[index].max_bound = make_float3(-1000, -1000, -1000);
        bvh_.leaf_nodes[index].min_bound = make_float3(1000, 1000, 1000);
        for (int i = bvh_.leaf_nodes[index].low_index;
            i <= bvh_.leaf_nodes[index].high_index;
            ++i)
        {
            bvh_.leaf_nodes[index].max_bound = fmaxf(
                fmaxf(bvh_.leaf_nodes[index].max_bound,
                    model_.triangles[bvh_.original_index[i]].v1),
                fmaxf(model_.triangles[bvh_.original_index[i]].v2,
                    model_.triangles[bvh_.original_index[i]].v3));
            bvh_.leaf_nodes[index].min_bound = fminf(
                fminf(bvh_.leaf_nodes[index].min_bound,
                    model_.triangles[bvh_.original_index[i]].v1),
                fminf(model_.triangles[bvh_.original_index[i]].v2,
                    model_.triangles[bvh_.original_index[i]].v3));
        }

        int current = bvh_.leaf_nodes[index].father_index;
        //内部节点

        while ( current != -1 )
        {
            // int atomicCAS(int* address, int compare, int val);
            // (old == compare ? val : old)，该函数将返回 old（比较并交换）;
            if (atomicCAS( &visited_[current], 0, 1) == 0)
                return;
            if (bvh_.inner_nodes[current].left_is_leaf)
            {
                bvh_.inner_nodes[current].min_bound
                    = bvh_.leaf_nodes[bvh_.inner_nodes[current].left_index].min_bound;
                bvh_.inner_nodes[current].max_bound
                    = bvh_.leaf_nodes[bvh_.inner_nodes[current].left_index].max_bound;
            }
            else
            {
                bvh_.inner_nodes[current].min_bound
                    = bvh_.inner_nodes[bvh_.inner_nodes[current].left_index].min_bound;
                bvh_.inner_nodes[current].max_bound
                    = bvh_.inner_nodes[bvh_.inner_nodes[current].left_index].max_bound;
            }

            if (bvh_.inner_nodes[current].right_is_leaf)
            {
                bvh_.inner_nodes[current].min_bound
                    = fminf(bvh_.inner_nodes[current].min_bound,
                    bvh_.leaf_nodes[bvh_.inner_nodes[current].right_index].min_bound);
                bvh_.inner_nodes[current].max_bound
                    = fmaxf(bvh_.inner_nodes[current].max_bound,
                    bvh_.leaf_nodes[bvh_.inner_nodes[current].right_index].max_bound);
            }
            else
            {
                bvh_.inner_nodes[current].min_bound
                    = fminf(bvh_.inner_nodes[current].min_bound,
                    bvh_.inner_nodes[bvh_.inner_nodes[current].right_index].min_bound);
                bvh_.inner_nodes[current].max_bound
                    = fmaxf(bvh_.inner_nodes[current].max_bound,
                    bvh_.inner_nodes[bvh_.inner_nodes[current].right_index].max_bound);
            }
            current = bvh_.inner_nodes[current].father_index;
        }
    }
};

Bvh BuildDeviceBvh(const Model& device_model)
{
    Bvh device_bvh = {nullptr, nullptr, nullptr};
    thrust::device_ptr<uint> devptr_morton_codes
        = thrust::device_malloc<uint>(device_model.triangle_number);
    const uint BLOCK_DIM = 256;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float time = 0.0;
    
    printf("Creating Morton Codes...");
    cudaEventRecord(start, 0);
    uint block_num = device_model.triangle_number / BLOCK_DIM + 1;
    thrust::counting_iterator<int> count(0);
    thrust::transform(count,
        count + device_model.triangle_number,
        devptr_morton_codes,
        CreateMortonCodesFunction(device_model));
    thrust::device_ptr<int> devptr_original_index
        = thrust::device_malloc<int>(device_model.triangle_number);
    thrust::sequence(devptr_original_index, devptr_original_index + device_model.triangle_number, 0);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Done %f ms.\n", time);
    printf("Sorting Morton Codes...");
    cudaEventRecord(start, 0);
    thrust::sort_by_key(devptr_morton_codes,
        devptr_morton_codes + device_model.triangle_number,
        devptr_original_index);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time,start,stop);
    printf("Done %f ms.\n", time);	

    printf("Contracting Morton Codes...");
    cudaEventRecord(start, 0);
    thrust::device_ptr<int> devptr_is_single
        = thrust::device_malloc<int>(device_model.triangle_number);
    thrust::transform(devptr_morton_codes,
        devptr_morton_codes + device_model.triangle_number - 1,
        devptr_morton_codes + 1,
        devptr_is_single,
        MortonIsSingleFunction());
    devptr_is_single[device_model.triangle_number - 1] = 1;
    thrust::device_ptr<int> devptr_is_single_scan
        = thrust::device_malloc<int>(device_model.triangle_number);
    thrust::exclusive_scan(devptr_is_single,
        devptr_is_single + device_model.triangle_number,
        devptr_is_single_scan);
    thrust::device_ptr<int> devptr_lastid
        = thrust::device_malloc<int>(device_model.triangle_number);
    thrust::scatter_if(count,
        count + device_model.triangle_number,
        devptr_is_single_scan,
        devptr_is_single,
        devptr_lastid);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Done %f ms.\n", time);

    printf("Creating Nodes...");
    cudaEventRecord(start, 0);
    int leaf_number = devptr_is_single_scan[device_model.triangle_number - 1] + 1;
    thrust::device_ptr<Bvh::LeafNode> devptr_leaf_nodes
        = thrust::device_malloc<Bvh::LeafNode>(leaf_number);
    thrust::transform(count,
        count + leaf_number,
        devptr_leaf_nodes,
        CreateLeavesFunction(device_model.triangle_number,
        leaf_number,
        devptr_lastid.get()));
    thrust::device_ptr<Bvh::InnerNode> devptr_inner_nodes
        = thrust::device_malloc<Bvh::InnerNode>(leaf_number - 1);
    block_num = leaf_number / BLOCK_DIM + 1;
    CreateInnerNodes<<<block_num, BLOCK_DIM>>>(leaf_number,
        devptr_morton_codes.get(),
        devptr_lastid.get(),
        devptr_inner_nodes.get(),
        devptr_leaf_nodes.get());
    device_bvh.inner_nodes = devptr_inner_nodes.get();
    device_bvh.leaf_nodes = devptr_leaf_nodes.get();
    device_bvh.original_index = devptr_original_index.get();
    device_bvh.leaf_number = leaf_number;
    thrust::device_ptr<int> visited = thrust::device_malloc<int>(leaf_number);
    thrust::fill_n(visited, leaf_number, 0);
    thrust::for_each(count,
        count + leaf_number,
        CreateBoundBoxesFunction(device_bvh, device_model, visited.get()));
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time,start,stop);
    printf("Done %f ms.\n", time);
    std::cout << device_bvh.leaf_number << std::endl;
    thrust::device_free(devptr_morton_codes);
    thrust::device_free(devptr_is_single);
    thrust::device_free(devptr_is_single_scan);
    return device_bvh;
}