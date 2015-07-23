#include "traversal.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <device_functions.h>
#include <limits.h>
#include "model.h"
#include "bvh.h"
#include "screen.h"
#include "stack.h"
#define BlockSize      256  
#define BlockDimX      256  
#define StackSize      64

struct NodeStack
{
    int nodeIdx;
    bool isleaf;
};

template<typename T>
__device__ __inline__ void swap(T* a, T* b)
{
    T t = *a;
    *a = *b;
    *b = t;
}

__device__ __inline__ float max3f(float3 a)
{
    return fmaxf(fmaxf(a.x, a.y), a.z);
}
__device__ __inline__ float min3f(float3 a)
{
    return fminf(fminf(a.x, a.y), a.z);
}

__device__ __inline__ float intersect(const float3& orig, const float3& dir, const Model& model, int triangle_id)
{
    float3 edge1 = model.triangles[triangle_id].v2
        - model.triangles[triangle_id].v1;
    float3 edge2 = model.triangles[triangle_id].v3
        - model.triangles[triangle_id].v1;

    float3 pvec = cross(dir, edge2);
    float det = dot(edge1, pvec);
    if(det > -EPSILON && det < EPSILON)
        return -1;

    float inv_det = 1.0 / det;
    float3 tvec = orig - model.triangles[triangle_id].v1;

    float3 qvec = cross(tvec, edge1);
    float u = dot(tvec, pvec) * inv_det;
    if(u < 0.0 || u > 1.0)
        return -1;

    float v = dot(dir,qvec) * inv_det;
    if(v < 0.0 || u + v > 1.0)
        return -1;

    return (dot(edge2, qvec) * inv_det );
}

__device__ thrust::pair<int, float> traversal(const Bvh& bvh, const Model& model, const float3& dir, const float3& orig)
{
    float3 idir;
    idir.x = 1.0f / (fabsf(dir.x) > EPSILON ? dir.x : ((dir.x>=0) ? EPSILON : -EPSILON));
    idir.y = 1.0f / (fabsf(dir.y) > EPSILON ? dir.y : ((dir.y>=0) ? EPSILON : -EPSILON));
    idir.z = 1.0f / (fabsf(dir.z) > EPSILON ? dir.z : ((dir.z>=0) ? EPSILON : -EPSILON));

    int stackPtr = -1;   
    int currentNode = 0;
    bool currentIsLeaf = false;
    NodeStack bvhStack[StackSize];	// Thrown into local memory by PTX, as slow as the global memory;

    int hit = -1;	
    float t = 1000.0;


    while (stackPtr < StackSize)
    {
        if (currentIsLeaf)
        {
            float pt = intersect(orig, dir, model, bvh.original_index[currentNode]);
            if (pt > 0 && pt < t)
            {
                t = pt;
                hit = bvh.original_index[currentNode];
            }
            // pop
            if ( stackPtr == -1 ) 
                break;
            currentNode = bvhStack[stackPtr].nodeIdx;
            currentIsLeaf = bvhStack[stackPtr].isleaf;
            --stackPtr;		
        }
        else
        {
            int left_node = bvh.inner_nodes[currentNode].left_index;
            int right_node = bvh.inner_nodes[currentNode].right_index;
            bool left_is_leaf = bvh.inner_nodes[currentNode].left_is_leaf;
            bool right_is_leaf = bvh.inner_nodes[currentNode].right_is_leaf;
            float3 c0low, c0high, c1low, c1high;
            //        printf("%d %d\n", inner_nodes_[current_node].left_is_leaf, inner_nodes_[current_node].right_is_leaf);
            c0low = left_is_leaf ? bvh.leaf_nodes[left_node].min_bound
                : bvh.inner_nodes[left_node].min_bound;
            c0high = left_is_leaf ? bvh.leaf_nodes[left_node].max_bound
                : bvh.inner_nodes[left_node].max_bound;
            c1low = right_is_leaf ? bvh.leaf_nodes[right_node].min_bound
                : bvh.inner_nodes[right_node].min_bound;
            c1high = right_is_leaf ? bvh.leaf_nodes[right_node].max_bound
                : bvh.inner_nodes[right_node].max_bound;
            c0low = (c0low - orig) * idir;
            c0high = (c0high - orig) * idir;
            c1low = (c1low - orig) * idir;
            c1high = (c1high - orig) * idir;
            float c0min = max3f(fminf(c0low, c0high));
            float c0max = min3f(fmaxf(c0low, c0high));
            float c1min = max3f(fminf(c1low, c1high));
            float c1max = min3f(fmaxf(c1low, c1high));
            bool traverseChild0 = (c0max >= c0min) && (t >= c0min);   
            bool traverseChild1 = (c1max >= c1min) && (t >= c1min); 
            currentNode = left_node;
            currentIsLeaf = left_is_leaf;

            // Condition 1: intersect with only one child.

            if( traverseChild0 != traverseChild1 )		
            {
                if (traverseChild1)
                {
                    currentNode = right_node;
                    currentIsLeaf = right_is_leaf;
                }
            }
            else
            {
                // Condition 2: both intersected, push the farther child into stack and deal with the closer one.
                if ( traverseChild0 ) 
                {
                    if( c1min < c0min )   // go right    
                    {	
                        swap(&left_node, &right_node);
                        swap(&left_is_leaf, &right_is_leaf);				
                    }
                    ++stackPtr;
                    currentNode = left_node;
                    currentIsLeaf = left_is_leaf;
                    bvhStack[stackPtr].nodeIdx = right_node;
                    bvhStack[stackPtr].isleaf = right_is_leaf;
                }
                // Condition 3: none intersected;s
                else
                {
                    // pop;
                    if ( stackPtr == -1 ) // This ray is not intersected with the whole scene.
                        break;
                    currentNode = bvhStack[stackPtr].nodeIdx;
                    currentIsLeaf = bvhStack[stackPtr].isleaf;
                    --stackPtr;
                }
            }			
        }
    }
    return thrust::make_pair(hit, t);
}

__global__ void GetColorOnePixel(const ScreenParams params,
                                 const Model model,
                                 const Bvh bvh,
                                 float *vcolor){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  

    if (idx < ScrHeight * ScrWidth)
    {
        int row = idx / ScrWidth;  
        int col = idx % ScrWidth; 
        int pixelsNum = 80;
        int lenRow = row / pixelsNum;
        int lenCol = col / pixelsNum;
        float3 orig = params.length_screen_start + lenCol * params.length_right + lenRow * params.length_up;
        float3 xdir = { LookAtX0 - EYEX0, LookAtY0 - EYEY0, LookAtZ0 - EYEZ0 };
        xdir = normalize(xdir);

        float3 rayorig = orig - xdir;
        float3 p0 = params.screen_left_down + col * params.pixel_right + row * params.pixel_up;
        float3 dir = normalize(p0 - rayorig);
        float t = 0;
        int k;
        thrust::tie(k, t) = traversal(bvh, model, dir, rayorig); 

        if (k >= 0) // get an intersected triangle 
        {	
            vcolor[idx] = KAmbient;	

            float3 V = t * dir;
            float3 pintersect = rayorig + V; //pintersect交点
            float3 light = make_float3(Lightx, Lighty, Lightz);
            float3 L = normalize(light - pintersect); //p表示交点，l表示光源，向量L是p->l
            V = -dir;
            float3 AB = model.triangles[k].v2
            - model.triangles[k].v1;
            float3 AC = model.triangles[k].v3
            - model.triangles[k].v1;
            float3 N = normalize(cross(AB, AC)); //三角形ABC,向量AB->  向量AC->  叉乘 右手准则，向量N垂直于三角形ABC			
            if (dot(N, dir) >= 0)
            {
                N = -N;
            }
            float diffuse = (dot(N, L) > 0) ? dot(N, L) : 0; // diffuse reflection;	
            float3 R = 2 * diffuse * N - L;
            R = normalize(R);											
            float specular = pow(dot(R, V), 20); // specular reflection;
            if (0) // 1: shadow; 0: without shadow;
            {
                float3 lorig = pintersect + 0.01 * L;
                int rebound;
                thrust::tie(rebound, t) = traversal(bvh, model, L, lorig); 

                if ( rebound == -1 || t > length(light - lorig))	
                {		
                    vcolor[idx] += KDiffuse * diffuse + KSpecular * specular;		
                }
                else
                { 
                    vcolor[idx] += KDiffuse * diffuse * 0.2;
                }
            } 
            else
            {
                vcolor[idx] += KDiffuse * diffuse + KSpecular * specular;		
            } 
        }
        else
        {
            vcolor[idx] = -1.0f;
        }
    }
}

cudaError_t GetColorAllPixels(const ScreenParams& params,
                              const Model& model,
                              const Bvh& bvh,
                              bool *vhit,
                              float *vcolor)
{	
    cudaError_t cudaStatus;
    int row;
    float time = 0.0;
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float* dev_color = NULL;
    cudaMalloc((void**)&dev_color, sizeof(float) * ScrHeight * ScrWidth);
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!\n");
    }

    int blocks = (ScrWidth * ScrHeight + BlockSize - 1) / BlockSize;  
    dim3 threads(BlockDimX, BlockSize / BlockDimX, 1);  
    printf("cuda traversal...");
    cudaEventRecord(start, 0);
    GetColorOnePixel<<<blocks, threads>>>(params, model, bvh, dev_color);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Done. %f ms\n", time);
    cudaMemcpy(vcolor, dev_color, sizeof(float) * ScrHeight * ScrWidth, cudaMemcpyDeviceToHost);
    std::cout << "Tracing speed: " << ScrHeight*ScrWidth/(time*1000) << " MRays/s" << std::endl;

    int ccnt = 0;
    for(int i = 0; i < ScrHeight * ScrWidth; i++)
    {
        if (vhit[i]) ++ccnt;
    }
    std::cout << ccnt << "Effective Tracing speed: " << ccnt / (time * 1000) << " MRays/s"<< std::endl;
    cudaFree(dev_color); 

    return cudaStatus;
}
