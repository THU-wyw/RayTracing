#include "model.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cuda_runtime.h>
#include <thrust/device_malloc.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
void ReadPlyHead(FILE* file, Model* model);

Model ReadModel(const char* filename)
{
    Model model;
    FILE* input = fopen(filename, "r");
    clock_t start = clock();
    printf("Reading model...");
    ReadPlyHead(input, &model);
    model.global_min = make_float3(1000.0f);
    model.global_max = make_float3(-1000.0f);
    model.points = new float3[model.point_number];
    model.triangles = new Triangle[model.triangle_number];
    for (uint i = 0; i < model.point_number; ++i)
    {
        fscanf(input,
            "%f %f %f",
            &(model.points[i].x),
            &(model.points[i].y),
            &(model.points[i].z));
        model.global_min = fminf(model.points[i], model.global_min);
        model.global_max = fmaxf(model.points[i], model.global_max);
    }
    int n;
    for (uint i = 0; i < model.triangle_number; ++i)
    {
        fscanf(input,
            "%d %d %d %d",
            &n,
            &(model.triangles[i].indexA),
            &(model.triangles[i].indexB),
            &(model.triangles[i].indexC));
        if (n != 3)
        {
            printf("Polygons except triangles are not supported.\n");
            exit(1);
        }
    }
    clock_t end = clock();
    printf("Done %d ms.\n", end - start);
    return model;
}

void ReadPlyHead(FILE* file, Model* model)
{

    char ply_head[200];
    do 
    {
        fgets(ply_head, 200, file);
        ply_head[strlen(ply_head) - 1] = 0;
        char * sub_string = NULL;
        sub_string = strtok(ply_head, " ");
        if (sub_string != NULL)
        {
            if (strcmp(sub_string, "element") == 0)
            {
                sub_string = strtok(NULL, " ");
                if (strcmp(sub_string, "vertex") == 0)
                {
                    sub_string = strtok(NULL, " ");
                    model->point_number = static_cast<uint>(atoi(sub_string));
                }
                else if (strcmp(sub_string, "face") == 0)
                {
                    sub_string = strtok(NULL, " ");
                    model->triangle_number = static_cast<uint>(atoi(sub_string));
                }
            }
        }
        else {
            printf("Illegal input format\n");
            exit(1);
        }
    } while (strcmp(ply_head, "end_header") != 0);
}

Model GetDeviceCopy(const Model& host_model)
{
    Model device_model = host_model;
    cudaMalloc((void**)(&device_model.points), sizeof(float3) * device_model.point_number);
    cudaMalloc((void**)&device_model.triangles, sizeof(Triangle) * device_model.triangle_number);
    cudaError_t err = cudaMemcpy(device_model.points,
        host_model.points,
        sizeof(float3) * device_model.point_number,
        cudaMemcpyHostToDevice);
    cudaMemcpy(device_model.triangles,
        host_model.triangles,
        sizeof(Triangle) * device_model.triangle_number,
        cudaMemcpyHostToDevice);
    return device_model;
}