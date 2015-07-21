#ifndef SCREEN_H
#define SCREEN_H

#include <cuda_runtime.h>
#define ScrWidth	2560
#define ScrHeight	1440
#define WinSizeW	2560
#define WinSizeH	1440
#define EPSILON     0.000001

#define AmbientRed	0.3
#define AmbientGrn	0.3
#define AmbientBlu	1.0

#define KAmbient	0.05
#define KDiffuse	0.5
#define KSpecular	0.4
#define KSecondray	0.1

#define Lightx		5
#define Lighty		29
#define Lightz		5

#define EYEX0		-8
#define EYEY0		19
#define EYEZ0		20

#define LookAtX0	0
#define LookAtY0	12
#define LookAtZ0	-3

#define Upx			0
#define Upy			1
#define Upz			0

#define ScrToEye	10

struct ScreenParams
{
    int pixel_number;
    float3 pixel_right;
    float3 pixel_up;
    float3 screen_left_down, screen_right_down, screen_right_up, screen_left_up;
    float3 length_right;
    float3 length_up;
    float3 length_screen_left_down, length_screen_start;
    float3 look_at;
};

extern ScreenParams InitScreenParams();

#endif