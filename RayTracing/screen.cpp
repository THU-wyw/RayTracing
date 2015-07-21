#include "screen.h"
#include "vector_types_math.h"
void SetPixels(ScreenParams& params);
void SetLength(ScreenParams& params);

ScreenParams InitScreenParams()
{
    ScreenParams params;
    params.pixel_number = 80;
    params.look_at = make_float3(LookAtX0, LookAtY0, LookAtZ0);
    SetPixels(params);
    SetLength(params);
    return params;
}

void SetPixels(ScreenParams& params)
{
    float3 dir = params.look_at - make_float3(EYEX0, EYEY0, EYEZ0);
    dir = normalize(dir);

    float3 up = {Upx,Upy,Upz};
    float3 u = normalize(cross(up, dir));

    float3 v = cross(dir, u);

    float3 midScr = {EYEX0 + ScrToEye * dir.x, EYEY0 + ScrToEye * dir.y, EYEZ0 + ScrToEye * dir.z};

    float3 downScr;
    int shijiW = 16;
    int shijiH = 9;
    downScr = midScr - shijiH * v;

    params.screen_left_down = downScr - shijiW * u;
    params.screen_right_down = downScr + shijiW * u;

    float3 upScr = midScr + shijiH * v;

    params.screen_left_up = upScr - shijiW * u;
    params.screen_right_up = upScr + shijiW * u;

    params.pixel_right = (params.screen_right_down - params.screen_left_down) / ScrWidth;
    params.pixel_up = (params.screen_left_up - params.screen_left_down) / ScrHeight;
}

void SetLength(ScreenParams& params)
{
    float3 dir = params.look_at - make_float3(EYEX0, EYEY0, EYEZ0);
    dir = normalize(dir);

    params.length_screen_left_down = params.screen_left_down - dir * 0.8;
    params.length_right = (params.screen_right_down - params.screen_left_down) * params.pixel_number / ScrWidth;
    params.length_up = (params.screen_left_up - params.screen_left_down) * params.pixel_number / ScrHeight;
    params.length_screen_start = params.length_screen_left_down;
}
