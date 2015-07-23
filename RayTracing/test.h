#ifndef TEST_H
#define TEST_H

struct Bvh;
struct ScreenParams;
struct Ray;
extern void test_bvh(const Bvh& device_bvh);
extern void test_contract_rays(const Ray* rays, int number, const ScreenParams& params);
#endif