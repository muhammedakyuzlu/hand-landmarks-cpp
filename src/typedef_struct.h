#pragma once
#include <opencv2/imgproc.hpp>

#define MAX_PALM_NUM   4
#define HAND_JOINT_NUM 21

typedef struct fvec2
{
    float x, y;
} fvec2;

typedef struct rect_t
{
    fvec2 topleft;
    fvec2 btmright;
} rect_t;

typedef struct fvec3
{
    float x, y, z;
} fvec3;

typedef struct _hand_landmark_result_t
{
    float score;
    fvec3 joint[HAND_JOINT_NUM];
    fvec2 offset;
} hand_landmark_result_t;

typedef struct _palm_t
{
    // model outputs after decoding
    float  hand_cx;
    float  hand_cy;
    float  hand_w;
    float  hand_h;
    fvec2  keys[7];

    // model outputs 
    float  score;
    
    // palm rectangle
    rect_t rect;

    // hole hand rectangle
    float  rotation;
    fvec2  hand_pos[4];
} palm_t;

typedef struct _palm_detection_result_t
{
    int num;
    palm_t palms[MAX_PALM_NUM];
} palm_detection_result_t;

typedef cv::Point3_<float> Pixel;


