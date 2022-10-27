#pragma once

#include <cstdint>
#include <vector>
#include <chrono>
#include <iostream>
#include <fstream>
#include <cmath>
#include <list>

#include <tensorflow/lite/model.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>


#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

#define MAX_PALM_NUM   4

typedef struct fvec2
{
    float x, y;
} fvec2;

typedef struct fvec3
{
    float x, y, z;
} fvec3;

typedef struct rect_t
{
    fvec2 topleft;
    fvec2 btmright;
} rect_t;

typedef struct _palm_t
{
    float  score;
    rect_t rect;
    fvec2  keys[7];
    float  rotation;

    float  hand_cx;
    float  hand_cy;
    float  hand_w;
    float  hand_h;
    fvec2  hand_pos[4];
} palm_t;
typedef struct _palm_detection_result_t
{
    int num;
    palm_t palms[MAX_PALM_NUM];
} palm_detection_result_t;


class PALM
{
public:
    // Take a model path as string
    void loadModel(const  std::string model_path);
    // Take an image and return a prediction
    void run(cv::Mat image, std::vector<cv::Rect> &bboxes);

    // thresh hold
    float confThreshold = 0.5;
    float nmsThreshold = 0.5;

    // number of threads
    int nthreads = 4;

    void preprocess(cv::Mat &image);


    // model's
    std::unique_ptr<tflite::FlatBufferModel> _model;
    std::unique_ptr<tflite::Interpreter> _interpreter;
    tflite::StderrReporter _error_reporter;
    // TfLiteTensor *pOutputTensor_bbox;
    // TfLiteTensor *pOutputTensor_prob;
    float *pOutputTensor_bbox;
    float *pOutputTensor_prob;
    // parameters of interpreter's input
    int _input;
    int _in_height;
    int _in_width;
    int _in_channels;
    int _in_type;

    // parameters of original image
    int _img_height;
    int _img_width;

    // Input of the interpreter
    float_t *_input_;

    // int _delegate_opt;
    // TfLiteDelegate *_delegate;
    // std::vector<std::vector<float>> anchors;

    template <typename T>
    void fill(T *in, cv::Mat &src);
    int decode_keypoints (std::list<palm_t> &palm_list, float score_thresh);
    std::vector<std::vector<float_t>> tensorToVector2D(TfLiteTensor *pOutputTensor, const int &row, const int &colum);
    std::vector<float_t>              tensorToVector1D(TfLiteTensor *pOutputTensor, const int &row);
    void nonMaximumSuppression(std::vector<std::vector<float>> pred_bbox, std::vector<float> pred_anchor, std::vector<cv::Rect> &bboxes);
    // void read_csv(std::string path,std::vector<std::vector<float>> &anchors);

    void print_2d_tensor(std::vector<std::vector<float>> v);
};