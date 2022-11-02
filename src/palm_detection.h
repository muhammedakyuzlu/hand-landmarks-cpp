#pragma once

#include <vector>
#include <list>

#include <tensorflow/lite/kernels/register.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "typedef_struct.h"


class PALM
{
public:
    // Take a model path as string
    void loadModel(const std::string &palm_model_path);
    // Take an image and return a prediction
    void run(const cv::Mat &frame, palm_detection_result_t &palm_result);
    // thresh hold
    float confThreshold = 0.5   ;
    float nmsThreshold  = 0.3;
    // number of threads
    int nthreads = 4;

private:

    // NonMaxSuppression
    float calc_intersection_over_union(rect_t &rect0, rect_t &rect1);
    // bool compare(palm_t &v1, palm_t &v2);
    int non_max_suppression(std::list<palm_t> &face_list, std::list<palm_t> &face_sel_list, float iou_thresh);
   
    // Expand palm to hand
    float normalize_radians(float angle);
    void compute_rotation(palm_t &palm);
    void rot_vec(fvec2 &vec, float rotation);
    void compute_hand_rect(palm_t &palm);
    void pack_palm_result(palm_detection_result_t *palm_result, std::list<palm_t> &palm_list);


    // Decode palm detection result
    int decode_keypoints (std::list<palm_t> &palm_list, float score_thresh);


    // palm model's
    std::unique_ptr<tflite::FlatBufferModel> _palm_model;
    std::unique_ptr<tflite::Interpreter> _palm_interpreter;
    tflite::StderrReporter _palm_error_reporter;

    // models info
    int _palm_input; // model input layer number 
    int _palm_in_height;
    int _palm_in_width;
    int _palm_in_channels;
    int _palm_in_type;

    // Input of the interpreter
    float *_pPalmInputLayer;

    // outputs of the interpreter
    float *_pPalmOutputLayerBbox;
    float *_pPalmOutputLayerProb;

    // int _delegate_opt;
    // TfLiteDelegate *_delegate;
};