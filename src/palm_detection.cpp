#include "palm_detection.h"
#include "anchors.cpp"
#include "image_preprocess.h"

#include <iostream>

/* -------------------------------------------------- *
 *  Load model
 * -------------------------------------------------- */
void PALM::loadModel(const std::string &palm_model_path)
{
 _palm_model = tflite::FlatBufferModel::BuildFromFile(palm_model_path.c_str(), &_palm_error_reporter);
    if (!_palm_model)
    {
        std::cout << "\nFailed to load palm model.\n"
                  << std::endl;
        exit(1);
    }
    tflite::ops::builtin::BuiltinOpResolver palm_resolver;
    tflite::InterpreterBuilder(*_palm_model.get(), palm_resolver)(&_palm_interpreter);
    TfLiteStatus status = _palm_interpreter->AllocateTensors();
    if (status != kTfLiteOk)
    {
        std::cout << "\nFailed to allocate the memory for tensors.\n"
                  << std::endl;
        exit(1);
    }
    else
    {
        std::cout << "\nMemory allocated for tensors.\n";
    }

    // input information
    _palm_input = _palm_interpreter->inputs()[0];
    TfLiteIntArray *dims = _palm_interpreter->tensor(_palm_input)->dims;
    _palm_in_height = dims->data[1];
    _palm_in_width = dims->data[2];
    _palm_in_channels = dims->data[3];
    _palm_in_type = _palm_interpreter->tensor(_palm_input)->type;
    // input layer
    _pPalmInputLayer = _palm_interpreter->typed_tensor<float>(_palm_input);

    // outputs layer
    _pPalmOutputLayerBbox = _palm_interpreter->typed_tensor<float>(_palm_interpreter->outputs()[0]);
    _pPalmOutputLayerProb = _palm_interpreter->typed_tensor<float>(_palm_interpreter->outputs()[1]);

    _palm_interpreter->SetNumThreads(nthreads);

    generate_ssd_anchors();
}

/* -------------------------------------------------- *
 *  Decode palm detection result
 * -------------------------------------------------- */
int PALM::decode_keypoints(std::list<palm_t> &palm_list, float score_thresh)
{

    palm_t palm_item;
    int i = 0;

    for (auto itr = s_anchors.begin(); itr != s_anchors.end(); i++, itr++)
    {
        Anchor anchor = *itr;
        float score0 = _pPalmOutputLayerProb[i];
        float score = 1.0f / (1.0f + exp(-score0));
        if (score > score_thresh)
        {
            float *p = _pPalmOutputLayerBbox + (i * 18);
            /* boundary box */
            float sx = p[0];
            float sy = p[1];
            float w = p[2];
            float h = p[3];

            float cx = sx + anchor.x_center * _palm_in_width;
            float cy = sy + anchor.y_center * _palm_in_height;

            cx /= (float)_palm_in_width;
            cy /= (float)_palm_in_height;
            w /= (float)_palm_in_width;
            h /= (float)_palm_in_height;

            fvec2 topleft, btmright;
            topleft.x = cx - w * 0.5f;
            topleft.y = cy - h * 0.5f;
            btmright.x = cx + w * 0.5f;
            btmright.y = cy + h * 0.5f;

            palm_item.score = score;
            palm_item.rect.topleft = topleft;
            palm_item.rect.btmright = btmright;

            /* landmark positions (7 keys) */
            for (int j = 0; j < 7; j++)
            {
                float lx = p[4 + (2 * j) + 0];
                float ly = p[4 + (2 * j) + 1];
                lx += anchor.x_center * _palm_in_width;
                ly += anchor.y_center * _palm_in_height;
                lx /= (float)_palm_in_width;
                ly /= (float)_palm_in_height;

                palm_item.keys[j].x = lx;
                palm_item.keys[j].y = ly;
            }

            palm_list.push_back(palm_item);
        }
    }
    return 0;
}



/* -------------------------------------------------- *
 *  Apply NonMaxSuppression:
 * -------------------------------------------------- */
float PALM::calc_intersection_over_union(rect_t &rect0, rect_t &rect1)
{
    float sx0 = rect0.topleft.x;
    float sy0 = rect0.topleft.y;
    float ex0 = rect0.btmright.x;
    float ey0 = rect0.btmright.y;
    float sx1 = rect1.topleft.x;
    float sy1 = rect1.topleft.y;
    float ex1 = rect1.btmright.x;
    float ey1 = rect1.btmright.y;

    float xmin0 = std::min(sx0, ex0);
    float ymin0 = std::min(sy0, ey0);
    float xmax0 = std::max(sx0, ex0);
    float ymax0 = std::max(sy0, ey0);
    float xmin1 = std::min(sx1, ex1);
    float ymin1 = std::min(sy1, ey1);
    float xmax1 = std::max(sx1, ex1);
    float ymax1 = std::max(sy1, ey1);

    float area0 = (ymax0 - ymin0) * (xmax0 - xmin0);
    float area1 = (ymax1 - ymin1) * (xmax1 - xmin1);
    if (area0 <= 0 || area1 <= 0)
        return 0.0f;

    float intersect_xmin = std::max(xmin0, xmin1);
    float intersect_ymin = std::max(ymin0, ymin1);
    float intersect_xmax = std::min(xmax0, xmax1);
    float intersect_ymax = std::min(ymax0, ymax1);

    float intersect_area = std::max(intersect_ymax - intersect_ymin, 0.0f) *
                           std::max(intersect_xmax - intersect_xmin, 0.0f);

    return intersect_area / (area0 + area1 - intersect_area);
}

static bool compare(palm_t &v1, palm_t &v2)
{
    if (v1.score > v2.score)
        return true;
    else
        return false;
}

int PALM::non_max_suppression(std::list<palm_t> &face_list, std::list<palm_t> &face_sel_list, float iou_thresh)
{
    face_list.sort(compare);

    for (auto itr = face_list.begin(); itr != face_list.end(); itr++)
    {
        palm_t face_candidate = *itr;

        int ignore_candidate = false;
        for (auto itr_sel = face_sel_list.rbegin(); itr_sel != face_sel_list.rend(); itr_sel++)
        {
            palm_t face_sel = *itr_sel;

            float iou = calc_intersection_over_union(face_candidate.rect, face_sel.rect);
            if (iou >= iou_thresh)
            {
                ignore_candidate = true;
                break;
            }
        }

        if (!ignore_candidate)
        {
            face_sel_list.push_back(face_candidate);
            if (face_sel_list.size() >= MAX_PALM_NUM)
                break;
        }
    }

    return 0;
}

/* -------------------------------------------------- *
 *  Expand palm to hand
 * -------------------------------------------------- */
float PALM::normalize_radians(float angle)
{
    return angle - 2 * M_PI * std::floor((angle - (-M_PI)) / (2 * M_PI));
}

void PALM::compute_rotation(palm_t &palm)
{
    float x0 = palm.keys[0].x; // Center of wrist.
    float y0 = palm.keys[0].y;
    float x1 = palm.keys[2].x; // MCP of middle finger.
    float y1 = palm.keys[2].y;

    float target_angle = M_PI * 0.5f;
    float rotation = target_angle - std::atan2(-(y1 - y0), x1 - x0);

    palm.rotation = normalize_radians(rotation);
}

void PALM::rot_vec(fvec2 &vec, float rotation)
{
    float sx = vec.x;
    float sy = vec.y;
    vec.x = sx * std::cos(rotation) - sy * std::sin(rotation);
    vec.y = sx * std::sin(rotation) + sy * std::cos(rotation);
}

void PALM::compute_hand_rect(palm_t &palm)
{
    float width = palm.rect.btmright.x - palm.rect.topleft.x;
    float height = palm.rect.btmright.y - palm.rect.topleft.y;
    float palm_cx = palm.rect.topleft.x + width * 0.5f;
    float palm_cy = palm.rect.topleft.y + height * 0.5f;
    float hand_cx;
    float hand_cy;
    float rotation = palm.rotation;
    float shift_x = 0.0f;
    float shift_y = -0.5f;

    if (rotation == 0.0f)
    {
        hand_cx = palm_cx + (width * shift_x);
        hand_cy = palm_cy + (height * shift_y);
    }
    else
    {
        float dx = (width * shift_x) * std::cos(rotation) -
                   (height * shift_y) * std::sin(rotation);
        float dy = (width * shift_x) * std::sin(rotation) +
                   (height * shift_y) * std::cos(rotation);
        hand_cx = palm_cx + dx;
        hand_cy = palm_cy + dy;
    }

    float long_side = std::max(width, height);
    width = long_side;
    height = long_side;
    float hand_w = width * 2.6f;
    float hand_h = height * 2.6f;

    palm.hand_cx = hand_cx;
    palm.hand_cy = hand_cy;
    palm.hand_w = hand_w;
    palm.hand_h = hand_h;

    float dx = hand_w * 0.5f;
    float dy = hand_h * 0.5f;

    palm.hand_pos[0].x = -dx;
    palm.hand_pos[0].y = -dy;
    palm.hand_pos[1].x = +dx;
    palm.hand_pos[1].y = -dy;
    palm.hand_pos[2].x = +dx;
    palm.hand_pos[2].y = +dy;
    palm.hand_pos[3].x = -dx;
    palm.hand_pos[3].y = +dy;

    for (int i = 0; i < 4; i++)
    {
        rot_vec(palm.hand_pos[i], rotation);
        palm.hand_pos[i].x += hand_cx;
        palm.hand_pos[i].y += hand_cy;
    }
}

void PALM::pack_palm_result(palm_detection_result_t *palm_result, std::list<palm_t> &palm_list)
{
    int num_palms = 0;
    for (auto itr = palm_list.begin(); itr != palm_list.end(); itr++)
    {
        palm_t palm = *itr;

        compute_rotation(palm);
        compute_hand_rect(palm);

        memcpy(&palm_result->palms[num_palms], &palm, sizeof(palm));
        num_palms++;
        palm_result->num = num_palms;

        if (num_palms >= MAX_PALM_NUM)
            break;
    }
}

/* -------------------------------------------------- *
 *  Main run
 * -------------------------------------------------- */
void PALM::run(const cv::Mat &frame, palm_detection_result_t &palm_result)
{

    cv::Mat palmInputImg;
    // resize image as model input
    cv::resize(frame, palmInputImg, cv::Size(_palm_in_width, _palm_in_height));

    // flatten rgb image to input layer.
    memcpy(_pPalmInputLayer, palmInputImg.ptr<float>(0),
           _palm_in_width * _palm_in_height * _palm_in_channels * sizeof(float));

    // Inference
    TfLiteStatus status = _palm_interpreter->Invoke();
    if (status != kTfLiteOk)
    {
        std::cout << "\nFailed to run inference!!\n";
        exit(1);
    }

    std::list<palm_t> palm_list;
    decode_keypoints(palm_list, confThreshold);

    std::list<palm_t> palm_nms_list;
    non_max_suppression(palm_list, palm_nms_list, nmsThreshold);

    pack_palm_result(&palm_result, palm_nms_list);


}
