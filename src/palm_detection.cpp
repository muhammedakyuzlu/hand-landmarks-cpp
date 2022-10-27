#include "palm_detection.h"
#include <cmath>

#include "anchors.cpp"


/* -------------------------------------------------- *
 *  Decode palm detection result
 * -------------------------------------------------- */

int PALM::decode_keypoints (std::list<palm_t> &palm_list, float score_thresh){
    palm_t palm_item;
    float *scores_ptr = (float *)pOutputTensor_bbox;
    float *points_ptr = (float *)pOutputTensor_prob;
    int img_w = _in_width;
    int img_h = _in_height;
    int i = 0;
    for (auto itr = s_anchors.begin(); itr != s_anchors.end(); i ++, itr ++)
    {
        Anchor anchor = *itr;
        float score0 = scores_ptr[i];
        float score = 1.0f / (1.0f + exp(-score0));

        std::cout << score0 << std::endl;
        std::cout << score << std::endl;
        std::cout << "*******" << std::endl;

        if (score > score_thresh)
        {
            float *p = points_ptr + (i * 18);

            /* boundary box */
            float sx = p[0];
            float sy = p[1];
            float w  = p[2];
            float h  = p[3];

            float cx = sx + anchor.x_center * img_w;
            float cy = sy + anchor.y_center * img_h;

            cx /= (float)img_w;
            cy /= (float)img_h;
            w  /= (float)img_w;
            h  /= (float)img_h;

            fvec2 topleft, btmright;
            topleft.x  = cx - w * 0.5f;
            topleft.y  = cy - h * 0.5f;
            btmright.x = cx + w * 0.5f;
            btmright.y = cy + h * 0.5f;

            palm_item.score         = score;
            palm_item.rect.topleft  = topleft;
            palm_item.rect.btmright = btmright;

            /* landmark positions (7 keys) */
            for (int j = 0; j < 7; j ++)
            {
                float lx = p[4 + (2 * j) + 0];
                float ly = p[4 + (2 * j) + 1];
                lx += anchor.x_center * img_w;
                ly += anchor.y_center * img_h;
                lx /= (float)img_w;
                ly /= (float)img_h;

                palm_item.keys[j].x = lx;
                palm_item.keys[j].y = ly;
            }

            palm_list.push_back (palm_item);
        }
    }
    return 0;
}

void PALM::loadModel(const std::string model_path)
{

    _model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    if (!_model)
    {
        std::cout << "\nFailed to load the model.\n" << std::endl;
        exit(1);
    }
    else
    {
        std::cout << "\nModel loaded successfully.\n";
    }
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*_model, resolver)(&_interpreter);
    TfLiteStatus status = _interpreter->AllocateTensors();
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
    _input = _interpreter->inputs()[0];
    TfLiteIntArray *dims = _interpreter->tensor(_input)->dims;
    _in_height = dims->data[1];
    _in_width = dims->data[2];
    _in_channels = dims->data[3];
    _in_type = _interpreter->tensor(_input)->type;
    _input_ = _interpreter->typed_tensor<float_t>(_input);
    _interpreter->SetNumThreads(nthreads);

    
    generate_ssd_anchors ();   
}

void PALM::print_2d_tensor(std::vector<std::vector<float>> v){
    for ( int i =0 ; i<v.size();i++){
        for( int j=0 ; j<4;j++){
            std::cout << v[i][j] << " ";
        }
        std::cout << std::endl;
    }
}


void PALM::preprocess(cv::Mat &image)
{
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    cv::resize(image, image, cv::Size(_in_height, _in_width), cv::INTER_CUBIC);
    image.convertTo(image,  CV_32F); //   1/255.0
    image -= 128.0f;
    image /= 128.0f;
    // image /= 256.0f;

        
}

template <typename T>
void PALM::fill(T *in, cv::Mat &src)
{
    int n = 0, nc = src.channels(), ne = src.elemSize();
    for (int y = 0; y < src.rows; ++y)
        for (int x = 0; x < src.cols; ++x)
            for (int c = 0; c < nc; ++c)
                in[n++] = src.data[y * src.step + x * ne + c];
}

std::vector<std::vector<float_t>> PALM::tensorToVector2D(TfLiteTensor *pOutputTensor, const int &row, const int &colum)
{
    std::vector<std::vector<float_t>> v;
    for (int i = 0; i < row; i++)
    {
        std::vector<float_t> _tem;
        for (int j = 0; j < colum; j++)
        {
            float_t val_float = (float_t)pOutputTensor->data.f[i * colum + j];
            _tem.push_back(val_float);
        }
        v.push_back(_tem);
    }
    return v;
}

std::vector<float_t> PALM::tensorToVector1D(TfLiteTensor *pOutputTensor, const int &row)
{
    std::vector<float_t> v;
    for (int j = 0; j < row; j++)
    {
        float_t val_float = (float_t)pOutputTensor->data.f[row + j];
        v.push_back(val_float);
    }   
    return v;
}

void _sigm(std::vector<float> &xs){
    for(auto & x : xs){ x = 1 / (1 +  std::exp(-x)); }
}

// void PALM::read_csv(std::string path,std::vector<std::vector<float>> &anchors){
//   io::CSVReader<4> in(path);
//   float x1,x2,x3,x4;
//   while(in.read_row(x1,x2,x3,x4)){
//     std::vector<float> anchor;
//     anchor.push_back(x1);
//     anchor.push_back(x2);
//     anchor.push_back(x3);
//     anchor.push_back(x4);
//     anchors.push_back(anchor);
//     anchor.clear();
//   }
// }

// void PALM::nonMaximumSuppression(std::vector<std::vector<float>> bbbox, std::vector<float> proberit){

// // out_reg shape is [number of anchors, 18]
// // Second dimension 0 - 4 are bounding box offset, width and height: dx, dy, w ,h
// // Second dimension 4 - 18 are 7 hand key point x and y coordinates: x1,y1,x2,y2,...x7,y7

// }


/* -------------------------------------------------- *
 *  Apply NonMaxSuppression:
 * -------------------------------------------------- */
static float
calc_intersection_over_union (rect_t &rect0, rect_t &rect1)
{
    float sx0 = rect0.topleft.x;
    float sy0 = rect0.topleft.y;
    float ex0 = rect0.btmright.x;
    float ey0 = rect0.btmright.y;
    float sx1 = rect1.topleft.x;
    float sy1 = rect1.topleft.y;
    float ex1 = rect1.btmright.x;
    float ey1 = rect1.btmright.y;
    
    float xmin0 = std::min (sx0, ex0);
    float ymin0 = std::min (sy0, ey0);
    float xmax0 = std::max (sx0, ex0);
    float ymax0 = std::max (sy0, ey0);
    float xmin1 = std::min (sx1, ex1);
    float ymin1 = std::min (sy1, ey1);
    float xmax1 = std::max (sx1, ex1);
    float ymax1 = std::max (sy1, ey1);
    
    float area0 = (ymax0 - ymin0) * (xmax0 - xmin0);
    float area1 = (ymax1 - ymin1) * (xmax1 - xmin1);
    if (area0 <= 0 || area1 <= 0)
        return 0.0f;

    float intersect_xmin = std::max (xmin0, xmin1);
    float intersect_ymin = std::max (ymin0, ymin1);
    float intersect_xmax = std::min (xmax0, xmax1);
    float intersect_ymax = std::min (ymax0, ymax1);

    float intersect_area = std::max (intersect_ymax - intersect_ymin, 0.0f) *
                           std::max (intersect_xmax - intersect_xmin, 0.0f);
    
    return intersect_area / (area0 + area1 - intersect_area);
}
static bool
compare (palm_t &v1, palm_t &v2)
{
    if (v1.score > v2.score)
        return true;
    else
        return false;
}
static int
non_max_suppression (std::list<palm_t> &face_list, std::list<palm_t> &face_sel_list, float iou_thresh)
{
    face_list.sort (compare);

    for (auto itr = face_list.begin(); itr != face_list.end(); itr ++)
    {
        palm_t face_candidate = *itr;

        int ignore_candidate = false;
        for (auto itr_sel = face_sel_list.rbegin(); itr_sel != face_sel_list.rend(); itr_sel ++)
        {
            palm_t face_sel = *itr_sel;

            float iou = calc_intersection_over_union (face_candidate.rect, face_sel.rect);
            if (iou >= iou_thresh)
            {
                ignore_candidate = true;
                break;
            }
        }

        if (!ignore_candidate)
        {
            face_sel_list.push_back(face_candidate);
            if (face_sel_list.size() >= 4)
                break;
        }
    }

    return 0;
}

/* -------------------------------------------------- *
 *  Expand palm to hand
 * -------------------------------------------------- */
static float
normalize_radians (float angle)
{
    return angle - 2 * M_PI * std::floor((angle - (-M_PI)) / (2 * M_PI));
}

static void
compute_rotation (palm_t &palm)
{
    float x0 = palm.keys[0].x;  // Center of wrist.
    float y0 = palm.keys[0].y;
    float x1 = palm.keys[2].x;  // MCP of middle finger.
    float y1 = palm.keys[2].y;

    float target_angle = M_PI * 0.5f;
    float rotation = target_angle - std::atan2(-(y1 - y0), x1 - x0);
    
    palm.rotation = normalize_radians (rotation);
}
static void
rot_vec (fvec2 &vec, float rotation)
{
    float sx = vec.x;
    float sy = vec.y;
    vec.x = sx * std::cos(rotation) - sy * std::sin(rotation);
    vec.y = sx * std::sin(rotation) + sy * std::cos(rotation);
}

static void
compute_hand_rect (palm_t &palm)
{
    float width    = palm.rect.btmright.x - palm.rect.topleft.x;
    float height   = palm.rect.btmright.y - palm.rect.topleft.y;
    float palm_cx  = palm.rect.topleft.x + width  * 0.5f;
    float palm_cy  = palm.rect.topleft.y + height * 0.5f;
    float hand_cx;
    float hand_cy;
    float rotation = palm.rotation;
    float shift_x =  0.0f;
    float shift_y = -0.5f;
    
    if (rotation == 0.0f)
    {
        hand_cx = palm_cx + (width  * shift_x);
        hand_cy = palm_cy + (height * shift_y);
    }
    else
    {
        float dx = (width  * shift_x) * std::cos(rotation) -
                   (height * shift_y) * std::sin(rotation);
        float dy = (width  * shift_x) * std::sin(rotation) +
                   (height * shift_y) * std::cos(rotation);
        hand_cx = palm_cx + dx;
        hand_cy = palm_cy + dy;
    }

    float long_side = std::max (width, height);
    width  = long_side;
    height = long_side;
    float hand_w = width  * 2.6f;
    float hand_h = height * 2.6f;

    palm.hand_cx = hand_cx;
    palm.hand_cy = hand_cy;
    palm.hand_w  = hand_w;
    palm.hand_h  = hand_h;

    float dx = hand_w * 0.5f;
    float dy = hand_h * 0.5f;

    palm.hand_pos[0].x = - dx;  palm.hand_pos[0].y = - dy;
    palm.hand_pos[1].x = + dx;  palm.hand_pos[1].y = - dy;
    palm.hand_pos[2].x = + dx;  palm.hand_pos[2].y = + dy;
    palm.hand_pos[3].x = - dx;  palm.hand_pos[3].y = + dy;

    for (int i = 0; i < 4; i ++)
    {
        rot_vec (palm.hand_pos[i], rotation);
        palm.hand_pos[i].x += hand_cx;
        palm.hand_pos[i].y += hand_cy;
    }
}
static void pack_palm_result(palm_detection_result_t *palm_result, std::list<palm_t> &palm_list)
{
    int num_palms = 0;
    for (auto itr = palm_list.begin(); itr != palm_list.end(); itr ++)
    {
        palm_t palm = *itr;
        
        compute_rotation (palm);
        compute_hand_rect (palm);

        memcpy (&palm_result->palms[num_palms], &palm, sizeof (palm));
        num_palms ++;
        palm_result->num = num_palms;

        if (num_palms >= MAX_PALM_NUM)
            break;
    }
}

void PALM::run(cv::Mat frame, std::vector<cv::Rect> &bboxes)
{

    _img_height = frame.rows;
    _img_width = frame.cols;
    
    preprocess(frame);
    fill(_input_, frame);
    

    // Inference
    TfLiteStatus status = _interpreter->Invoke();
    if (status != kTfLiteOk)
    {
        std::cout << "\nFailed to run inference!!\n";
        exit(1);
    }



    // out_reg shape is [number of anchors, 18]
    // Second dimension 0 - 4 are bounding box offset, width and height: dx, dy, w ,h
    // Second dimension 4 - 18 are 7 hand keypoint x and y coordinates: x1,y1,x2,y2,...x7,y7

    int _out_bbox   = _interpreter->outputs()[0];
    TfLiteIntArray *_out_dims_bbox   = _interpreter->tensor(_out_bbox)->dims;
    // --> 1 2016 18

    // it is the classification score if there is a hand for each anchor box
    int _out_prob = _interpreter->outputs()[1];
    TfLiteIntArray *_out_dims_prob = _interpreter->tensor(_out_prob)->dims;
    // --> 1 2016 0
    
    // std::cout <<_out_dims_bbox->data[0]<<" "<<_out_dims_bbox->data[1] <<" "<<_out_dims_bbox->data[2] << std::endl;
    // std::cout <<_out_dims_prob->data[0]<<" "<<_out_dims_prob->data[1] <<" "<<_out_dims_prob->data[2] << std::endl;

    pOutputTensor_bbox = _interpreter->typed_tensor<float>(_interpreter->outputs()[0]);
    pOutputTensor_prob = _interpreter->typed_tensor<float>(_interpreter->outputs()[1]);
    
    // pOutputTensor_bbox = _interpreter->tensor(_interpreter->outputs()[0]);
    // pOutputTensor_prob = _interpreter->tensor(_interpreter->outputs()[1]);

    // pOutputTensor_bbox = _interpreter->tensor(_interpreter->typed_tensor()[0]);
    // pOutputTensor_prob = _interpreter->tensor(_interpreter->typed_tensor()[1]);

    std::list<palm_t> palm_list;
    decode_keypoints (palm_list, confThreshold);

    // for(auto p : palm_list){
    //     std::cout << p.rect.btmright.x << std::endl;
    // }

    std::list<palm_t> palm_nms_list;
    non_max_suppression (palm_list, palm_nms_list, nmsThreshold);
    

    std::cout << "NMS = "<<palm_nms_list.size() << std::endl; 
  

    palm_detection_result_t palm_result = {0};
    pack_palm_result (&palm_result, palm_nms_list);

    for(int i =0 ; i < palm_result.num;  i++){
        cv::Point pTopLeft;
        cv:: Point pBottomRight;
        pTopLeft.x = palm_result.palms->rect.topleft.x;
        pTopLeft.y = palm_result.palms->rect.topleft.y;
        pBottomRight.x = palm_result.palms->rect.btmright.x;
        pBottomRight.y = palm_result.palms->rect.btmright.y;
        cv::Rect _r(pTopLeft,pBottomRight);
        bboxes.push_back(_r); 
    }








    // int _out_row_bbox   = _out_dims_bbox->data[1];   
    // int _out_colum_bbox = _out_dims_bbox->data[2]; 
    // std::vector<std::vector<float_t>> pred_bbox = tensorToVector2D(pOutputTensor_bbox, _out_row_bbox, _out_colum_bbox);
    // int _out_row_prob = _out_dims_prob->data[1];
    // std::vector<float_t> pred_prob = tensorToVector1D(pOutputTensor_prob, _out_row_prob);
    

    // // // print_2d_tensor(pred_bbox);
  
    // _sigm(pred_prob);

    // std::vector<float> probabilities;  
    // std::vector<std::vector<float>> candidate_detect;
    // std::vector<Anchor> candidate_anchors;

    // for (int i = 0; i < pred_prob.size(); i++)
    // {
    //     if (pred_prob[i] > 0.5){
    //         candidate_detect.push_back(pred_bbox[i]);
    //         candidate_anchors.push_back(s_anchors[i]);
    //         probabilities.push_back(pred_prob[i]);
    //     }
    // }
    // std::cout << "probabilities = "<< probabilities.size() << std::endl;
    // std::cout << "****************"<< std::endl;


    // if (candidate_detect.size() == 0){
    //     std::cout  << "No hands found" << std::endl;
    //     return;
    // }
    




    // std::vector<cv::Rect> moved_candidate_detect;

    // for ( int i = 0; i<candidate_detect.size();i++){
        
    //     // height--> image.rows,  width--> image.cols;
    //     int left = candidate_detect[i][0] + (candidate_anchors[i].x_center * _img_width);
    //     int top = candidate_detect[i][1] + (candidate_anchors[i].y_center * _img_height);
    //     int w = candidate_detect[i][2] * _img_width;
    //     int h = candidate_detect[i][3] * _img_height;

    //     moved_candidate_detect.push_back(cv::Rect(left, top, w, h));

    // }

    // // print_2d_tensor(moved_candidate_detect);    
    
    // std::vector<int> indices;
    // cv::dnn::NMSBoxes(moved_candidate_detect,probabilities, confThreshold, nmsThreshold, indices);
    
    // // std::cout << indices << std::endl;
    // // std::cout << indices.size() << std::endl;

    // for ( int i =0 ; i<indices.size();i++){
    //     std::cout << indices[i] << std::endl;

        

    // }
    
    // float dx = candidate_detect[0][0];
    // float dy = candidate_detect[0][1];
    // float w = candidate_detect[0][2];
    // float h = candidate_detect[0][3];

    // float cx = candidate_anchors[0][0]*256;
    // float cy = candidate_anchors[0][1]*256;
    // nonMaximumSuppression(pred_bbox,pred_anchor,bboxes);

};