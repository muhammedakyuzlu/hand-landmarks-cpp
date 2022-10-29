#include "palm_detection.h"
#include "hand_landmark_detection.h"
#include <iostream>
#include "image_preprocess.h"

int main(int argc, char *argv[])
{

    const std::string palmModel_path = "models/palm_detection_full.tflite";
    const std::string hand_model_path = "models/hand_landmark_full.tflite";

    PALM palmModel;
    HandLandmark handLandmarkModel;

    // conf palm model
    palmModel.confThreshold = 0.30;
    palmModel.nmsThreshold = 0.40;
    palmModel.nthreads = 4;

    // conf hand landmark model
    handLandmarkModel.confThreshold = 0.30;
    handLandmarkModel.nthreads = 4;

    // Load the saved_model
    palmModel.loadModel(palmModel_path);
    handLandmarkModel.loadModel(hand_model_path);

    cv::VideoCapture capture;
    cv::Mat frame;

    // // load input video or open web cam
    // if (all_of(video_path.begin(), video_path.end(), ::isdigit) == false)
    //     capture.open(video_path);
    // else
    //     capture.open(stoi(video_path));
    // if (!capture.isOpened())
    //     throw "\nError when reading video steam\n";
    // cv::namedWindow("output", 1);
    capture.open(0);

    // save video config
    // bool save = false;
    // auto fourcc = capture.get(cv::CAP_PROP_FOURCC);
    int _img_width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
    int _img_height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
    // cv::VideoWriter video(save_path, fourcc, 30, cv::Size(frame_width, frame_height), true);

    for (;;)
    {

        capture >> frame;
        cv::Mat  normalizedImg;
        preprocess(frame, normalizedImg);

        palm_detection_result_t palm_result = {0};
        palmModel.run(normalizedImg, palm_result);
        if (palm_result.num > 0)
        {
            hand_landmark_result_t hand_result = {0};
            handLandmarkModel.run(normalizedImg,hand_result,palm_result);
 
            // show palm rectangle
            for (auto palm : palm_result.palms)
            {
                cv::Point topLeft;
                cv::Point bottomRight;
                topLeft.x = palm.hand_pos[0].x * _img_width;
                topLeft.y = palm.hand_pos[0].y * _img_height;
                bottomRight.x = topLeft.x + palm.hand_w * _img_width;
                bottomRight.y = topLeft.y + palm.hand_h * _img_height;
                cv::Rect _r(topLeft, bottomRight);
                cv::rectangle(frame, _r, (0, 0, 255), 1);
            }

            // show hand landmark
            for(auto l:hand_result.joint){
                cv::circle(frame,cv::Point(l.x,l.y),5,(255,0,0),1);
            }
        }

        if (frame.empty())
            break;
        
        cv::imshow("output", frame);
        cv::waitKey(10);
    }
    capture.release();
    cv::destroyAllWindows();
}
