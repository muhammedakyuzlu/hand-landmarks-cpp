#include "palm_detection.h"
#include "hand_landmark_detection.h"
#include <iostream>
#include "image_preprocess.h"


#include <chrono>
using namespace std::chrono;

int main(int argc, char *argv[])
{
    auto start = high_resolution_clock::now();
    const std::string palmModel_path = "/models/palm_detection_lite.tflite";
    const std::string hand_model_path = "/models/hand_landmark_lite.tflite";

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
    // cv::namedWindow("w", 1);
    
    capture.open(0);
    // save video config
    // bool save = false;
    // auto fourcc = capture.get(cv::CAP_PROP_FOURCC);
    int _img_width  = capture.get(cv::CAP_PROP_FRAME_WIDTH);
    int _img_height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
    handLandmarkModel._img_width = _img_width;
    handLandmarkModel._img_height = _img_height;
    // cv::VideoWriter video(save_path, fourcc, 30, cv::Size(frame_width, frame_height), true);

    for (;;)
    {
        capture.read(frame);
        if (frame.empty())
            break;
        
        cv::flip(frame,frame,1);
        cv::Mat  normalizedImg;
        preprocess(frame, normalizedImg);

        palm_detection_result_t palm_result = {0};
        palmModel.run(normalizedImg, palm_result);
        
        if (palm_result.num > 0)
        {
            std::vector<hand_landmark_result_t> hand_results = {};
            handLandmarkModel.run(normalizedImg,hand_results,palm_result);

            cv::Point topLeft;
            cv::Point bottomRight;
            // show palm rectangle
            for (auto palm : palm_result.palms)
            {

                if(palm.score<0.5)
                    continue;               
                // ################### full hand rectangle with orientation   #############
                // output line    
                float x0 = palm.hand_pos[0].x* _img_width;
                float y0 = palm.hand_pos[0].y*_img_height;
                float x1 = palm.hand_pos[1].x* _img_width;
                float y1 = palm.hand_pos[1].y*_img_height;
                float x2 = palm.hand_pos[2].x* _img_width;
                float y2 = palm.hand_pos[2].y*_img_height;
                float x3 = palm.hand_pos[3].x* _img_width;
                float y3 = palm.hand_pos[3].y*_img_height;
                
                 //fillConvexPoly example 1 
                // cv::Point ptss[4]; 
                // ptss[0] = cv::Point(x0, y0);
                // ptss[1] = cv::Point(x1, y1);
                // ptss[2] = cv::Point(x2, y2);
                // ptss[3] = cv::Point(x3, y3);
                // cv::fillConvexPoly(frame, ptss, 4, cv::Scalar(255, 255, 255),1);
                
                std::vector< cv::Point> contour;
                contour.push_back(cv::Point(x0, y0));
                contour.push_back(cv::Point(x1, y1));
                contour.push_back(cv::Point(x2, y2));
                contour.push_back(cv::Point(x3, y3));
                cv::polylines(frame, contour, 4, cv::Scalar(255, 255, 255));


                //  ############### full hand rectangle   #############
                // float small_x = _img_width;
                // float small_y = _img_height;
                // float large_x = 0;
                // float large_y = 0;

                // for(int ll =0 ; ll<4;ll++){
                //     small_x = palm.hand_pos[ll].x < small_x ? palm.hand_pos[ll].x : small_x;
                //     small_y = palm.hand_pos[ll].y < small_y ? palm.hand_pos[ll].y : small_y;
                //     large_x = palm.hand_pos[ll].x > large_x ? palm.hand_pos[ll].x : large_x;
                //     large_y = palm.hand_pos[ll].y > large_y ? palm.hand_pos[ll].y : large_y;
                // }

                // topLeft.x = small_x * _img_width;
                // topLeft.y = small_y * _img_height ;
                // bottomRight.x = large_x * _img_width;
                // bottomRight.y = large_y * _img_height;


                // ######## palm rectangle   #############
                // topLeft.x = palm.rect.topleft.x * _img_width;
                // topLeft.y = palm.rect.topleft.y * _img_height ;
                // bottomRight.x = palm.rect.btmright.x * _img_width;
                // bottomRight.y = palm.rect.btmright.y * _img_height;

                cv::Rect _r(topLeft,bottomRight);
                cv::rectangle(frame,_r,cv::Scalar(255,255,255),1);
            }

            for(auto hand : hand_results){
                // show hand landmark
                for(auto landmark:hand.joint){

                    cv::Point p = cv::Point(landmark.x + hand.offset.x ,landmark.y + hand.offset.y);
                    cv::circle(frame,p,3,cv::Scalar(0,0,255),-1);
                }
            }

        }

        // cv::flip(frame,frame,1);
        cv::imshow("output", frame);
        cv::waitKey(10);
    }
    capture.release();
    cv::destroyAllWindows();
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << duration.count() << std::endl; 
}
