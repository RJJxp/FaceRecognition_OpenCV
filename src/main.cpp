// system
#include <iostream>
// opencv
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "FaceRecognition.h"

// declaration
void detectAndDisplay(cv::Mat frame);
void detectAndShow(cv::Mat frame); // only ROI

// global variable
cv::CascadeClassifier face_cascade;
cv::CascadeClassifier eyes_cascade;
cv::CascadeClassifier nose_cascade;
cv::CascadeClassifier mouth_cascade;

int main(int argc, char* argv[]) {
    std::cout << "This is a test for OpenCV_3.4.8 face recognition" << std::endl;
    std::cout << "OpenCV version : " << CV_VERSION << std::endl;
    std::cout << "Major version : " << CV_MAJOR_VERSION << std::endl;
    std::cout << "Minor version : " << CV_MINOR_VERSION << std::endl;
    std::cout << "Subminor version : " << CV_SUBMINOR_VERSION << std::endl;

    FaceRecognition* fr = new FaceRecognition();
    if (!fr->loadModel("/home/rjp/rjp_code/face/test4opencv_3.4/conf/ModelPath.yaml")) {
        std::cout << "load model failed in main.cpp." << std::endl;
        return -1;
    }

    if (!fr->detectAll("/home/rjp/rjp_code/face/human_face/face_01.jpg")) {
        std::cout << "detect failed." << std::endl;
        return -1;
    }

    fr->showResult();
    return 0;
}

// deprecated
/*
void detectAndShow(cv::Mat frame) {
    std::vector<cv::Rect> faces;
    cv::Mat frame_gray;

    cv::cvtColor(frame, frame_gray, CV_BGR2GRAY);
    cv::equalizeHist(frame_gray, frame_gray);

    // --Detect faces
    std::vector<int> reject_level;
    std::vector<double> level_width;
    face_cascade.detectMultiScale(frame_gray, faces, reject_level, level_width, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(30, 30), cv::Size(), true);
    std::cout << "face size is " << faces.size() << std::endl;
    std::cout << "reject level size is " << reject_level.size() << std::endl;
    std::cout << "reject level is " << reject_level[0] << std::endl;
    std::cout << "reject level width is " << level_width[0] << std::endl;
    
    for( size_t i = 0; i < faces.size(); i++ )
    {
        // cv::Point center(faces[i].x + faces[i].width * 0.5, faces[i].y + faces[i].height * 0.5);
        // cv::ellipse(frame, center, cv::Size( faces[i].width * 0.5, faces[i].height * 0.5), 0, 0, 360, cv::Scalar( 255, 0, 255 ), 4, 8, 0);
        cv::Mat faceROI_2show = frame(faces[i]);
        cv::Mat faceROI = frame_gray(faces[i]);
        std::vector<cv::Rect> eyes;

        // -- In each face, detect eyes
        std::vector<int> eyes_reject_level;
        std::vector<double> eyes_level_wight;
        eyes_cascade.detectMultiScale( faceROI, eyes, eyes_reject_level, eyes_level_wight, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, cv::Size(30, 30), cv::Size(), true);
        std::cout << "eyes: " << eyes.size() << std::endl;
        for( size_t j = 0; j < eyes.size(); j++ )
        {
           cv::Point center(eyes[j].x + eyes[j].width*0.5, eyes[j].y + eyes[j].height*0.5);
           int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25);
           cv::circle(faceROI_2show, center, radius, cv::Scalar(255, 0, 0), 4, 8, 0);
           std::cout << j << "eyes level is " << eyes_reject_level[j] << " weight is " << eyes_level_wight[j] << std::endl;
        }

        // detect nose;
        std::vector<int> nose_reject_level;
        std::vector<double> nose_level_weight;
        std::vector<cv::Rect> nose;
        nose_cascade.detectMultiScale(faceROI, nose, nose_reject_level, nose_level_weight, 1.1, 2 , 0 |CV_HAAR_SCALE_IMAGE, cv::Size(30, 30), cv::Size(), true);
        std::cout << "nose: " << nose.size() << std::endl;
        for (size_t j = 0; j < nose.size(); ++j) {
            cv::Point center(nose[j].x + nose[j].width*0.5, nose[j].y + nose[j].height*0.5);
            int radius = cvRound((nose[j].width + nose[j].height)*0.25);
            cv::circle(faceROI_2show, center, radius, cv::Scalar(255, 255, 0), 4, 8, 0);
            std::cout << j << "nose level is " << nose_reject_level[j] << " weight is " << nose_level_weight[j] << std::endl;
        }

        // detect mouth
        std::vector<cv::Rect> mouth;
        std::vector<int> mouth_reject_level;
        std::vector<double> mouth_level_weight;
        mouth_cascade.detectMultiScale(faceROI, mouth, mouth_reject_level, mouth_level_weight, 1.1, 2 , 0 |CV_HAAR_SCALE_IMAGE, cv::Size(30, 30), cv::Size(), true);
        std::cout << "mouth: " << mouth.size() << std::endl;
        for (size_t j = 0; j < mouth.size(); ++j) {
            cv::Point center(mouth[j].x + mouth[j].width*0.5, mouth[j].y + mouth[j].height*0.5);
            int radius = cvRound( (mouth[j].width + mouth[j].height)*0.25);
            cv::circle(faceROI_2show, center, radius, cv::Scalar(0, 0, 255), 4, 8, 0);
            std::cout << j << "nose level is " << mouth_reject_level[j] << " weight is " << mouth_reject_level[j] << std::endl;
        }
        // show
        cv::imshow("face" + std::to_string(i), faceROI_2show);
    }
    //-- Show what you got
    // cv::imshow( "opencv_34", frame);
}


void detectAndDisplay(cv::Mat frame) {
    std::vector<cv::Rect> faces;
    cv::Mat frame_gray;

    cv::cvtColor(frame, frame_gray, CV_BGR2GRAY);
    cv::equalizeHist(frame_gray, frame_gray);

    //-- Detect faces
    face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(30, 30));

    for( size_t i = 0; i < faces.size(); i++ )
    {
        cv::Point center(faces[i].x + faces[i].width * 0.5, faces[i].y + faces[i].height * 0.5);
        cv::ellipse(frame, center, cv::Size( faces[i].width * 0.5, faces[i].height * 0.5), 0, 0, 360, cv::Scalar( 255, 0, 255 ), 4, 8, 0);

        cv::Mat faceROI = frame_gray(faces[i]);
        std::vector<cv::Rect> eyes;

        // -- In each face, detect eyes
        eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, cv::Size(30, 30) );
        std::cout << "eyes: " << eyes.size() << std::endl;
        for( size_t j = 0; j < eyes.size(); j++ )
        {
           cv::Point center( faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5);
           int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25);
           cv::circle( frame, center, radius, cv::Scalar(255, 0, 0), 4, 8, 0);
        }
        // detect nose;
        std::vector<cv::Rect> nose;
        nose_cascade.detectMultiScale(faceROI, nose, 1.1, 2 , 0 |CV_HAAR_SCALE_IMAGE, cv::Size(30, 30));
        std::cout << "nose: " << nose.size() << std::endl;
        for (size_t j = 0; j < nose.size(); ++j) {
           cv::Point center( faces[i].x + nose[j].x + nose[j].width*0.5, faces[i].y + nose[j].y + nose[j].height*0.5);
           int radius = cvRound( (nose[j].width + nose[j].height)*0.25);
           cv::circle( frame, center, radius, cv::Scalar(255, 255, 0), 4, 8, 0);
        }
        // detect mouth
        std::vector<cv::Rect> mouth;
        mouth_cascade.detectMultiScale(faceROI, mouth, 1.1, 2 , 0 |CV_HAAR_SCALE_IMAGE, cv::Size(30, 30));
        std::cout << "mouth: " << mouth.size() << std::endl;
        for (size_t j = 0; j < mouth.size(); ++j) {
            cv::Point center( faces[i].x + mouth[j].x + mouth[j].width*0.5, faces[i].y + mouth[j].y + mouth[j].height*0.5);
           int radius = cvRound( (mouth[j].width + mouth[j].height)*0.25);
           cv::circle( frame, center, radius, cv::Scalar(0, 0, 255), 4, 8, 0);
        }
    }
    //-- Show what you got
    cv::imshow( "opencv_34", frame);
 }

 */