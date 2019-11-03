#ifndef FR_H
#define FR_H

// cpp
#include <iostream>

// opencv
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"


// Recognize the human frontal face when no huge rotation 
// e.g.
// FaceRecognition* fr = new FaceRecognition();
// fr->loadModel(".../conf.yaml");
// fr->detectAndShow(".../pic.jpg");
class FaceRecognition {
public:
    FaceRecognition();
    int loadModel(std::string path);    // using YAML to load model
    void detectAndShow(std::string path);   // detect human face and show ROI
    ~FaceRecognition();

private:
    // there may be a lot of false recognition
    // so dthese functions are to judge the results.
    int getPreciseEyes(const std::vector<cv::Rect>& eyes_rects);
    int getPreciseNose(const std::vector<cv::Rect>& nose_rects);
    int getPreciseMouth(const std::vector<cv::Rect>& mouth_rects);
private:
    cv::CascadeClassifier _face_cascade;
    cv::CascadeClassifier _eyes_cascade;
    cv::CascadeClassifier _nose_cascade;
    cv::CascadeClassifier _mouth_cascade;

    cv::Rect _eye_01, _eye_02;
    cv::Rect _nose, _mouth;
    cv::Mat _roi_img;
};

#endif