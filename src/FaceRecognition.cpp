#include "FaceRecognition.h"

// cpp
#include <cmath>

// yaml
#include "yaml-cpp/yaml.h"

FaceRecognition::FaceRecognition() {

}

FaceRecognition::~FaceRecognition() {

}

int FaceRecognition::loadModel(std::string path) {
    // read path in config
    YAML::Node config = YAML::LoadFile(path);
    std::string face_model_path = config["FaceModelPath"].as<std::string>();
    std::string eyes_model_path = config["EyesModelPath"].as<std::string>();
    std::string nose_model_path = config["NoseModelPath"].as<std::string>();
    std::string mouth_model_path = config["MouthModelPaht"].as<std::string>();

    std::cout << face_model_path << std::endl;
    std::cout << eyes_model_path << std::endl;
    std::cout << nose_model_path << std::endl;
    std::cout << mouth_model_path << std::endl;

    if (!_face_cascade.load(face_model_path)) {
        std::cout << "loading face model failed." << std::endl;
        return 0;
    }

    if (!_eyes_cascade.load(eyes_model_path)) {
        std::cout << "loading eyes model failed." << std::endl;
        return 0;
    }

    if (!_nose_cascade.load(nose_model_path)) {
        std::cout << "loading nose model failed." << std::endl;
        return 0;
    }

    if (!_mouth_cascade.load(mouth_model_path)) {
        std::cout << "loading mouth model failed." << std::endl;
        return 0;
    }

    std::cout << "load all model sucessfully." << std::endl;
    return 1;
}

void FaceRecognition::detectAndShow(std::string path) {
    cv::Mat frame = cv::imread(path);
    if (frame.empty()) {
        std::cout << "load image failed." << std::endl;
        return;
    }

    std::vector<cv::Rect> faces;
    cv::Mat frame_gray;

    cv::cvtColor(frame, frame_gray, CV_BGR2GRAY);
    cv::equalizeHist(frame_gray, frame_gray);

    // --Detect faces
    std::vector<int> reject_level;
    std::vector<double> level_width;
    _face_cascade.detectMultiScale(frame_gray, faces, reject_level, level_width, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(30, 30), cv::Size(), true);
    std::cout << "face size is " << faces.size() << std::endl;
    std::cout << "reject level size is " << reject_level.size() << std::endl;
    std::cout << "reject level is " << reject_level[0] << std::endl;
    std::cout << "reject level width is " << level_width[0] << std::endl;
    
    for( size_t i = 0; i < 1; i++ )
    {
        // cv::Point center(faces[i].x + faces[i].width * 0.5, faces[i].y + faces[i].height * 0.5);
        // cv::ellipse(frame, center, cv::Size( faces[i].width * 0.5, faces[i].height * 0.5), 0, 0, 360, cv::Scalar( 255, 0, 255 ), 4, 8, 0);
        cv::Mat faceROI_2show = frame(faces[i]);
        cv::Mat faceROI = frame_gray(faces[i]);
        _roi_img = faceROI;
        std::vector<cv::Rect> eyes;

        // -- In each face, detect eyes
        std::vector<int> eyes_reject_level;
        std::vector<double> eyes_level_wight;
        _eyes_cascade.detectMultiScale( faceROI, eyes, eyes_reject_level, eyes_level_wight, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, cv::Size(30, 30), cv::Size(), true);
        std::cout << "eyes: " << eyes.size() << std::endl;
        for( size_t j = 0; j < eyes.size(); j++ )
        {
           cv::Point center(eyes[j].x + eyes[j].width*0.5, eyes[j].y + eyes[j].height*0.5);
           int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25);
           cv::circle(faceROI_2show, center, radius, cv::Scalar(255, 0, 0), 4, 8, 0);
           std::cout << j << "\t eyes level is " << eyes_reject_level[j] << " weight is " << eyes_level_wight[j] << std::endl;
        }

        // detect nose;
        std::vector<int> nose_reject_level;
        std::vector<double> nose_level_weight;
        std::vector<cv::Rect> nose;
        _nose_cascade.detectMultiScale(faceROI, nose, nose_reject_level, nose_level_weight, 1.1, 2 , 0 |CV_HAAR_SCALE_IMAGE, cv::Size(30, 30), cv::Size(), true);
        std::cout << "nose: " << nose.size() << std::endl;
        for (size_t j = 0; j < nose.size(); ++j) {
            cv::Point center(nose[j].x + nose[j].width*0.5, nose[j].y + nose[j].height*0.5);
            int radius = cvRound((nose[j].width + nose[j].height)*0.25);
            cv::circle(faceROI_2show, center, radius, cv::Scalar(255, 255, 0), 4, 8, 0);
            std::cout << j << "\t nose level is " << nose_reject_level[j] << " weight is " << nose_level_weight[j] << std::endl;
        }

        // detect mouth
        std::vector<cv::Rect> mouth;
        std::vector<int> mouth_reject_level;
        std::vector<double> mouth_level_weight;
        _mouth_cascade.detectMultiScale(faceROI, mouth, mouth_reject_level, mouth_level_weight, 1.1, 2 , 0 |CV_HAAR_SCALE_IMAGE, cv::Size(30, 30), cv::Size(), true);
        std::cout << "mouth: " << mouth.size() << std::endl;
        for (size_t j = 0; j < mouth.size(); ++j) {
            cv::Point center(mouth[j].x + mouth[j].width*0.5, mouth[j].y + mouth[j].height*0.5);
            int radius = cvRound( (mouth[j].width + mouth[j].height)*0.25);
            cv::circle(faceROI_2show, center, radius, cv::Scalar(0, 0, 255), 4, 8, 0);
            std::cout << j << "\t nose level is " << mouth_reject_level[j] << " weight is " << mouth_reject_level[j] << std::endl;
        }
        // show
        cv::imshow("face\t" + std::to_string(i), faceROI_2show);
        cv::waitKey(0);
    }
}

int FaceRecognition::getPreciseEyes(const std::vector<cv::Rect>& eyes_rects) {
    int eyes_size = eyes_rects.size();
    switch (eyes_size) {
    case 1: {
        cv::Point eye_01_center(eyes_rects[0].x + eyes_rects[0].width * 0.5, 
                                eyes_rects[0].y + eyes_rects[1].height * 0.5);
        if (eye_01_center.y < _roi_img.rows * 0.5) {    // make sure they are on the top of human face
            _eye_01 = eyes_rects[0];
            return 1;
        } else {
            std::cout << "eyes detection failed." << std::endl;
            return 0;
        }
        break;
    }
    case 2: {
        cv::Point eye_01_center(eyes_rects[0].x + eyes_rects[0].width * 0.5, 
                                eyes_rects[0].y + eyes_rects[1].height * 0.5);
        cv::Point eye_02_center(eyes_rects[1].x + eyes_rects[1].width * 0.5, 
                                eyes_rects[1].y + eyes_rects[1].height * 0.5);
        int diff_y = abs(eye_01_center.y - eye_02_center.y);    // check whether the 2 eyes are level
        int center_y = (eye_01_center.y + eye_02_center.y) / 2; // make sure they are on the top of human face
        if (diff_y < _roi_img.rows * 0.1 && center_y < _roi_img.rows * 0.5) {
            _eye_01 = eyes_rects[0];
            _eye_02 = eyes_rects[1];
            return 1;
        } else {
            std::cout << "eyes detection failed." << std::endl;
            return 0;
        }
        break;
    }
    default: {
        // TODO: if eyes detection is more than 3, fuck'em
        return 0;
        break;
    }   // swithc---default
    }   // switch
}

int FaceRecognition::getPreciseNose(const std::vector<cv::Rect>& nose_rects) {

}

int FaceRecognition::getPreciseMouth(const std::vector<cv::Rect>& mouth_rects) {

}