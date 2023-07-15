#pragma once

#include <jni.h>
#include <Utils.hpp>
#include <ThreadPool.hpp>
#include <torch/script.h>


// Singleton Instance 
// Java에 CardFinder 인스턴스를 저장 못 하기 때문에
// 매 프레임마다 인스턴스 초기화를 방지하기 위해 싱글톤을 사용
class CardFinder
{
public:
    // 상단, 좌측, 하단, 우측 영상을 구분하는 enum class
    enum class AreaLocation
    {
        TOP,
        LEFT,
        BOTTOM,
        RIGHT
    };
    // Singleton Instance 생성 함수
    static CardFinder& GetInstance(JNIEnv& env, jobject& obj, int& w, int& h);
    // 영상 분석 시작함수
    auto Start(cv::Mat& src) -> std::string;
    // 전체 관심영역 반환 함수
    auto GetCapturedArea() -> cv::Rect&;
    // 체크카드를 포착하여 얻은 네 개의 교점을 반환하는 함수
    auto GetCoordinates() -> std::vector<float>;
    // 소멸자
    virtual ~CardFinder() = default;

private:
    // CardFinder Instance 생성자
    CardFinder(JNIEnv& env, jobject& obj, int& w, int& h);
    // android studio java PreviewView의 roi size를 획득하는 함수
    auto GetImageViewSize(JNIEnv& env, jobject& obj, const char* class_dir) -> cv::Size;
    // 전체 관심영역을 획득하는 함수
    auto SetCapturedArea(int w, int h) -> cv::Rect;
    // 스케일팩터를 계산하는 함수
    auto SetScaleFactor(int w, int h) -> cv::Point2f;
    // 부분관심영역(가로, 세로)을 획득하는 함수
    auto SetPartsOfCapturedArea() -> std::vector<cv::Rect>;
    // 최소자승밝기보정에 사용되는 행렬을 만드는 함수
    auto SetBrightCorrectionModels() -> cv::Mat;
    // 최소자승밝기보정의 결과 영상을 담을 vector<Mat> 타입의 데이터를 만드는 함수
    auto SetBrightCorrectionFields() -> cv::Mat;
    // 푸리에 변환에 사용되는 가우시안 필터를 만드는 함수
    auto SetGaussianFilters(cv::Size size, double D0) -> const std::vector<cv::Mat>;
    // 네 개의 교점에 Scale factor를 곱하는 함수
    auto SetCoordinates(cv::Point2f& pt1, cv::Point2f& pt2, cv::Point2f& pt3, cv::Point2f& pt4) -> void;
    // Contrast Limiting Adaptive Histogram Equalization 객체 생성 함수
    auto SetCLAHE(double limit_var, cv::Size tile_size) -> cv::Ptr<cv::CLAHE>;
    // pytorch 모델을 불러오는 함수
    auto SetTorchModel(JNIEnv& env, jobject& obj, const char* class_dir) -> std::vector<torch::jit::script::Module>;
    // 체크카드의 가로 세로 직선을 얻는 함수
    auto GetLines(cv::Mat& src, AreaLocation arealoc) -> std::vector<cv::Vec4f>;
    // 체크카드의 가로 세로 직선들에서 교점을 구하고 세 점을 이용하여 각도를 얻는 함수
    auto GetCrossPointFromTwoLines(std::vector<cv::Vec4f>& line1, std::vector<cv::Vec4f>& line2)->cv::Point2f;
    // 두 점이 이루는 각도(체크카드의 기울기)를 구하는 함수
    auto GetAngleFromTwoPoints(cv::Point2f pt1, cv::Point2f pt2, AreaLocation arealoc) -> float;
    // 최소자승밝기보정을 수행하는 함수
    auto BrightCorrect(cv::Mat& src) -> cv::Mat&;
    // 푸리에 HomomorhpicFitering을 수행하는 함수
    virtual auto HomomorphicCorrect(cv::Mat& src) -> cv::Mat;
    // 숫자영역의 이미지를 전처리하는 함수
    auto AreaPreProcess(cv::Mat& src, cv::Mat& dst) -> void;
    // 이진영상에서 객체들의 잡음을 없애는 함수
    auto AreaMasking(cv::Mat& src) -> void;
    // 이진영상에서 객체들을 분류하는 함수
    virtual auto AreaSegmant(cv::Mat& src, int offset_width, int offset_height) -> std::vector<cv::Rect>;
    // 숫자영역들만 추려내는 함수
    virtual auto DataClassification(std::vector<cv::Rect>& rois) -> std::vector<cv::Rect>;
    // 숫자영역을 Pytorch Script를 활용하여 인식하는 함수
    auto DataDiscrimination(cv::Mat& src, std::vector<cv::Rect>& areas, torch::jit::script::Module& module, std::map<int, char>& labels) -> std::string;

    // 스레드 갯수
    int m_thread_num;
    // android studio java PreviewView의 roi size
    cv::Size m_image_view_size;
    // 전체 관심영역
    cv::Rect m_captured_area;
    // ImageAnalysis의 roi에서 획득한 교점들의 위치를 보정해주는 스케일팩터
    cv::Point2f m_scale_factor;
    // 전체 관심영역에서 상단, 좌측, 하단, 우측 관심영역을 생성할 때 필요한 네 개의 교점
    std::vector<cv::Point2f> m_cross_pt4;
    // 상단, 좌측, 하단, 우측 관심영역
    std::vector<cv::Rect> m_parts_of_captured_area;
    // 최소자승의 전체영역, 가로, 세로에 해당하는 A 행렬
    cv::Mat m_A;
    // 최소자승 밝기 보정의 결과가 담기는 전체, 가로, 세로, 영역의 Mat 타입 이미지
    cv::Mat m_br_correction_field;
    // 스레드풀
    ThreadPool::ThreadPool m_pool;
    // pytorch 스크립트 모듈
    std::vector<torch::jit::script::Module> m_modules;
    // 이진영상의 침식 및 확장 연산 사용되는 커널
    cv::Mat m_kernel;
    // 히스토그램 평준화에 사용되는 CLAHE 객체
    cv::Ptr<cv::CLAHE> m_clahe;
    // 전체 관심영역으로 포착된 이미지
    std::vector<jfloat> m_res_coordinate;
    // 레이블
    std::map<int, char> m_labels_number, m_labels_alphabet;


protected:
    // SubProcess instance 생성 시 사용되는 생성자
    CardFinder(int& w, int& h);
    // 가우시안 저주파 및 고주파 필터
    std::vector<cv::Mat> m_gaussian_filters;

};


