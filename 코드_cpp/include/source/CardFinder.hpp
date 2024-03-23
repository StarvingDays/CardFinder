#pragma once

#include <jni.h>
#include <Utils.hpp>
#include <queue>
#include <vector>
#include <thread>
#include <mutex>

// Singleton Instance
// Java에 CardFinder 인스턴스를 저장 못 하기 때문에
// 매 프레임마다 인스턴스 초기화를 방지하기 위해 싱글톤을 사용
class CardFinder
{
public:
    using Lines = std::vector<cv::Vec4f>;
    using Rects = std::vector<cv::Rect>;
    using RectCornerPoints = std::tuple<cv::Point2f, cv::Point2f, cv::Point2f, cv::Point2f>;
    // 상단, 좌측, 하단, 우측 영상을 구분하는 enum class
    enum class AreaLocation
    {
        TOP,
        LEFT,
        BOTTOM,
        RIGHT
    };

    // Singleton Instance 생성 함수
    static CardFinder& GetInstance(JNIEnv& env, jobject& obj, int row, int col);
    // 이미지 프록시 사이
    auto GetImageProxySize() -> cv::Size&;
    // 안드로이드 스튜디오 java PreviewView의 관심영역의 크기를 획득하는 함수
    auto GetImageViewSize(JNIEnv& env, jobject& obj, const char* class_dir) -> cv::Size;
    // 프록시 이미지 사이즈를 획득하는 함수
    auto SetImageProxySize(int row, int col) -> cv::Size;
    // 전체 관심영역을 획득하는 함수
    auto SetCapturedArea(int row, int col) -> cv::Rect;
    // 스케일팩터를 계산하는 함수
    auto SetScaleFactor(int row, int col) -> cv::Point2f;
    // 부분관심영역(가로, 세로)을 획득하는 함수
    auto SetPartsOfCapturedArea() -> Rects;
    // 최소자승밝기보정에 사용되는 행렬을 만드는 함수
    auto SetBrightCorrectionModel() -> cv::Mat;
    // 최소자승밝기보정의 결과 영상을 담을 vector<Mat> 타입의 데이터를 만드는 함수
    auto SetBrightCorrectionField() -> cv::Mat;
    // 푸리에 변환에 사용되는 가우시안 필터를 만드는 함수
    auto SetGaussianFilters(cv::Size size, double D0) -> std::vector<cv::Mat>;
    // Contrast Limiting Adaptive Histogram Equalization 객체 생성 함수
    auto SetCLAHE(double limit_var, cv::Size tile_size) -> cv::Ptr<cv::CLAHE>;
    // 소멸자
    ~CardFinder();


private:
    // CardFinder Instance 생성자
    CardFinder(JNIEnv& env, jobject& obj, int& row, int& col);
    // CardFinder Process 실행 함수
    auto RunCardFindProcess(std::vector<uchar> img_buffer) -> void;
    // 체크카드 모서리 교점 네 개를 획득하는 함수
    auto FindRectCorner(cv::Mat& img) -> RectCornerPoints;
    // 직선이 기울어진 각도를 구하는 함수
    auto FindCosAngleFromLine(cv::Point2f pt1, cv::Point2f pt2, AreaLocation areaLoc) -> float;
    // 각도 만큼 영상을 회전시키는 함수
    auto RotateByAngle(cv::Mat& img, float ang) -> cv::Mat;
    // 이미지 전처리 함수
    auto RunImagePreProcess(cv::Mat& img) -> cv::Mat;
    // 체크카드의 가로 세로 직선을 얻는 함수
    auto FindLines(cv::Mat& src, AreaLocation arealoc) -> Lines;
    // 체크카드의 모서리 지점을 찾는 함수
    auto FindRightAngleCorner(Lines& lines1, Lines& lines2) -> cv::Point2f;
    // 최소자승밝기보정을 수행하는 함수
    auto BrightCorrect(cv::Mat& src) -> cv::Mat&;
    // 푸리에 HomomorhpicFitering을 수행하는 함수
    auto HomomorphicCorrect(cv::Mat& src, cv::Mat& filter) -> cv::Mat;


private:
    // android studio java analize 함수에서 받아오는 ImageProxy 인스턴스의 가로 세로 사이즈
    cv::Size m_image_view_size;
    // android studio java PreviewView의 roi size
    cv::Size m_image_proxy_size;
    // 전체 관심영역
    cv::Rect m_captured_area;
    // ImageAnalysis의 roi에서 획득한 교점들의 위치를 보정해주는 스케일팩터
    cv::Point2f m_scale_factor;
    // 전체 관심영역에서 우측 및 하단 영역의 시작지점
    cv::Point m_start_pt_of_right_area, m_start_pt_of_bottom_area;
    // 상단, 좌측, 하단, 우측 관심영역
    Rects m_parts_of_captured_area;
    // 최소자승의 전체영역, 가로, 세로에 해당하는 A 행렬
    cv::Mat m_A;
    // 최소자승 밝기 보정의 결과가 담기는 전체, 가로, 세로, 영역의 Mat 타입 이미지
    cv::Mat m_br_correction_field;
    // 히스토그램 평준화에 사용되는 CLAHE 객체
    std::shared_ptr<cv::CLAHE> m_clahe;
    // 가우시안 저주파 및 고주파 필터
    std::vector<cv::Mat> m_gaussian_filters;
    // pulling 스레드 내부의 while 루프용 atomic<bool>
    std::atomic_bool m_thread_pool_on;
    // 스레드풀
    std::vector<std::thread> m_thread_pool;

public:
    // 이미지 전처리 작업 큐
    std::queue<std::vector<uchar>> m_image_data_queue;
    // 전체 관심영역으로 포착된 이미지
    std::vector<jfloat> m_res_coordinate;
    // json 포멧의 response 패킷의 body 메세지가 저장되는 string 객체
    std::vector<uchar> m_checkcard_image_buffer;
    // 스레드 임계영역 처리용 mutex
    std::mutex m_thread_pool_mutex, m_card_find_process_mutex;
    // 스레드 대기용 객체
    std::condition_variable m_conv;
};
