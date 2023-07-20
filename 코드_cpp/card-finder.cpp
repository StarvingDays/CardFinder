#include <CardFinder.hpp>
#include "SubProcess.hpp"


CardFinder::CardFinder(int& w, int& h)
        :
        m_pool(0),
        m_gaussian_filters(SetGaussianFilters(cv::Size(w, h), 30))
{

}

CardFinder::CardFinder(JNIEnv& env, jobject& obj, int& w, int& h)
        :
        m_thread_num(GetThreadInfo() / 2),                                                                           // 스레드 개수
        m_image_view_size(GetImageViewSize(env, obj, "com/sjlee/cardfinder/ViewActivity")),                          // 미리보기 관심영역 사이즈
        m_captured_area(SetCapturedArea(w, h)),                                                                      // 전체 영역
        m_scale_factor(SetScaleFactor(w, h)),                                                                        // 관심영역에 놓일 교점의 위치비율을 보정하는 스케일 팩터
        m_parts_of_captured_area(SetPartsOfCapturedArea()),                                                          // 부분영역(가로, 세로)
        m_A(SetBrightCorrectionModels()),                                                                            // 최소자승 A행렬들(전체영역, 가로영역, 세로영역)
        m_br_correction_field(SetBrightCorrectionFields()),                                                          // 최소자승 보정으로 인한 결과 영상을 저장하는 영상(전체영역, 가로영역, 세로영역)
        m_pool(m_thread_num),                                                                                        // 스레드풀
        m_modules(SetTorchModel(env, obj, "com/sjlee/cardfinder/ViewActivity")),                                     // 파이토치 모델
        m_kernel(cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3))),                                        // 침식 및 확장 연산시 사용되는 커널
        m_clahe(SetCLAHE(4.0, cv::Size(8, 8))),                                                                      // 명암 보정에 쓰이는 CLAHE 객체
        m_gaussian_filters(SetGaussianFilters(m_captured_area.size(), 30))                                           // 가우시안 low, high 필터
{
    m_labels_number = {                                                                                              // 숫자 레이블
            {0, '0'}, {1, '1'},
            {2, '2'}, {3, '3'}, {4, '4'}, {5, '5'},
            {6, '6'}, {7, '7'}, {8, '8'}, {9, '9'}
    };
    m_labels_alphabet = {                                                                                            // 알파벳 레이블
            {0, 'A'}, {1, 'B'}, {2, 'C'}, {3, 'D'},
            {4, 'E'}, {5, 'F'}, {6, 'G'}, {7, 'H'},
            {8, 'I'}, {9, 'J'}, {10, 'K'}, {11, 'L'},
            {12, 'N'}, {13, 'M'}, {14, 'O'}, {15, 'P'},
            {16, 'Q'}, {17, 'R'},{18, 'S'}, {19, 'T'},
            {20, 'U'}, {21, 'V'}, {22, 'W'}, {23, 'X'},
            {24, 'Y'}, {25, 'Z'}
    };
}

CardFinder& CardFinder::GetInstance(JNIEnv& env, jobject& obj, int& w, int& h)
{
    static CardFinder singleton(env, obj, w, h);                                                                     // 싱글톤 인스턴스 생성

    return singleton;
}


auto CardFinder::GetImageViewSize(JNIEnv& env, jobject& obj, const char* class_dir) -> cv::Size
{
    jclass cls = env.FindClass(class_dir);                                                                           // Java 클래스 탐색

    jmethodID method_id =                                                                                            // ViewActivity class에서 GetFileDir 함수의 ID 획득
            env.GetMethodID(cls, "GetImageViewSize", "()[I");

    jintArray int_arr = static_cast<jintArray>(env.CallObjectMethod(obj, method_id));                                // GetImageViewSize함수로 반환된 결과값 획득

    jint* jint_ptr = env.GetIntArrayElements(int_arr, NULL);                                                         // int type 배열의 원소값 획득

    cv::Size data(jint_ptr[0], jint_ptr[1]);                                                                         // ImageView 사이즈를 저장하는 객체 생성

    env.ReleaseIntArrayElements(int_arr, jint_ptr, 0);                                                               // 자원할당 해제

    return data;
}


auto CardFinder::SetCapturedArea(int w, int h) -> cv::Rect
{
    int roi_width = w * 0.25f;                                                                                       // 전체 가로 길이에서 0.25를 곱한 길이
    int roi_height = roi_width * 1.618f;                                                                             // 전체 세로 길이에서 1.618을 곱한 길이
    int left = (w - roi_width) / 2;                                                                                  // 관심영역이 놓일 x축
    int top = (h - roi_height) / 2;                                                                                  // 관심영역이 놓일 y축

    return cv::Rect(left, top, roi_width, roi_height);
}

auto CardFinder::SetScaleFactor(int w, int h) -> cv::Point2f
{
    CV_Assert(m_image_view_size.width != 0 && m_image_view_size.height != 0);
    CV_Assert(m_captured_area.width != 0 && m_captured_area.height != 0);

    return cv::Point2f(                                                                                              // Java의 ImageView의 사이즈와 이미지분석 화면의 사이즈를 나눈 스케일 팩터를 반환
            static_cast<float>(m_image_view_size.width) / static_cast<float>(m_captured_area.height),
            static_cast<float>(m_image_view_size.height) / static_cast<float>(m_captured_area.width));
}


auto CardFinder::SetPartsOfCapturedArea() -> std::vector<cv::Rect>
{
    cv::Point pt1(
            0, (m_captured_area.height - 1) * 0.1f);
    cv::Point pt2(
            m_captured_area.width - 1, (m_captured_area.height - 1) * 0.1f);

    cv::Point pt3(
            (m_captured_area.width - 1) * 0.1f, 0);
    cv::Point pt4(
            (m_captured_area.width - 1) * 0.1f, m_captured_area.height - 1);

    cv::Point pt5(
            0, (m_captured_area.height - 1) * 0.9f);
    cv::Point pt6(
            m_captured_area.width - 1, (m_captured_area.height - 1) * 0.9f);

    cv::Point pt7(
            (m_captured_area.width - 1) * 0.9f, 0);
    cv::Point pt8(
            (m_captured_area.width - 1) * 0.9f, m_captured_area.height - 1);

    m_cross_pt4.push_back(GetCrossPointFromPT4(pt1, pt2, pt3, pt4));                                                 // 교점1
    m_cross_pt4.push_back(GetCrossPointFromPT4(pt1, pt2, pt7, pt8));                                                 // 교점2
    m_cross_pt4.push_back(GetCrossPointFromPT4(pt3, pt4, pt5, pt6));                                                 // 교점3
    m_cross_pt4.push_back(GetCrossPointFromPT4(pt5, pt6, pt7, pt8));                                                 // 교점4

    return std::vector<cv::Rect>() = {
            cv::Rect(                                                                                                // 상단 관심영역
                    cv::Point(m_cross_pt4[0].x + 1, 0),
                    cv::Point(m_cross_pt4[1].x - 1, m_cross_pt4[1].y)),
            cv::Rect(                                                                                                // 좌측 관심영역
                    cv::Point(0, 0),
                    cv::Point(m_cross_pt4[2].x, m_captured_area.height - 1)),
            cv::Rect(                                                                                                // 하단 관심영역
                    cv::Point(m_cross_pt4[2].x + 1, m_cross_pt4[2].y),
                    cv::Point(m_cross_pt4[3].x - 1, m_captured_area.height - 1)),
            cv::Rect(                                                                                                // 우측 관심영역
                    cv::Point(m_cross_pt4[1].x, 0),
                    cv::Point(m_captured_area.width - 1, m_captured_area.height - 1))
    };

}

auto CardFinder::SetBrightCorrectionModels() -> cv::Mat
{
    CV_Assert(m_captured_area.width != 0 && m_captured_area.height != 0);

    int n = 0, total = m_captured_area.width * m_captured_area.height;
    cv::Mat A = cv::Mat::zeros(total, 10, CV_32FC1);                                                                 // (가로*세로)행, 10열 A행렬 생성


    for (int y = 0; y < m_captured_area.height; y++)                                                                 // A 행렬에 원소값 삽입
    {
        for (int x = 0; x < m_captured_area.width; x++)
        {
            A.ptr<float>(n)[0] = x * x * x;
            A.ptr<float>(n)[1] = x * x * y;
            A.ptr<float>(n)[2] = x * y * y;
            A.ptr<float>(n)[3] = y * y * y;
            A.ptr<float>(n)[4] = x * x;
            A.ptr<float>(n)[5] = x * y;
            A.ptr<float>(n)[6] = y * y;
            A.ptr<float>(n)[7] = x;
            A.ptr<float>(n)[8] = y;
            A.ptr<float>(n)[9] = 1;

            ++n;
        }
    }


    return A;
}

auto CardFinder::SetBrightCorrectionFields() -> cv::Mat
{
    CV_Assert(m_captured_area.height != 0 && m_captured_area.width != 0);

    return cv::Mat::zeros(                                                                                           // 최소자승 밝기 보정의 결과가 들어가는 zero 행렬 반환
            m_captured_area.height, m_captured_area.width, CV_8UC1);

}

auto CardFinder::SetGaussianFilters(cv::Size size, double D0) -> const std::vector<cv::Mat>
{
    std::vector<cv::Mat> filter = {
            cv::Mat::zeros(size.height, size.width, CV_32FC1),
            cv::Mat::zeros(size.height, size.width, CV_32FC1) };

    int u, v;
    double D;
    double H;                                                                                                        // High값 원소
    double L;                                                                                                        // Low값 원소
    double centerU = size.width / 2;                                                                                 // mask의 가로 중심
    double centerV = size.height / 2;                                                                                // mask의 세로 중심

    for (v = 0; v < size.height; v++)
    {
        for (u = 0; u < size.width; u++)
        {
            D = sqrt(pow(u - centerU, 2) + pow(v - centerV, 2));                                                     // 가우시안 필터의 지름
            L = exp((-1 * pow(D, 2)) / (2.0 * pow(D0, 2)));
            H = 1 - L;

            filter[0].ptr<float>(v)[u] = L;
            filter[1].ptr<float>(v)[u] = H;
        }
    }
    Shift(filter[0]);                                                                                                // 가운데에 위치한 저주파 필터를 1,2,3,4 분면의 끝부분으로 이동시킴
    Shift(filter[1]);                                                                                                // 가운데에 위치한 고주파 필터를 1,2,3,4 분면의 끝부분으로 이동시킴

    return filter;
}

auto CardFinder::SetCoordinates(cv::Point2f& pt1, cv::Point2f& pt2, cv::Point2f& pt3, cv::Point2f& pt4) -> void
{
    m_res_coordinate = {                                                                                             // 관심영역에 위치한 체크카드의 교점 네 곳이 ImageView에서의 위치비율과
            pt1.x * m_scale_factor.x, pt1.y * m_scale_factor.y,                                                      // 호환되도록 교점의 x,y축에 스케일 팩터를 곱한다
            pt2.x * m_scale_factor.x, pt2.y * m_scale_factor.y,
            pt3.x * m_scale_factor.x, pt3.y * m_scale_factor.y,
            pt4.x * m_scale_factor.x, pt4.y * m_scale_factor.y,
    };
}

auto CardFinder::SetCLAHE(double limit_var, cv::Size tile_size) -> cv::Ptr<cv::CLAHE>
{
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();                                                                    // CLAHE 객체 생성

    clahe->setClipLimit(limit_var);                                                                                  // 리미트 값 설정
    clahe->setTilesGridSize(tile_size);                                                                              // 그리드 사이즈 설정


    return std::move(clahe);
}

auto CardFinder::SetTorchModel(JNIEnv& env, jobject& obj, const char* class_dir)
-> std::vector<torch::jit::script::Module>
{
    jclass cls = env.FindClass(class_dir);                                                                           // Java class 탐색

    jmethodID method_id =                                                                                            // ViewActivity class에서 GetFileDir 함수의 ID를 얻는다
            env.GetMethodID(cls, "GetFileDir", "()Ljava/lang/String;");

    jstring path_jstring =
            static_cast<jstring>(env.CallObjectMethod(obj, method_id));                                              // GetFileDir 함수 호출하여 jstring 타입의 결과를 얻는다


    const char* path_char = env.GetStringUTFChars(path_jstring, NULL);                                               // jstring에서 const char* 타입으로 변환한다

    std::string filename_number = path_char;                                                                         // 숫자 모델위치 경로
    std::string filename_alphabet = path_char;                                                                       // 알파벳 모델위치 경로

    filename_number += "/number_model.pt";                                                                           // 숫자 모델 이름 추가
    filename_alphabet += "/alphabet_model.pt";                                                                       // 알파벳 모델 이름 추가

    torch::jit::script::Module module_number = torch::jit::load(filename_number);                                    // 숫자 모델 불러오기
    torch::jit::script::Module module_alphabet = torch::jit::load(filename_alphabet);                                // 알파벳 모델 불러오기

    module_number.eval();                                                                                            // 모델을 평가 모드로 전환
    module_alphabet.eval();

    env.ReleaseStringUTFChars(path_jstring, path_char);                                                              // 자원할당 해제

    return std::vector<torch::jit::script::Module>() = { module_number, module_alphabet };
}

auto CardFinder::GetLines(cv::Mat& src, AreaLocation arealoc) -> std::vector<cv::Vec4f> {
    std::vector<cv::Vec4f> linesP, resultP;                                                                          // 발견된 직선을 저장하는 객체

    cv::HoughLinesP(src, linesP, 1, CV_PI / 180, 50, 50, 5);                                                         // 영상에서 직선을 검출하는 OpenCV 함수

    bool col_is_big = src.cols > src.rows;                                                                           // 가로 세로의 길이 비교

    size_t line_size = linesP.size();                                                                                // 검출된 직선들의 개수

    if (line_size <= 20)
    {
        for (size_t j = 0; j < line_size; ++j)
        {
            cv::Point2f pt1(linesP[j][0], linesP[j][1]);                                                             // 직선의 시작지점
            cv::Point2f pt2(linesP[j][2], linesP[j][3]);                                                             // 직선의 끝지점

            float m = (pt2.y - pt1.y) / (pt2.x - pt1.x);                                                             // x절편
            float n = ((-1.0f * m) * pt1.x) + pt1.y;                                                                 // y절편

            if (col_is_big == true)                                                                                  // src의 cols 길이가 rows 길이보다 클 경우
            {
                float x1 = pt1.x - src.cols;                                                                         // 시작점 x축 보정
                float y1 = (m * x1) + n;                                                                             // 시작점 y축 계산

                float x2 = pt2.x + src.cols;                                                                         // 끝지점 x축 보정
                float y2 = (m * x2) + n;                                                                             // 끝지점 y축 계산

                if (arealoc == AreaLocation::TOP)                                                                    // 상단 영역
                {
                    resultP.push_back(cv::Vec4f(x1, y1, x2, y2));
                }
                else if (arealoc == AreaLocation::BOTTOM)                                                            // 하단 영역
                {
                    resultP.push_back(
                            cv::Vec4f(x1, y1 + m_cross_pt4[2].y, x2, y2 + m_cross_pt4[2].y));
                }
            }
            else if (col_is_big == false)                                                                            // src의 cols 길이가 rows 길이보다 작을 경우
            {

                float y1 = pt1.y - src.rows;
                float x1 = (y1 + (-1.0f * n)) / m;

                float y2 = pt2.y + src.rows;
                float x2 = (y2 + (-1.0f * n)) / m;

                if (arealoc == AreaLocation::LEFT)                                                                   // 좌측 영역
                {
                    resultP.push_back(cv::Vec4f(x1, y1, x2, y2));
                }
                else if (arealoc == AreaLocation::RIGHT)                                                             // 우측 영역
                {
                    resultP.push_back(
                            cv::Vec4f(x1 + m_cross_pt4[1].x, y1, x2 + m_cross_pt4[1].x, y2));
                }
            }
        }
    }

    return resultP;
}

auto CardFinder::GetCrossPointFromTwoLines(
        Lines& line1,
        Lines& line2) -> cv::Point2f
{
    cv::Point2f res_pt;
    if (line1.size() != 0 && line2.size() != 0)
    {
        size_t size_col_mn = line1.size();
        size_t size_row_mn = line2.size();

        for (int i = 0; i < size_col_mn; ++i)
        {
            cv::Point2f pt1(line1[i][0], line1[i][1]);                                                               // 직선1 시작지점
            cv::Point2f pt2(line1[i][2], line1[i][3]);                                                               // 직선1 끝지점

            for (int j = 0; j < size_row_mn; ++j)
            {
                cv::Point2f pt3(line2[j][0], line2[j][1]);                                                           // 직선2 시작지점
                cv::Point2f pt4(line2[j][2], line2[j][3]);                                                           // 직선2 끝지점

                cv::Point2f cross_point = GetCrossPointFromPT4(pt1, pt2, pt3, pt4);                                  // 직선 두개의 지작점과 끝지점을 받아 교점을 획득


                float cos_ang = GetAngleFromDotProduct(                                                              // 벡터의 내적에서 코사인 각을 획득
                        cross_point - pt2,
                        cross_point - pt4);

                if (cos_ang > 89.800f && cos_ang < 90.200f)                                                          // 코사인 각이 90도가 나오는 경우
                {
                    res_pt = cross_point;

                    goto exit;                                                                                       // 결과값을 저장하고 이중 for문 탈출
                }
            }
        }
    }
    exit:

    return res_pt;
}

auto CardFinder::GetAngleFromTwoPoints(cv::Point2f pt1, cv::Point2f pt2, AreaLocation arealoc) -> float
{
    cv::Point2f pt3;
    float angle = 0.0f;

    if (pt1.y > pt2.y && arealoc == AreaLocation::TOP)                                                               // 상단영역이 반시계방향으로 기운 경우
    {
        pt3 = cv::Point2f(pt1.x, pt2.y);
        angle = -1.0f * GetAngleFromDotProduct(pt3 - pt2, pt1 - pt2);
    }
    else if (pt1.y < pt2.y && arealoc == AreaLocation::TOP)                                                          //상단영역이 시계방향으로 기운 경우
    {
        pt3 = cv::Point2f(pt2.x, pt1.y);
        angle = GetAngleFromDotProduct(pt3 - pt1, pt2 - pt1);
    }
    else if (pt1.y > pt2.y && arealoc == AreaLocation::BOTTOM)                                                       // 하단영역이 반시계방향으로 기운 경우
    {
        pt3 = cv::Point2f(pt2.x, pt1.y);
        angle = -1.0f * GetAngleFromDotProduct(pt3 - pt1, pt2 - pt1);
    }
    else if (pt1.y < pt2.y && arealoc == AreaLocation::BOTTOM)                                                       // 하단영역이 시계방향으로 기운 경우
    {
        pt3 = cv::Point2f(pt1.x, pt2.y);
        angle = GetAngleFromDotProduct(pt3 - pt2, pt1 - pt2);
    }
    else if (pt1.x < pt2.x && arealoc == AreaLocation::LEFT)                                                         // 좌측영역이 반시계방향으로 기운 경우
    {
        pt3 = cv::Point2f(pt1.x, pt2.y);
        angle = -1.0f * GetAngleFromDotProduct(pt3 - pt1, pt2 - pt1);
    }
    else if (pt1.x > pt2.x && arealoc == AreaLocation::LEFT)                                                         // 좌측영역이 시계방향으로 기운 경우
    {
        pt3 = cv::Point2f(pt2.x, pt1.y);
        angle = GetAngleFromDotProduct(pt3 - pt2, pt1 - pt2);
    }
    else if (pt1.x < pt2.x && arealoc == AreaLocation::RIGHT)                                                        // 우측영역이 반시계방향으로 기운 경우
    {
        pt3 = cv::Point2f(pt2.x, pt1.y);
        angle = -1.0f * GetAngleFromDotProduct(pt3 - pt2, pt1 - pt2);
    }
    else if (pt1.x > pt2.x && arealoc == AreaLocation::RIGHT)                                                        // 우측영역이 시계방향으로 기운 경우
    {
        pt3 = cv::Point2f(pt1.x, pt2.y);
        angle = GetAngleFromDotProduct(pt3 - pt1, pt2 - pt1);
    }



    return angle;

}

auto CardFinder::GetCapturedArea() -> cv::Rect&
{
    return m_captured_area;                                                                                          // 전체 관심영역 반환
}

auto CardFinder::GetCoordinates() -> std::vector<float>
{
    return std::move(m_res_coordinate);                                                                              // 스케일 팩터로 보정된 교점들을 반환
}

auto CardFinder::BrightCorrect(cv::Mat& src) -> cv::Mat&
{
    cv::Mat X, A, Y;                                                                                                 // X행렬, A행렬, Y행렬
    int total = src.total();                                                                                         // src.rows * src.cols
    int counter = 0;

    src.convertTo(Y, CV_32FC1);                                                                                      // float 타입으로 변환
    Y = Y.reshape(1, total);                                                                                         // 1행 total열 행렬으로 변환

    X = (m_A.t() * m_A).inv() * m_A.t() * Y;                                                                         // X행렬 생성

    float& a = *X.ptr<float>(0);
    float& b = *X.ptr<float>(1);
    float& c = *X.ptr<float>(2);
    float& d = *X.ptr<float>(3);
    float& e = *X.ptr<float>(4);
    float& f = *X.ptr<float>(5);
    float& g = *X.ptr<float>(6);
    float& h = *X.ptr<float>(7);
    float& i = *X.ptr<float>(8);
    float& j = *X.ptr<float>(9);

    for (int y = 0; y < src.rows; ++y)
    {
        for (int x = 0; x < src.cols; ++x)
        {
            float Z =                                                                                                // 3차 식으로 Z값 계산
                    (a*(x*x*x)) + (b*(x*x)*y) + (c*x*(y*y)) +
                    (d*(y*y*y)) + (e*(x*x)) + (f*x*y) +
                    (g*(y*y)) + (h*x) + (i*y) + j;

            m_br_correction_field.ptr<uchar>(y)[x] =                                                                 // y행 x열에 보정 값을 삽입
                    static_cast<uchar>(static_cast<float>(src.ptr<uchar>(y)[x]) - Z + 128);
        }
    }



    return m_br_correction_field;
}

auto CardFinder::HomomorphicCorrect(cv::Mat& src) -> cv::Mat
{
    cv::Mat log, complex;

    cv::Mat planes[] = { cv::Mat(), cv::Mat() };                                                                     // 실수행렬과 허수행렬이 들어갈 planes

    src.convertTo(log, CV_32FC1);                                                                                    // float 타입 변환

    cv::log((log / 255) + cv::Scalar::all(1), log);                                                                  // log를 취한 영상 획득

    cv::dft(log, complex, cv::DFT_COMPLEX_OUTPUT);                                                                   // 푸리에 연산 수행(2채널 실수+허수 영상 획득)

    cv::split(complex, planes);                                                                                      // 영상 분할

    cv::multiply(planes[0], m_gaussian_filters[0], planes[0]);                                                       // 실수부분과 가우시안 필터 곱연산
    cv::multiply(planes[1], m_gaussian_filters[0], planes[1]);                                                       // 허수부분과 가우시안 필터 곱연산

    cv::merge(planes, 2, complex);                                                                                   // 실수 및 허수 영상 병합

    cv::idft(complex, complex, cv::DFT_REAL_OUTPUT);                                                                 // 역푸리에 연산 후 실수영상 획득

    cv::normalize(complex, complex, 0, 1, cv::NORM_MINMAX);                                                          // 역푸리에 변환으로 얻는 실수영상을 0과 1로 정규화

    cv::exp(complex, complex);                                                                                       // 지수함수 적용

    cv::normalize(complex, complex, 0, 255, cv::NORM_MINMAX, CV_8UC1);                                               // 0과 255로 정규화

    return complex;
}

auto CardFinder::AreaPreProcess(cv::Mat& src, cv::Mat& dst) -> void
{
    cv::resize(src, dst, src.size() * 8, cv::INTER_CUBIC);                                                           // 영상 크기 확대

    erode(dst, dst, m_kernel, cv::Point(-1, -1), 1);                                                                 // 침식연산

    morphologyEx(dst, dst, cv::MORPH_OPEN, m_kernel, cv::Point(-1, -1), 1);                                          // 열림연산(침식후 팽창)

    cv::resize(dst, dst, src.size() / 2, cv::INTER_LANCZOS4);                                                        // 영상 크기 축소
}

auto CardFinder::AreaMasking(cv::Mat& src) -> void
{
    std::vector<std::vector<cv::Point>> contours;                                                                    // 외곽선이 이루는 점들을 저장하는 변수

    cv::findContours(src, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);                                       // 외곽선획득 연산 실행

    int i, size = contours.size();
    for (i = 0; i < size; ++i)
    {
        std::vector<cv::Point2f> approx;                                                                             // 외곽선을 근사하는 점들을 저장하는 변수

        approxPolyDP(contours[i], approx, 0.01, true);                                                               // 근사화된 외곽선 획득

        cv::Rect r = cv::boundingRect(approx);                                                                       // 근사화된 외곽선을 둘러싸는 바운딩 박스 획득

        if (r.area() <= 10)
            cv::rectangle(src, r, cv::Scalar(0), -1);                                                                // 잡음 제거
    }

}

auto CardFinder::AreaSegmant(cv::Mat& src, int offset_width, int offset_height) -> std::vector<cv::Rect>
{
    std::vector<cv::Rect> loc;

    cv::Mat labels, stats, centroid;                                                                                 // 레이블링, 레이블링 Info, 중심값이 들어가는 행렬 객체

    int numLabels = cv::connectedComponentsWithStats(                                                                // OpenCV 레이블링 함수
            src, labels, stats, centroid, 8, CV_32SC1);


    int* mStats_ptr = nullptr;
    for (int j = 0; j < numLabels; ++j)
    {
        mStats_ptr = stats.ptr<int>(j);

        int& left = mStats_ptr[cv::CC_STAT_LEFT];                                                                    // 레이블 시작지점의 x축
        int& top = mStats_ptr[cv::CC_STAT_TOP];                                                                      // 레이블 시작지점의 y축
        int& width = mStats_ptr[cv::CC_STAT_WIDTH];                                                                  // 레이블의 가로길이
        int& height = mStats_ptr[cv::CC_STAT_HEIGHT];                                                                // 레이블의 세로길이

        cv::Rect rect(cv::Point(left, top), cv::Point(left + width, top + height));                                  // 레이블링된 객체를 감싸는 사각형 영역 생성

        if ((rect.width < rect.height) && (rect.height > offset_width && rect.height < offset_height))
            loc.emplace_back(rect);

    }


    return loc;
}

auto CardFinder::DataClassification(std::vector<cv::Rect>& rois) -> std::vector<cv::Rect>
{
    std::vector<cv::Rect> areas;
    int i, j, k, size = rois.size();
    int count1 = 0, count2 = 0;
    int dist = 0;

    std::sort(rois.begin(), rois.end(),                                                                              // y축을 기준으로 정렬
              [&rois](cv::Rect& l, cv::Rect& r) {
                  return l.y < r.y;
              });

    for (i = 0; i < size; ++i)
    {
        for (j = 0; j < size; ++j)
        {
            if (i == j) continue;                                                                                    // i와 j가 같은 index일 경우 continue

            int gap_of_y_pos = cv::abs(rois[i].tl().y - rois[j].tl().y);

            if (gap_of_y_pos < 5 && gap_of_y_pos > 0)                                                                // roi[i]의 y축과 rois[j]의 y축의 차이가 0보다크고 5보다 작은 경우(체크카드 숫자가 놓인 위치로 주정)
                areas.emplace_back(rois[j]);
        }
        if (areas.size() == 15)                                                                                      // 위 조건에 해당하여 areas에 삽입된 원소 개수가 15개인 경우
        {
            areas.push_back(rois[i]);                                                                                // rois[i]를 삽입(숫자 16자리로 추정되는 영역들을 확보)
            goto exit;                                                                                               // 2중 for문 탈출
        }
        areas.clear();                                                                                               // 위 조건을 통화할 시 areas 초기화
    }

    exit:

    if (areas.size() == 16)
    {
        std::sort(areas.begin(), areas.end(),                                                                        // x축 기준으로 정렬
                  [&areas](cv::Rect& l, cv::Rect& r) {
                      return l.x < r.x;
                  });

        for (int i = 0; i < 4; ++i)
        {
            j = (i + 1) * 4;                                                                                         // 16개 숫자영역 중 4, 8, 12번째 자리의 index

            for (k = i * 4; k < j - 1; ++k)                                                                          // 0~3, 4~7, 8~11, 12~15에 해당하는 k, k+1 인덱스를 순회하여 길이를 비교하는 for문
            {
                dist = GetDist(cv::Point(areas[k].br().x, areas[k].tl().y), areas[k + 1].tl());                      // 숫자간의 간격

                if (dist < 5)                                                                                        // 숫자간의 간격이 5보다 작을 경우
                {
                    ++count1;                                                                                        // count1을 1씩 증가
                }
            }

            if (j - 1 < 12)
            {
                dist = GetDist(
                        cv::Point(areas[j - 1].br().x, areas[j - 1].tl().y), areas[j].tl());                         // 3-4, 7-8, 11-12 번째 숫자간의 간격
                if (dist > 6 && dist < 15) ++count2;                                                                 // 숫자간 사이 간격의 조건이 참이면 count2를 1씩 증가

            }

        }

        if (count1 != 12 || count2 != 3) areas.clear();                                                              // 16자리 숫자가 서로 열두 쌍을 이루고 숫자 네 개가 모인 영역 개수가 3이 아닌 경우 areas를 초기화
    }

    return areas;
}

auto CardFinder::DataDiscrimination(
        cv::Mat& src, std::vector<cv::Rect>& areas,
        Module& module, std::map<int, char>& labels) -> std::string
{
    if (src.empty() || src.type() != CV_8UC1) return std::string();
     // 숫자인식의 결과가 저장되는 변수
    std::string digits_str;
    int size = areas.size();
    for (int i = 0; i < size; ++i)
    {
        cv::Mat roi_img = src(areas[i]), num_img;                                                                    // 숫자영상

        AreaPreProcess(roi_img, num_img);                                                                            // 숫자영상 전처리

        num_img = PlaceMiddle(num_img);                                                                              // 숫자객체를 영상의 중앙으로 옮기는 연산

        cv::resize(num_img, num_img, cv::Size(28, 28), cv::INTER_LANCZOS4);                                          // 숫자영상을 28 X 28 크기로 변형

        cv::normalize(num_img, num_img, 0, 1, cv::NORM_MINMAX, CV_32FC1);                                            // 숫자영상의 원소값을 0과 1사이의 float 타입으로 변환

        std::vector<torch::IValue> input;                                                                            // float 타입 숫자영상이 텐서타입으로 변환된 값을 저장하는 변수

        input.push_back(                                                                                             // Mat type num_img를 torch::kFloat32타입으로 변환하여 input에 삽입한다
                torch::from_blob(
                        num_img.data, { 1, 1, 28, 28 }, torch::kFloat32).to(torch::kCPU));

        c10::IValue forward = module.forward(input);                                                                 // 모델 추론 연산 수행

        torch::Tensor output = forward.toTensor();                                                                   // output 텐스 획득
 
        auto pred = output.argmax(1, true);                                                                          // 가장 큰 값에 해당하는 공간 인덱스를 반환받는다

        digits_str += labels[pred.cpu()[0][0].template item<int>()];                                                 // 숫자인식 결과 획득
    }

    return digits_str;
}

auto CardFinder::Start(cv::Mat& src) -> std::string
{
    std::string result;                                                                                              // 체크카드 인식 결과를 받는 변수

    try
    {
        auto col_area_img_proc_f1 = m_pool.PushJob([&, this]() {                                                     // 엣지 및 직선 검출 : 관심영역 상단 부분 가로
            cv::Mat temp;
            cv::Canny(src(m_parts_of_captured_area[0]), temp, 100, 500);
            return this->GetLines(temp, AreaLocation::TOP);
        });

        auto row_area_img_proc_f1 = m_pool.PushJob([&, this]() {                                                     // 엣지 및 직선 검출 : 관심영역 좌측 부분 가로
            cv::Mat temp;
            cv::Canny(src(m_parts_of_captured_area[1]), temp, 100, 500);
            return this->GetLines(temp, AreaLocation::LEFT);
        });

        auto line_col1 = col_area_img_proc_f1.get();
        auto line_row1 = row_area_img_proc_f1.get();

        if (!line_col1.empty() && !line_row1.empty())                                                                // 상단과 좌측의 영역에서 직선들이 발견된 경우
        {
            auto col_area_img_proc_f2 = m_pool.PushJob([&, this]() {                                                 // 엣지 및 직선 검출 : 관심영역 하단 부분 세로
                cv::Mat temp;
                cv::Canny(src(m_parts_of_captured_area[2]), temp, 100, 500);
                return this->GetLines(temp, AreaLocation::BOTTOM);
            });

            auto row_area_img_proc_f2 = m_pool.PushJob([&, this]() {                                                 // 엣지 및 직선 검출 : 관심영역 우측 부분 세로
                cv::Mat temp;
                cv::Canny(src(m_parts_of_captured_area[3]), temp, 100, 500);
                return this->GetLines(temp, AreaLocation::RIGHT);
            });

            auto line_col2 = col_area_img_proc_f2.get();
            auto line_row2 = row_area_img_proc_f2.get();

            if (!line_col2.empty() && !line_row2.empty())                                                            // 하단과 우측의 영역에서 직선들이 발견된 경우
            {
                cv::Point2f zero(0.0f, 0.0f);
                cv::Point2f pt1 = this->GetCrossPointFromTwoLines(line_col1, line_row1);                             // 상단과 좌측영역에서 발견된 직선이 이루는 교점 획득
                cv::Point2f pt2 = this->GetCrossPointFromTwoLines(line_col1, line_row2);                             // 상단과 하단영역에서 발견된 직선이 이루는 교점 획득
                cv::Point2f pt3 = this->GetCrossPointFromTwoLines(line_col2, line_row1);                             // 하단과 좌측영역에서 발견된 직선이 이루는 교점 획득
                cv::Point2f pt4 = this->GetCrossPointFromTwoLines(line_col2, line_row2);                             // 하단과 우측영역에서 발견된 직선이 이루는 교점 획득

                if (pt1 != zero && pt2 != zero && pt3 != zero && pt4 != zero)                                        // 체크카드 네 곳의 모서리 교점을 획득한 경우
                {
                    bool is_inner_pt1 = ComparePosition(                                                             // 교점이 관심영역의 11시 방향의 영역에 들어가는 여부 확인
                            cv::Point2f(0, 0),
                            cv::Point2f(m_cross_pt4[0].x, m_cross_pt4[0].y), pt1);

                    bool is_inner_pt2 = ComparePosition(                                                             // 교점이 관심영역의 1시 방향의 영역에 들어가는 여부 확인
                            cv::Point2f(m_cross_pt4[1].x, 0),
                            cv::Point2f(src.cols, m_cross_pt4[1].y), pt2);

                    bool is_inner_pt3 = ComparePosition(                                                             // 교점이 관심영역의 7시 방향의 영역에 들어가는 여부 확인
                            cv::Point2f(0, m_cross_pt4[2].y),
                            cv::Point2f(m_cross_pt4[2].x, src.rows), pt3);

                    bool is_inner_pt4 = ComparePosition(                                                             // 교점이 관심영역의 5시 방향의 영역에 들어간느 여부 확인
                            cv::Point2f(m_cross_pt4[3].x, m_cross_pt4[3].y),
                            cv::Point2f(src.cols, src.rows), pt4);



                    if ((is_inner_pt1 && is_inner_pt2 && is_inner_pt3 && is_inner_pt4) == true)                      // 교점이 모든 네 곳의 영역 내부에 위치하는 경우
                    {
                        float ang1 = this->GetAngleFromTwoPoints(pt1, pt2, AreaLocation::TOP);                       // 교점 pt1과 pt2를 이은 직선이 기울어진 각도를 계산
                        float ang2 = this->GetAngleFromTwoPoints(pt1, pt3, AreaLocation::LEFT);                      // 교점 pt1과 pt3를 이은 직선이 기울어진 각도를 계산
                        float ang3 = this->GetAngleFromTwoPoints(pt3, pt4, AreaLocation::BOTTOM);                    // 교점 pt3와 pt4를 이은 직선이 기울어진 각도를 계산
                        float ang4 = this->GetAngleFromTwoPoints(pt2, pt4, AreaLocation::RIGHT);                     // 교점 pt2와 pt4를 이은 직선이 기울어진 각도를 계산


                        if ((ang1 != 0.0f) && (ang2 != 0.0f) && (ang3 != 0.0f) && (ang4 != 0.0f))                    // 기울기 각도 네 개를 모두 획득한 경우
                        {
                            SubProcess& instance = SubProcess::GetInstance(src.cols, src.rows);

                            auto f_contrast_equalizing_proc                                                          // 체크카드가 포착된 영상을 명암과 밝기를 보정한다
                                    = m_pool.PushJob([this, &instance, &src]() {

                                        m_clahe->apply(src, src);                                                    // CLAHE Equalization 적용

                                        src = this->BrightCorrect(src);                                              // 최소자승 밝기보정 적용

                                        src =                                                                        // HomomorphicFiltering : low pass, high pass 연산
                                                (0.3f * this->HomomorphicCorrect(src)) +
                                                (1.5f * instance.HomomorphicCorrect(src));

                                    });

                            cv::Mat thresh_black_char_img, thresh_non_black_char_img;
                            std::vector<cv::Rect> loc1, loc2, loc3;
                            bool thresh_inv_bool = false, thresh_bool = false;

                            pt1 = cv::Point2f(m_captured_area.height - 1 - pt1.y, pt1.x);                            // pt1을 시계방향으로 90도 회전
                            pt2 = cv::Point2f(m_captured_area.height - 1 - pt2.y, pt2.x);                            // pt2을 시계방향으로 90도 회전
                            pt3 = cv::Point2f(m_captured_area.height - 1 - pt3.y, pt3.x);                            // pt3을 시계방향으로 90도 회전
                            pt4 = cv::Point2f(m_captured_area.height - 1 - pt4.y, pt4.x);                            // pt4을 시계방향으로 90도 회전

                            this->SetCoordinates(pt3, pt1, pt4, pt2);                                                // 교점에 스케일 팩터를 곱하여 저장

                            cv::Point center(m_captured_area.width / 2, m_captured_area.height / 2);                 // 관심영역의 중심점을 획득

                            cv::Mat m =                                                                              // angle 만큼 회전하는 회전 영상획득
                                    cv::getRotationMatrix2D(center,
                                                            (ang1 + ang2 + ang3 + ang4) / 4, 1);

                            f_contrast_equalizing_proc.get();

                            cv::warpAffine(
                                    src, src, m, m_captured_area.size());                                            // 회전 연산 실행

                                        

                            src = Rotate90<uchar, CV_8UC1>(src);                                                     // 90도 회전 실행

                            cv::threshold(                                                                           // 영상 이진화
                                    src, thresh_black_char_img,
                                    128, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

                            thresh_non_black_char_img = ~thresh_black_char_img;                                      // 이진화된 영상을 다시 반전시킨다(검은 글씨가 아닌 체크카드인 경우)

                            auto f_classify_area1 =
                                    m_pool.PushJob(
                                            [this, &thresh_black_char_img, &thresh_inv_bool](){

                                this->AreaMasking(thresh_black_char_img);                                            // 이진영상 내부의 잡음을 제거

                                auto local_loc =                                                                     // 이진영상 내부에서 특정 조건에 해당하는 영역들을 획득한다
                                        this->AreaSegmant(thresh_black_char_img, 14, 100);

                                local_loc = this->DataClassification(local_loc);                                     // 숫자영역으로 추정되는 16자리 영역을 획득한다

                                if(local_loc.size() == 16) thresh_inv_bool = true;                                   // 숫자영역 16개를 발견한 경우 boolean값 true로 설정

                                return local_loc;
                            });

                            auto f_classify_area2 =
                                    m_pool.PushJob(
                                            [this, &thresh_non_black_char_img, &thresh_bool](){

                                this->AreaMasking(thresh_non_black_char_img);

                                auto local_loc =
                                        this->AreaSegmant(thresh_non_black_char_img, 14, 100);

                                local_loc = this->DataClassification(local_loc);

                                if(local_loc.size() == 16) thresh_bool = true;

                                return local_loc;
                            });


                            loc1 = f_classify_area1.get();
                            loc2 = f_classify_area2.get();

                            if(loc1.size() == 16) loc3 = loc1;
                            if(loc2.size() == 16) loc3 = loc2;

                            if (loc3.size() == 16)                                                                   // 숫자 16개 영역을 획득 시
                            {
                                if(thresh_inv_bool == true)                                                          // 검은글씨 체크카드에서 숫자 16개 영역을 획득한 경우
                                    result = this->DataDiscrimination(                                               // 숫자 16자리 영역 이미지 인식 실행
                                                    thresh_black_char_img, loc3, m_modules[0], m_labels_number);
                                if(thresh_bool == true)                                                              // 검은글씨가 아닌 체크카드에서 숫자 16개 영역을 획득한 경우
                                    result = this->DataDiscrimination(
                                                    thresh_non_black_char_img, loc3, m_modules[0], m_labels_number);

                                for (int i = 0, j = 0; i < 16; i++)                                                  // 숫자 네 자리 마다 - 기호 삽입
                                {
                                    if (i > 0 && i % 4 == 0)
                                    {
                                        result.insert(i + j, "-");                                                   // 숫자영역 네 개가 뭉친 영역 사이에 - 삽입
                                        ++j;
                                    }
                                }

                                int max_y = std::max(loc3[0].br().y, loc3[15].br().y);                               // 숫자 16자리 영역들 중 첫번째와 마지막 숫자 영역의 하단 부분 y축의 최대값 계산

                                cv::Mat area_of_under_of_numbers;                                                    // 숫자 16자리 영역 아래 영역에 해당하는 영상이 저장될 객체
                                if(thresh_inv_bool == true)
                                    area_of_under_of_numbers =                                                       // 하단영역의 영상을 지정한다
                                            thresh_black_char_img(cv::Rect(
                                                    cv::Point(0, max_y),
                                                    cv::Point(thresh_black_char_img.cols - 1, thresh_black_char_img.rows - 1)));
                                if(thresh_bool == true)
                                    area_of_under_of_numbers =
                                            thresh_non_black_char_img(cv::Rect(
                                                    cv::Point(0, max_y),
                                                    cv::Point(thresh_non_black_char_img.cols - 1, thresh_non_black_char_img.rows - 1)));

                                loc3.clear();

                                loc3 = this->AreaSegmant(
                                        area_of_under_of_numbers, 10, 50);                                           // 객체 영역 검출

                                loc3 = instance.DataClassification(loc3);                                            // 하단 영역 영상에서 알파벳 이름이 있는 아홉자리 영역을 추출한다

                                if(loc3.size() == 9)                                                                 // 체크카드 이름 알파벳영역 9개를 검출한 경우
                                {
                                    result += ("/" + this->DataDiscrimination(                                       // 알파벳 9자리 영역 이미지 인식 실행
                                            area_of_under_of_numbers, loc3, m_modules[1], m_labels_alphabet));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    catch (std::exception& ex)
    {
        std::string error = ex.what();
    }

    return result;                                                                                                   // 숫자 및 알파벤 인식 결과값 반환
}



