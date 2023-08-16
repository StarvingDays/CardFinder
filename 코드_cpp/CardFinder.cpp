#include <CardFinder.hpp>


CardFinder::CardFinder(JNIEnv& env, jobject& obj, int& w, int& h)
        :
        m_image_proxy_size(cv::Size(w, h)),
        m_image_view_size(GetImageViewSize(env, obj, "com/sjlee/cardfinder/ViewActivity")),                          // 미리보기 관심영역 사이즈
        m_captured_area(SetCapturedArea(w, h)),                                                                      // 전체 영역
        m_scale_factor(SetScaleFactor(w, h)),                                                                        // 관심영역에 놓일 교점의 위치비율을 보정하는 스케일 팩터
        m_parts_of_captured_area(SetPartsOfCapturedArea()),                                                          // 부분영역(가로, 세로)
        m_A(SetBrightCorrectionModel()),                                                                             // 최소자승 A행렬들(전체영역, 가로영역, 세로영역)
        m_br_correction_field(SetBrightCorrectionFields()),                                                          // 최소자승 보정으로 인한 결과 영상을 저장하는 영상(전체영역, 가로영역, 세로영역)
        m_kernel(cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3))),                                        // 침식 및 확장 연산시 사용되는 커널
        m_clahe(SetCLAHE(4.0, cv::Size(8, 8))),                                                                      // 명암 보정에 쓰이는 CLAHE 객체
        m_gaussian_filters(SetGaussianFilters(m_captured_area.size(), 30)),                                          // 가우시안 low, high 필터
        m_pull_thr_on(true),
        m_pull_thr(2),
        m_client("0.0.0.0", "0000", "/OCR", 11)
{
    for(int i = 0; i < m_pull_thr.size(); ++i)
    {
        m_pull_thr[i] = std::thread([this](){                                                                        // 버퍼에서 저장된 future 타입객체를 꺼내는 스레드를 생성하여 실행한다

            while(m_pull_thr_on.load(std::memory_order_release))                                                     // m_pull_thr_on 값이 참일 때 while 루프 실행
            {
                std::future<void> job;
                {
                    std::unique_lock<std::mutex> lcok(m_mutex);                                                      // 임계 영역 설정
                    m_conv.wait(lcok, [this]{ return !m_image_preprocessing_jobs.empty(); });                        // m_processing_jobs 버퍼가 비어있는 경우 모든 스레드를 대기시킨다
                    job = std::move(m_image_preprocessing_jobs.front());                                             // 대기상태에서 꺠어난 스레드가 로컬 future 변수(job)에 버퍼에 있는 future 객체의 소유권을 이전한다
                    m_image_preprocessing_jobs.pop();                                                                // queue 맨 앞쪽 객체 제거
                }
                job.get();                                                                                           // 로컬 future 객체 실행
            }
        });
    }
}

CardFinder::~CardFinder()
{
    m_conv.notify_all();                                                                                             // 소멸자 호출 시 모든 스레드들을 꺠운다

    m_pull_thr_on.store(false, std::memory_order_release);                                                           // while 루프 탈출

    for(int i = 0; i < m_pull_thr.size(); ++i) m_pull_thr[i].join();                                                 // thread join
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


auto CardFinder::SetPartsOfCapturedArea() -> Rects
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

    cv::Point cross_pt1 = FindCrossPoint(pt1, pt2, pt3, pt4);                                                        // 교점1
    cv::Point cross_pt2 = FindCrossPoint(pt1, pt2, pt7, pt8);                                                        // 교점2
    cv::Point cross_pt3 = FindCrossPoint(pt3, pt4, pt5, pt6);                                                        // 교점3
    cv::Point cross_pt4 = FindCrossPoint(pt5, pt6, pt7, pt8);                                                        // 교점4

    m_start_pt_of_right_area = cross_pt2;
    m_start_pt_of_bottom_area = cross_pt3;

    return Rects() = {
            cv::Rect(cv::Point(cross_pt1.x, 0), cross_pt2),                                                          // 상단 관심영역
            cv::Rect(pt1, cross_pt3),                                                                                // 좌측 관심영역
            cv::Rect(cross_pt3, cv::Point(cross_pt4.x, pt8.y)),                                                      // 하단 관심영역
            cv::Rect(cross_pt2, pt6)                                                                                 // 우측 관심영역
    };

}

auto CardFinder::SetBrightCorrectionModel() -> cv::Mat
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

auto CardFinder::SetGaussianFilters(cv::Size size, double D0) -> std::vector<cv::Mat>
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
    std::mutex m;
    std::lock_guard<std::mutex> lock(m);
    {
    m_res_coordinate = {                                                                                             // 관심영역에 위치한 체크카드의 교점 네 곳이 ImageView에서의 위치비율과
                pt1.x * m_scale_factor.x, pt1.y * m_scale_factor.y,                                                  // 호환되도록 교점의 x,y축에 스케일 팩터를 곱한다
                pt2.x * m_scale_factor.x, pt2.y * m_scale_factor.y,
                pt3.x * m_scale_factor.x, pt3.y * m_scale_factor.y,
                pt4.x * m_scale_factor.x, pt4.y * m_scale_factor.y,
        };
    }
}

auto CardFinder::SetCLAHE(double limit_var, cv::Size tile_size) -> cv::Ptr<cv::CLAHE>
{
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();                                                                    // CLAHE 객체 생성

    clahe->setClipLimit(limit_var);                                                                                  // 리미트 값 설정
    clahe->setTilesGridSize(tile_size);                                                                              // 그리드 사이즈 설정

    return std::move(clahe);
}

auto CardFinder::FindLines(cv::Mat& src, AreaLocation arealoc) -> Lines {
    Lines linesP, resultP;                                                                                           // 발견된 직선을 저장하는 객체

    cv::HoughLinesP(src, linesP, 1, CV_PI / 180, 50, 50, 5);                                                         // 영상에서 직선을 검출하는 OpenCV 함수

    bool col_is_big = src.cols > src.rows;                                                                           // 가로 세로의 길이 비교

    size_t line_size = linesP.size();                                                                                // 검출된 직선들의 개수

    if (line_size <= 10)
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
                            cv::Vec4f(x1, y1 + m_start_pt_of_bottom_area.y, x2, y2 + m_start_pt_of_bottom_area.y));
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
                            cv::Vec4f(x1 + m_start_pt_of_right_area.x, y1, x2 + m_start_pt_of_right_area.x, y2));
                }
            }
        }
    }

    return resultP;
}


auto CardFinder::FindCorner(Lines& lines1, Lines& lines2) -> cv::Point2f
{
    cv::Point2f res_pt;

    if (lines1.size() != 0 && lines2.size() != 0)
    {
        size_t size_col_mn = lines1.size();
        size_t size_row_mn = lines2.size();

        for (int i = 0; i < size_col_mn; ++i)
        {
            cv::Point2f pt1(lines1[i][0], lines1[i][1]);                                                             // 직선1 시작지점
            cv::Point2f pt2(lines1[i][2], lines1[i][3]);                                                             // 직선1 끝지점

            for (int j = 0; j < size_row_mn; ++j)
            {
                cv::Point2f pt3(lines2[j][0], lines2[j][1]);                                                         // 직선2 시작지점
                cv::Point2f pt4(lines2[j][2], lines2[j][3]);                                                         // 직선2 끝지점

                cv::Point2f cross_point = FindCrossPoint(pt1, pt2, pt3, pt4);                                        // 직선 두개의 시작점과 끝지점을 받아 교점을 획득


                float cos_ang = FindAngle(                                                                           // 벡터의 내적에서 코사인 각을 획득
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

auto CardFinder::GetCoordinates() -> std::vector<float>
{
    std::vector<float> res;
    std::mutex m;
    std::lock_guard<std::mutex> lock(m);
    {
        res = std::move(m_res_coordinate);
    }

    return res;                                                                                                      // 스케일 팩터로 보정된 교점들을 반환
}

auto CardFinder::GetResult() -> std::string
{
    if(m_client.Connect())
        m_client.StartAnalysis();

    return m_client.GetMessage();
};

auto CardFinder::BrightCorrect(cv::Mat& src) -> cv::Mat&
{
    cv::Mat X, A, Y;                                                                                                 // X행렬, A행렬, Y행렬
    int total = src.total();                                                                                         // src.rows * src.cols
    int counter = 0;

    src.convertTo(Y, CV_32FC1);                                                                                      // float 타입으로 변환
    Y = Y.reshape(1, total);                                                                                         // 1행 total열 행렬으로 변환

    X = (m_A.t() * m_A).inv() * m_A.t() * Y;                                                                         // X행렬 생성
                                                                                                                     // a ~ j : x 값 획득
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

auto CardFinder::HomomorphicCorrect(cv::Mat& src, cv::Mat& filter) -> cv::Mat
{
    cv::Mat log, complex;

    cv::Mat planes[] = { cv::Mat(), cv::Mat() };                                                                     // 실수행렬과 허수행렬이 들어갈 planes

    src.convertTo(log, CV_32FC1);                                                                                    // float 타입 변환

    cv::log((log / 255) + cv::Scalar::all(1), log);                                                                  // log를 취한 영상 획득

    cv::dft(log, complex, cv::DFT_COMPLEX_OUTPUT);                                                                   // 푸리에 연산 수행(2채널 실수+허수 영상 획득)

    cv::split(complex, planes);                                                                                      // 영상 분할

    cv::multiply(planes[0], filter, planes[0]);                                                                      // 실수부분과 가우시안 필터 곱연산
    cv::multiply(planes[1], filter, planes[1]);                                                                      // 허수부분과 가우시안 필터 곱연산

    cv::merge(planes, 2, complex);                                                                                   // 실수 및 허수 영상 병합

    cv::idft(complex, complex, cv::DFT_REAL_OUTPUT);                                                                 // 역푸리에 연산 후 실수영상 획득

    cv::normalize(complex, complex, 0, 1, cv::NORM_MINMAX);                                                          // 역푸리에 변환으로 얻는 실수영상을 0과 1로 정규화

    cv::exp(complex, complex);                                                                                       // 지수함수 적용

    cv::normalize(complex, complex, 0, 255, cv::NORM_MINMAX, CV_8UC1);                                               // 0과 255로 정규화

    return complex;
}

auto CardFinder::Start(unsigned char* data, jint& col, jint& row) -> void
{
    if(m_stop_image_preprocessing.load(std::memory_order_release) == false)                                                  // m_atom_bool값이 true일 경우 영상 전처리 실행
    {
        std::mutex m;
        {
            std::lock_guard<std::mutex>lock1(m);

            m_image_preprocessing_jobs.push(std::async(std::launch::async, [this](std::vector<unsigned char>&& img_buffur)   // 이미지 전처리 및 기타 연산들을 std::async를 사용하여 비동기로 실행한다
            {
                try
                {
                    int captured_width = m_captured_area.width;
                    int captured_height = m_captured_area.height;

                    cv::Mat img = cv::Mat(
                            m_image_proxy_size.height, m_image_proxy_size.width,
                            CV_8UC1, img_buffur.data(), m_image_proxy_size.width)(m_captured_area);


                    cv::Mat canny;
                    cv::Canny(img(m_parts_of_captured_area[0]), canny, 100, 500);                                           // Canny 알고리즘 적용
                    auto line_col1 = this->FindLines(canny, AreaLocation::TOP);                                             // 직선검출(상단)


                    cv::Canny(img(m_parts_of_captured_area[1]), canny, 100, 500);
                    auto line_row1 = this->FindLines(canny, AreaLocation::LEFT);                                            // 직선검출(좌측)


                    cv::Canny(img(m_parts_of_captured_area[2]), canny, 100, 500);
                    auto line_col2 = this->FindLines(canny, AreaLocation::BOTTOM);                                          // 직선검출(하단)


                    cv::Canny(img(m_parts_of_captured_area[3]), canny, 100, 500);
                    auto line_row2 = this->FindLines(canny, AreaLocation::RIGHT);                                           // 직선검출(우측)

                    if (!line_col1.empty() && !line_row1.empty() && !line_col2.empty() && !line_row2.empty())
                    {
                        cv::Point2f zero(0.0f, 0.0f);

                        cv::Point2f pt1 = this->FindCorner(line_col1, line_row1);                                           // 상단과 좌측영역에서 발견된 직선이 이루는 교점 획득
                        cv::Point2f pt2 = this->FindCorner(line_col1, line_row2);                                           // 상단과 하단영역에서 발견된 직선이 이루는 교점 획득
                        cv::Point2f pt3 = this->FindCorner(line_col2, line_row1);                                           // 하단과 좌측영역에서 발견된 직선이 이루는 교점 획득
                        cv::Point2f pt4 = this->FindCorner(line_col2, line_row2);                                           // 하단과 우측영역에서 발견된 직선이 이루는 교점 획득

                        if (pt1 != zero && pt2 != zero && pt3 != zero && pt4 != zero)
                        {                                                                                                   // 전체 관심영역에 교점이 포함되는 여부 확인
                            bool is_inner_pt1 =
                                    (pt1.x < img.cols && pt1.y < img.rows) &&
                                    (pt1.x > 0 && pt1.y > 0);

                            bool is_inner_pt2 =
                                    (pt2.x < img.cols && pt1.y < img.rows) &&
                                    (pt2.x > 0 && pt1.y > 0);

                            bool is_inner_pt3 =
                                    (pt3.x < img.cols && pt1.y < img.rows) &&
                                    (pt3.x > 0 && pt1.y > 0);

                            bool is_inner_pt4 =
                                    (pt4.x < img.cols && pt1.y < img.rows) &&
                                    (pt4.x > 0 && pt1.y > 0);

                            if ((is_inner_pt1 && is_inner_pt2 && is_inner_pt3 && is_inner_pt4) == true)                      // 교점이 모든 네 곳의 영역 내부에 위치하는 경우
                            {
                                cv::Point2f tan_pos_pt;                                                                      // tan 지점 교점

                                float ang1 = 0.0f, ang2 = 0.0f, ang3 = 0.0f, ang4 = 0.0f;

                                if (pt1.y > pt2.y)                                                                           // 상단영역이 반시계방향으로 기운 경우
                                {
                                    tan_pos_pt = cv::Point2f(pt1.x, pt2.y);
                                    ang1 = FindAngle(tan_pos_pt - pt2, pt1 - pt2);
                                }
                                if (pt1.y < pt2.y)                                                                           // 상단영역이 시계방향으로 기운 경우
                                {
                                    tan_pos_pt = cv::Point2f(pt2.x, pt1.y);
                                    ang1 = FindAngle(tan_pos_pt - pt1, pt2 - pt1);
                                }
                                if (pt3.y > pt4.y)                                                                           // 하단영역이 반시계방향으로 기운 경우
                                {
                                    tan_pos_pt = cv::Point2f(pt4.x, pt3.y);
                                    ang2 = FindAngle(tan_pos_pt - pt3, pt4 - pt3);
                                }
                                if (pt3.y < pt4.y)                                                                           // 하단영역이 시계방향으로 기운 경우
                                {
                                    tan_pos_pt = cv::Point2f(pt3.x, pt4.y);
                                    ang2 = FindAngle(tan_pos_pt - pt4, pt3 - pt4);
                                }
                                if (pt1.x < pt3.x)                                                                           // 좌측영역이 반시계방향으로 기운 경우
                                {
                                    tan_pos_pt = cv::Point2f(pt1.x, pt3.y);
                                    ang3 = FindAngle(tan_pos_pt - pt1, pt3 - pt1);
                                }
                                if (pt1.x > pt3.x)                                                                           // 좌측영역이 시계방향으로 기운 경우
                                {
                                    tan_pos_pt = cv::Point2f(pt3.x, pt1.y);
                                    ang3 = FindAngle(tan_pos_pt - pt3, pt1 - pt3);
                                }
                                if (pt2.x < pt4.x)                                                                           // 우측영역이 반시계방향으로 기운 경우
                                {
                                    tan_pos_pt = cv::Point2f(pt4.x, pt2.y);
                                    ang4 = FindAngle(tan_pos_pt - pt4, pt2 - pt4);
                                }
                                if (pt2.x > pt4.x)                                                                           // 우측영역이 시계방향으로 기운 경우
                                {
                                    tan_pos_pt = cv::Point2f(pt2.x, pt4.y);
                                    ang4 = FindAngle(tan_pos_pt - pt2, pt4 - pt2);
                                }


                                if ((ang1 != 0.0f) && (ang2 != 0.0f) && (ang3 != 0.0f) && (ang4 != 0.0f))                    // 기울기 각도 네 개를 모두 획득한 경우
                                {

                                    m_clahe->apply(img, img);                                                                // CLAHE Equalization 적용

                                    img = this->BrightCorrect(img);                                                          // 최소자승 밝기보정 적용

                                    img =                                                                                    // HomomorphicFiltering : low pass + high pass 연산
                                            (0.3f * this->HomomorphicCorrect(img, m_gaussian_filters[0])) +
                                            (1.5f * this->HomomorphicCorrect(img, m_gaussian_filters[1]));

                                    cv::Point center(captured_width / 2, captured_height / 2);                               // 관심영역의 중심점을 획득

                                    cv::Mat rot = cv::getRotationMatrix2D(center, (ang1 + ang2 + ang3 + ang4) / 4, 1);       // angle 만큼 회전하는 회전 영상획득

                                    cv::warpAffine(img, img, rot, cv::Size(captured_width, captured_height));                // 회전 연산 실행


                                    img = Rotate90<uchar, CV_8UC1>(img);                                                     // 90도 회전 실행

                                    std::mutex m;
                                    {
                                        std::lock_guard<std::mutex> lock(m);                                                 // 한 개의 스레드만 접근 하도록 lock 실행

                                        if(m_stop_image_preprocessing.load(std::memory_order_release) == false)
                                        {
                                            pt1 = cv::Point2f(captured_height - 1 - pt1.y, pt1.x);                           // pt1을 시계방향으로 90도 회전
                                            pt2 = cv::Point2f(captured_height - 1 - pt2.y, pt2.x);                           // pt2을 시계방향으로 90도 회전
                                            pt3 = cv::Point2f(captured_height - 1 - pt3.y, pt3.x);                           // pt3을 시계방향으로 90도 회전
                                            pt4 = cv::Point2f(captured_height - 1 - pt4.y, pt4.x);                           // pt4을 시계방향으로 90도 회전

                                            this->SetCoordinates(pt3, pt1, pt4, pt2);                                        // 교점에 스케일 팩터를 곱한다

                                            m_client.SetData(img);                                                           // 이미지 데이터를 client 인스턴스에 base64 포멧으로 변환하여 저장

                                            m_stop_image_preprocessing.store(true, std::memory_order_release);               // m_atom_bool을 false로 저장하여 추가로 영상 데이터가 저장되는 것을 막는다
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                catch (std::exception &ex) {

                }

            }, std::vector<unsigned char>(data, data + (col * row))));

            m_conv.notify_one();
        }
    }
}

auto CardFinder::PullJobs() -> void
{
    m_stop_image_preprocessing.store(true, std::memory_order_release);

    std::mutex m;
    std::lock_guard<std::mutex> lock(m);
    {
        while(!m_image_preprocessing_jobs.empty())                                                                           // task queue가 비어있을 때 까지 while 루프 실행
        {

        }
    }


}


auto CardFinder::ResetCoordinates() -> void
{
    std::mutex m;
    {
        std::lock_guard<std::mutex> lock(m);
        m_res_coordinate.clear();                                                                                            // 저장된 교점 데이터 초기화
    }
}


auto CardFinder::ResetClientBuffer() -> void
{
    std::mutex m;
    {
        std::lock_guard<std::mutex> lock(m);
        m_client.ResetBuffer();                                                                                              // 클라이언트에 저장된 버퍼 초기화
    }
}

auto CardFinder::ResetStopImagePreprocessingBool() -> void
{
    m_stop_image_preprocessing.store(false);                                                                                 // 이미지 전처리용 atomic<bool> 객체 초기화
}

auto CardFinder::IsEmptyBuffer() -> bool
{
    bool is_empty;
    std::mutex m;
    {
        std::lock_guard<std::mutex> lock(m);
        is_empty = m_client.IsBufferEmpty();                                                                                 // 클라이언트에 버퍼가 비어있는 여부 확인
    }
    return is_empty;
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_sjlee_cardfinder_ViewActivity_GetResult(JNIEnv *env, jobject thiz)
{
    int w = 0, h = 0;

    CardFinder& instance = CardFinder::GetInstance(*env, thiz, w, h);

    return env->NewStringUTF(instance.GetResult().c_str());
}

extern "C"
JNIEXPORT void JNICALL
Java_com_sjlee_cardfinder_ViewActivity_ImagePreProcessing(
        JNIEnv *env, jobject thiz, jbyteArray array, jint width, jint height)
{

    CardFinder& instance = CardFinder::GetInstance(*env, thiz, width, height);                                               // 싱글턴 CardFinder 객체 생성

    jbyte* yData = env->GetByteArrayElements(array, nullptr);                                                                // yData는 GetByteArrayElements함수로 받아온 jbyte 타입 객체를 가리킨다

    instance.Start(reinterpret_cast<unsigned char*>(yData), width, height);                                                  // 이미지 전처리 시작

    env->ReleaseByteArrayElements(array, yData, JNI_ABORT);                                                                  // 자원 할당 해제
}

extern "C"
JNIEXPORT jfloatArray JNICALL
Java_com_sjlee_cardfinder_ViewActivity_GetCoordinates(
        JNIEnv *env, jobject thiz, jint width, jint height)
{
    CardFinder& instance = CardFinder::GetInstance(*env, thiz, width, height);

    std::vector<float> arr = instance.GetCoordinates();                                                                      // 체크카드 네 개 모서리의 교점을 획득

    jfloatArray javaArray = env->NewFloatArray(0);                                                                           // 비어있는 jfloatArray 생성

    if(!arr.empty())
    {
        int size = arr.size();
        javaArray = env->NewFloatArray(size);                                                                                // javaArray에 size크기 만큼 할당
        env->SetFloatArrayRegion(javaArray, 0, size, arr.data());                                                            // javaArray에 arr에 있는 원소들을 삽입
    }

    return javaArray;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_sjlee_cardfinder_ViewActivity_PullJobs(
        JNIEnv *env, jobject thiz, jint width, jint height)
{

    CardFinder& instance = CardFinder::GetInstance(*env, thiz, width, height);
    instance.PullJobs();                                                                                                     // task queue가 비어있을 때 까지 대기
}

extern "C"
JNIEXPORT void JNICALL
Java_com_sjlee_cardfinder_ViewActivity_SetDefaultValue(
        JNIEnv *env, jobject thiz, jint width, jint height)
{

    CardFinder& instance = CardFinder::GetInstance(*env, thiz, width, height);
                                                                                                                             // 초기값 설정
    instance.ResetCoordinates();
    instance.ResetStopImagePreprocessingBool();
    instance.ResetClientBuffer();
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_sjlee_cardfinder_ViewActivity_IsEmptyClientBuffer(
        JNIEnv *env, jobject thiz, jint width, jint height) {

    CardFinder& instance = CardFinder::GetInstance(*env, thiz, width, height);

    return instance.IsEmptyBuffer();                                                                                         // 클라이언트 내부 버퍼가 비어있는 여부 확인
}
