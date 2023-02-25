#include <CardFinder.hpp>

// android studio java sdk가 매 프레임 마다 호출하는 함수
extern "C"
JNIEXPORT jstring JNICALL
Java_com_sjlee_cardfinder_ViewActivity_PlayCamera(
        JNIEnv *env, jobject thiz,
        jbyteArray array, int width, int height) {

    jbyte* yuv  = env->GetByteArrayElements(array, 0);

    cv::Mat img(height, width, CV_8UC1, (uchar *)yuv);

    CardFinder& instance = CardFinder::GetInstance(height, width, *env, thiz);

    env->ReleaseByteArrayElements(array, yuv, 0);

    return env->NewStringUTF(instance.Start(img).c_str());
}


