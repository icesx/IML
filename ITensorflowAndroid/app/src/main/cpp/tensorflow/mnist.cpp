//
// Created by i on 20-5-1.
//
#include <android/log.h>

#define LOG_TAG "GPIO-SYSFS-JNI"
#define LOGI(...)  __android_log_print(ANDROID_LOG_INFO,LOG_TAG,__VA_ARGS__)
#define LOGE(...)  __android_log_print(ANDROID_LOG_ERROR,LOG_TAG,__VA_ARGS__)

int wpiToSysfs(int physPin);
