#include <opencv2/opencv.hpp>
#include <omp.h>
#include <iostream>

using namespace cv;
using namespace std;

// Функция для применения сверточного слоя с заданным ядром
Mat applyConvolution(const Mat& inputImage, const Mat& kernel) {
    // Замер времени начала выполнения
    double startTime = omp_get_wtime();

    Mat outputImage = inputImage.clone();
    int kernelRadius = kernel.rows / 2;

#pragma omp parallel for collapse(2) shared(inputImage, outputImage, kernel) num_threads(6)
    for (int y = kernelRadius; y < inputImage.rows - kernelRadius; ++y) {
        for (int x = kernelRadius; x < inputImage.cols - kernelRadius; ++x) {
            // Применение свертки в текущем пикселе
            float sum = 0.0f;
            for (int ky = -kernelRadius; ky <= kernelRadius; ++ky) {
                for (int kx = -kernelRadius; kx <= kernelRadius; ++kx) {
                    sum += inputImage.at<uchar>(y + ky, x + kx) * kernel.at<float>(ky + kernelRadius, kx + kernelRadius);
                }
            }

            // Запись результата в выходное изображение
            outputImage.at<uchar>(y, x) = saturate_cast<uchar>(sum);
        }
    }

    // Замер времени окончания выполнения
    double endTime = omp_get_wtime();
    cout << "Time : " << endTime - startTime << " seconds" << endl;

    return outputImage;
}

// ядро свертки для усиления границ
Mat simpleKernel = (Mat_<float>(3, 3) << -1, -1, -1, -1, 8, -1, -1, -1, -1);

Mat simpleGenerateRandomKernel(int size) {
    Mat kernel(size, size, CV_32F);
    srand(static_cast<unsigned>(time(0)));

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            kernel.at<float>(i, j) = static_cast<float>(rand()) / RAND_MAX - 0.5f;
        }
    }

    return kernel;
}

int main() {
    Mat inputImage = imread("test.png", IMREAD_GRAYSCALE);

    if (inputImage.empty()) {
        cerr << "Could not open or find the image!" << endl;
        return -1;
    }

    Mat kernel = simpleKernel;

    // Применение сверточного слоя
    Mat outputImage = applyConvolution(inputImage, kernel);

    // Вывод результатов
    namedWindow("Input Image", WINDOW_NORMAL);
    namedWindow("Output Image", WINDOW_NORMAL);

    imshow("Input Image", inputImage);
    imshow("Output Image", outputImage);

    waitKey(0);

    return 0;
}
