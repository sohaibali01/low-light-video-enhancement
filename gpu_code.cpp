#include <iostream>
#include <vector>
#include <string>
//#include <cctype>
//#include <numeric>
//#include <ppl.h>
//#include <windows.h>
//#include <random>
//#include <algorithm>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>

using namespace std;

#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudalegacy.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudabgsegm.hpp>
#include <opencv2/cudacodec.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/cudawarping.hpp>


using namespace cv;
using namespace std;

std::vector<cv::cuda::GpuMat> contrastPyramidDecomposition(cv::cuda::GpuMat grayImage, const int level);
cv::cuda::GpuMat contrastPyramidReconstruction(std::vector<cv::cuda::GpuMat> contrastPyramid);
//cv::cuda::GpuMat enhanceUWT(cv::cuda::GpuMat grayImage);
//std::vector<cv::cuda::GpuMat> cropImage (cv::cuda::GpuMat image);
//cv::cuda::GpuMat mergeImages(std::vector<cv::cuda::GpuMat> images);

int main() {

    cv::setUseOptimized(true);
//    cout<< "threads \t" << cv::getNumThreads() << endl;

    cv::setNumThreads(16);
    cv::VideoCapture inputVideo("input.mp4"); // open the default camera
    cv::VideoWriter outputVideo("reslult.avi", CV_FOURCC('P','I','M','1'), inputVideo.get(CV_CAP_PROP_FPS),
                                cv::Size(inputVideo.get(CV_CAP_PROP_FRAME_WIDTH), inputVideo.get(CV_CAP_PROP_FRAME_HEIGHT)));

    if( ! inputVideo.isOpened () )  // check if we succeeded
        return -1;

    cv::Size S = Size((int) inputVideo.get(CV_CAP_PROP_FRAME_WIDTH),    // Acquire input size
                      (int) inputVideo.get(CV_CAP_PROP_FRAME_HEIGHT));

    double frnb ( inputVideo.get ( CV_CAP_PROP_FRAME_COUNT ) );
    std::cout << "Frame count = " << frnb << std::endl;
    std::cout << "Frame size = " << S.width << " height " << S.height;
    double execution_time = 0;

    int level = 3;
    cv::cuda::GpuMat img_cuda, rgbImage, hsvImage, grayImage;
    cv::Mat img, out_image;
    std::vector<cv::cuda::GpuMat> inputChannels;

    frnb = 100;
    for(int no = 0; no < frnb; ++no) {
        bool success = inputVideo.read(img);
        if (!success) {
            std::cout << "Cannot read  frame " << std::endl;
            break;
        }
//        cv::resize(img, img, cv::Size(1280, 720));

        img_cuda.upload(img);
        double e1 = cv::getTickCount();

        cv::cuda::cvtColor(img_cuda, hsvImage, cv::COLOR_BGR2HSV);
        cv::cuda::split(hsvImage, inputChannels);

        grayImage = inputChannels[2];

        std::vector<cv::cuda::GpuMat> contrastPyramid = contrastPyramidDecomposition (grayImage, level);
        cv::cuda::GpuMat outImage =  contrastPyramidReconstruction (contrastPyramid);
//        outImage.convertTo(outImage, CV_8UC1 ,0, 255);
        inputChannels[2] = outImage;
//        inputChannels[2] = grayImage;
        cv::cuda::merge(inputChannels, hsvImage);
        cv::cuda::cvtColor(hsvImage, rgbImage, cv::COLOR_HSV2BGR);


//        cv::namedWindow("hsv_image", WINDOW_AUTOSIZE);
//        cv::imshow("hsv_image", out_image);

        double e2 = cv::getTickCount();
        execution_time = execution_time + (e2 - e1) / cv::getTickFrequency();

        // cout << no << "\t" << endl;
        rgbImage.download(out_image);
//        out_image.convertTo(out_image, CV_8UC3 , 255);
        outputVideo.write(out_image);
    }

        std::cout << "\n No of frames per seconds : " << frnb/execution_time << std::endl;

    inputVideo.release();
    outputVideo.release();
    return 0;
}

std::vector<cv::cuda::GpuMat> contrastPyramidDecomposition(cv::cuda::GpuMat grayImage, const int level) {
//	int meanInput = cv::mean(grayImage).val[0];
    cv::Scalar mean, stddev;
    cuda::meanStdDev(grayImage, mean, stddev);
    int meanInput = mean.val[0];

    int bias = std::max(38, meanInput);
    std::vector<cv::cuda::GpuMat> contrastPyramid(level), downSampleImgs(level + 1), upSampleImgs(level);
    downSampleImgs[0] = grayImage;

    for (int i = 0; i < level; i++) {
//		cuda::pyrDown(downSampleImgs[i], downSampleImgs[i + 1], Size(downSampleImgs[i].cols / 2, downSampleImgs[i].rows / 2));
//		cuda::pyrUp(downSampleImgs[i + 1], upSampleImgs[i], Size(downSampleImgs[i + 1].cols * 2, downSampleImgs[i + 1].rows * 2));
        cuda::pyrDown(downSampleImgs[i], downSampleImgs[i + 1]);
        cuda::pyrUp(downSampleImgs[i + 1], upSampleImgs[i]);

        cv::cuda::GpuMat add_values;
//        double scale = 1;

        if (i != level - 1) {
            cuda::subtract(downSampleImgs[i], upSampleImgs[i], contrastPyramid[i]);
            cuda::add(upSampleImgs[i], bias, add_values);
            cuda::divide(1, add_values, add_values);
//            cuda::divide( add_values, 1, add_values);
            cuda::multiply(contrastPyramid[i], add_values, contrastPyramid[i]);
            cuda::multiply(255, contrastPyramid[i], contrastPyramid[i]);
//		    contrastPyramid[i] = 255 * ((downSampleImgs[i] - upSampleImgs[i]).mul(1 / (upSampleImgs[i] + bias)));
        } else {
            cuda::add(upSampleImgs[i], bias, add_values);
            cuda::divide(downSampleImgs[i] , add_values, downSampleImgs[i] );
//            cuda::divide(add_values, downSampleImgs[i], downSampleImgs[i] );
            cuda::multiply(255, downSampleImgs[i], contrastPyramid[i]);
//			contrastPyramid[i] = 255 * (downSampleImgs[i] / (upSampleImgs[i] + bias));
        }
//        contrastPyramid[i].convertTo(contrastPyramid[i], CV_8UC1 , 255);
    }

    return contrastPyramid;
}

cv::cuda::GpuMat contrastPyramidReconstruction(std::vector<cv::cuda::GpuMat> contrastPyramid) {
    int levels = contrastPyramid.size();
    cv::cuda::GpuMat outputImage = contrastPyramid[levels - 1];
    for (int i = levels - 1; i > 0; i--) {
        cv::cuda::GpuMat temp;
        cv::cuda::pyrUp(outputImage, temp);
//		cv::cuda::pyrUp(outputImage, temp, Size(outputImage.cols * 2, outputImage.rows * 2));
        cv::cuda::add(temp, contrastPyramid[i -1], outputImage);
//		outputImage = temp + contrastPyramid[i - 1];
    }
//    outputImage.convertTo(outputImage, CV_8UC1 , 255);
    return outputImage;
}


//cv::cuda::GpuMat enhanceUWT(cv::cuda::GpuMat grayImage) {

//    int nplanes = 3;
//    // Filtering operation
//    std::vector<float> f{ 0.0625f, 0.250f, 0.375f, 0.250f, 0.0625f };
//    //std::vector<float> f{ 0.25, 0.5, 0.25 };

//    cv::Mat K1(1, f.size(), CV_32F);
//    cv::Mat K1_trans;
//    std::vector<cv::Mat> kernel_vector;

//    // convert the std::vector cv::mat for easy mathematical operations
//    for (uint i = 0; i < f.size(); ++i) {
//        K1.col(i) = f[i];
//    }

//    for (int j = 0; j < nplanes; ++j) {
//        // Filtering
//        cv::transpose(K1, K1_trans);
////        K2 = K1_trans * K1;
////		kernel_vector.push_back(K2);
//        kernel_vector.push_back(K1);
//        int s = K1.cols, index;
//        index = 2 * (j + 1);
//        cv::Mat K3 = cv::Mat(1, 2 * s - 1, CV_32F, double(0));
//        for (int k = 0; k < 2 * s; k += index) {
//            K3.col(k) = f[k / index];
//        }
//        K1.release();
//        K1 = K3;
//    }
//    grayImage.convertTo(grayImage, CV_32FC1, 1.f / 255);
////	double meanInput = cv::mean(grayImage).val[0];
//    cv::Scalar mean, stddev, mean1, stddev1;
//    cuda::meanStdDev(grayImage, mean, stddev);
//    double meanInput = mean.val[0];
//    std::vector<cv::cuda::GpuMat> LP_vector(nplanes + 1), contrastPyramid(nplanes);
//    LP_vector[0] = grayImage;
//    for (int i = 0; i < nplanes; i++) {
////		cv::cuda::transpose(kernel_vector[i], K1_trans);
//        cv::transpose(kernel_vector[i], K1_trans);
//        cv::Ptr<cv::cuda::Filter> filter = cv::cuda::createSeparableLinearFilter(CV_32FC1, CV_32FC1, kernel_vector[i], K1_trans, Point(-1, -1), cv::BORDER_DEFAULT, -1);
//        filter->apply(LP_vector[i], LP_vector[i + 1]);
////		cv::cuda::sepFilter2D(LP_vector[i], LP_vector[i + 1], -1, kernel_vector[i], K1_trans, Point(-1, -1), cv::BORDER_DEFAULT, -1);

//        if (i != nplanes - 1) {
//            cv::cuda::GpuMat temp1;
////			contrastPyramid[i] = (LP_vector[i] - LP_vector[i + 1]).mul(1 / (LP_vector[i + 1] + std::max(0.1*255, meanInput)));
//            cuda::subtract(LP_vector[i], LP_vector[i + 1], contrastPyramid[i] );
//            cuda::add(LP_vector[i + 1], std::max(0.1*255, meanInput), temp1);
//            cuda::divide(1, temp1, temp1);
//            cuda::multiply(contrastPyramid[i], temp1, contrastPyramid[i]);
//        } else	{
//            //contrastPyramid[i] = downSampleImgs[i];
//            cv::cuda::GpuMat temp;
////			cv::cuda::exp(-LP_vector[i + 1] / (std::max(0.05, meanInput)), temp);
//            cv::cuda::divide(std::max(0.05, meanInput),LP_vector[i + 1], LP_vector[i + 1]);
//            cv::cuda::multiply(LP_vector[i + 1], -1, LP_vector[i + 1]);
//            cv::cuda::exp(LP_vector[i + 1], temp);

////			contrastPyramid[i] = LP_vector[i].mul(1 / (LP_vector[i + 1] + 0.3*temp));
//            cuda::multiply(temp, 0.3, temp);
//            cuda::add(LP_vector[i + 1], temp, temp);
//            cuda::divide(1,temp, temp );
//            cuda::multiply(LP_vector[i], temp, contrastPyramid[i]);


//            //contrastPyramid[i] = LP_vector[i].mul(1 / (LP_vector[i + 1] + 0.1));
////			float factor = std::min(1.0, 2 * meanInput / cv::mean(contrastPyramid[i]).val[0]);
//            cv::cuda::meanStdDev(contrastPyramid[i], mean1,stddev1);
//            float factor = std::min(1.0, (2 * meanInput) / mean1.val[0]);

////			contrastPyramid[i] = contrastPyramid[i] * factor;
//            cuda::multiply(contrastPyramid[i], factor, contrastPyramid[i]);
//        }

//    }
//    cv::cuda::GpuMat out;
//    cuda::add(contrastPyramid[0], contrastPyramid[1],out);
//    cuda::add(out, contrastPyramid[2], out);
//    out.convertTo(out, CV_8UC1, 255);
//    return out;
//}

//std::vector<cv::cuda::GpuMat> cropImage(cv::cuda::GpuMat image) {

//    cv::Rect TopLeft(0, 0, image.cols / 2, image.rows / 2);
//    cv::Rect TopRight((image.cols / 2) - 1, 0, image.cols / 2, image.rows / 2);
//    cv::Rect BottomLeft(0, (image.rows / 2) - 1, image.cols / 2, image.rows / 2);
//    cv::Rect BottomRight((image.cols / 2) - 1, (image.rows / 2) - 1, image.cols / 2, image.rows / 2);

//    std::vector<cv::cuda::GpuMat> images;
//    images.push_back(image(TopLeft));
//    images.push_back(image(TopRight));
//    images.push_back(image(BottomLeft));
//    images.push_back(image(BottomRight));

//    return images;
//}

//cv::cuda::GpuMat mergeImages (std::vector<cv::cuda::GpuMat> images) {
//    cv::cuda::GpuMat outImg;
////	cv::Mat img, row1, row2,outImg;
////	std::vector<cv::Mat> images_new;
////	for (int i = 0; i < images.size(); ++i) {
////	    images[i].download(img);
////	    images_new.push_back(img);
////	}

//    int c = images[1].rows;
//    int r = images[1].cols;

//    cv::cuda::resize (outImg, outImg, Size(2 * r  , 2 * c), 0, 0, INTER_CUBIC );

//    images[0].copyTo(outImg(cv::Rect(0,0,r-1,c-1)));
//    images[1].copyTo(outImg(cv::Rect(r,0,2*(r-1), c-1)));
//    images[2].copyTo(outImg(cv::Rect(0,c,r-1, 2*(c-1))));
//    images[3].copyTo(outImg(cv::Rect(r,c,2*(r-1), 2*(c-1))));

////	cv::hconcat(images_new[0], images_new[1], row1);
////	cv::hconcat(images_new[2], images_new[3], row2);

////	cv::vconcat(row1, row2, outImg);
//    return outImg;
//}


