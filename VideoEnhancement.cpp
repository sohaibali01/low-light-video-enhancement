#include <iostream>
#include <vector>
#include <string>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
//#include <opencv2/core/eigen.hpp>

using namespace cv;
using namespace std;

std::vector<cv::Mat> contrastPyramidDecomposition(cv::Mat grayImage, const int level);
cv::Mat contrastPyramidReconstruction(std::vector<cv::Mat> contrastPyramid);
cv::Mat enhanceUWT(cv::Mat grayImage);

int main() {
	
	cv::Mat img;
    cv::VideoCapture inputVideo("input.avi"); // open the default camera
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

    for(int no = 0; no < frnb; ++no) {
    	bool success = inputVideo.read(img);
    	if (!success) {
    	    std::cout << "Cannot read  frame " << std::endl;
    	    break;
    	}

		int level = 3;
		cv::Mat rgbImage, hsvImage, grayImage, out_image, frame;
		std::vector<cv::Mat> inputChannels;

		cv::cvtColor(img, hsvImage, cv::COLOR_BGR2HSV);

		double e1 = cv::getTickCount();
		cv::split(hsvImage, inputChannels);

		grayImage = inputChannels[2];

		std::vector<cv::Mat> contrastPyramid = contrastPyramidDecomposition(grayImage, level);
		cv::Mat outImage =  contrastPyramidReconstruction(contrastPyramid);

		inputChannels[2] = outImage;
		cv::merge(inputChannels, hsvImage);
		cv::cvtColor(hsvImage, rgbImage, cv::COLOR_HSV2BGR);

		double e2 = cv::getTickCount();
		double e = (e2 - e1) / cv::getTickFrequency();
		execution_time +=e;

		outputVideo.write(rgbImage);
    }
//	double e2 = cv::getTickCount();
//	double execution_time = (e2 - e1) / cv::getTickFrequency();
	std::cout << "\n No of frames per seconds : " << frnb/execution_time << std::endl;

	inputVideo.release();
	outputVideo.release();
	return 0;
}

cv::Mat enhanceUWT(cv::Mat grayImage){

	int nplanes = 3;

	// Filtering operation
	std::vector<float> f{ 0.0625, 0.25, 0.375, 0.25, 0.0625 };
	//std::vector<float> f{ 0.25, 0.5, 0.25 };
	cv::Mat K1(1, f.size(), CV_32F);
	cv::Mat K2, K1_trans;
	std::vector<cv::Mat> kernel_vector;

	// convert the std::vector cv::mat for easy mathematical operations
	for (uint i = 0; i < f.size(); ++i) {
		K1.col(i) = f[i];
	}

	for (int j = 0; j < nplanes; ++j) {
		// Filtering
		cv::transpose(K1, K1_trans);
		K2 = K1_trans * K1;
		//kernel_vector.push_back(K2);
		kernel_vector.push_back(K1);
		int s = K1.cols, index;
		index = 2 * (j + 1);
		cv::Mat K3 = cv::Mat(1, 2 * s - 1, CV_32F, double(0));
		for (int k = 0; k < 2 * s; k += index) {
			K3.col(k) = f[k / index];
		}
		K1.release();
		K1 = K3;
	}
	//double e1 = cv::getTickCount();
	grayImage.convertTo(grayImage, CV_32FC1, 1.0 / 255);
	double meanInput = cv::mean(grayImage).val[0];
	std::vector<cv::Mat> LP_vector(nplanes + 1), contrastPyramid(nplanes);
	cv::Mat temp;
	LP_vector[0] = grayImage;
	for (int i = 0; i < nplanes; i++)
	{
		cv::transpose(kernel_vector[i], K1_trans);
		//cv::filter2D(LP_vector[i], LP_vector[i + 1], -1, kernel_vector[i], cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
		cv::sepFilter2D(LP_vector[i], LP_vector[i + 1], -1, kernel_vector[i], K1_trans, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

		////cout << kernel_vector[i] << endl;
		////cout << K1_trans << endl;
		////cv::filter2D(temp, LP_vector[i + 1], -1, K1_trans, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

		if (i != nplanes - 1)
		{
			contrastPyramid[i] = (LP_vector[i] - LP_vector[i + 1]).mul(1 / (LP_vector[i + 1] + std::max(0.1*255, meanInput)));
			//contrastPyramid[i] = downSampleImgs[i] - upSampleImgs[i];
		}
		else
		{
			//contrastPyramid[i] = downSampleImgs[i];
			cv::Mat temp;
			cv::exp(-LP_vector[i + 1] / (std::max(0.05, meanInput)), temp);
			contrastPyramid[i] = LP_vector[i].mul(1 / (LP_vector[i + 1] + 0.3*temp));
			//contrastPyramid[i] = LP_vector[i].mul(1 / (LP_vector[i + 1] + 0.1));
			float factor = std::min(1.0, 2 * meanInput / cv::mean(contrastPyramid[i]).val[0]);
			contrastPyramid[i] = contrastPyramid[i] * factor;
		}

	}
	cv::Mat out = contrastPyramid[0] + contrastPyramid[1] + contrastPyramid[2];
	out.convertTo(out, CV_8UC1, 255);
	//double e2 = cv::getTickCount();
	//cout  << i << endl;
	//cout << "time " << (e2 - e1) / cv::getTickFrequency();
	return out;
}

std::vector<cv::Mat> contrastPyramidDecomposition(cv::Mat grayImage, const int level){
	
	//grayImage.convertTo(grayImage, CV_32FC1, 1.0 / 255);
	int meanInput = cv::mean(grayImage).val[0];
	//cout << "mean " << meanInput << endl;
	std::vector<cv::Mat> contrastPyramid(level), downSampleImgs(level + 1), upSampleImgs(level);
	downSampleImgs[0] = grayImage;

	for (int i = 0; i < level; i++)
	{
		//double e1 = cv::getTickCount();


		if (i != level - 1)
		{
			pyrDown(downSampleImgs[i], downSampleImgs[i + 1], Size(downSampleImgs[i].cols / 2, downSampleImgs[i].rows / 2));
			pyrUp(downSampleImgs[i + 1], upSampleImgs[i], Size(downSampleImgs[i + 1].cols * 2, downSampleImgs[i + 1].rows * 2));
			contrastPyramid[i] = 255 * ((downSampleImgs[i] - upSampleImgs[i]).mul(1 / (upSampleImgs[i] + std::max(20, meanInput))));
			//contrastPyramid[i] = downSampleImgs[i] - upSampleImgs[i];
		}
		else
		{
			pyrDown(downSampleImgs[i], downSampleImgs[i + 1], Size(downSampleImgs[i].cols / 2, downSampleImgs[i].rows / 2));
			pyrUp(downSampleImgs[i + 1], upSampleImgs[i], Size(downSampleImgs[i + 1].cols * 2, downSampleImgs[i + 1].rows * 2));
			//contrastPyramid[i] = downSampleImgs[i];

			cv::Mat temp, tempUpSample;//, tempDownsample;
			//contrastPyramid[i].convertTo(contrastPyramid[i], CV_16UC1);
			downSampleImgs[i + 1].convertTo(downSampleImgs[i + 1], CV_32FC1, 1.0 / 255);
			//downSampleImgs[i].convertTo(downSampleImgs[i], CV_16UC1);
			cv::exp(-downSampleImgs[i + 1] / (std::max(0.05, (double)meanInput/255)), temp);
			temp = 0.3*temp;
			temp.convertTo(temp, CV_8UC1, 255);
			pyrUp(temp, tempUpSample, Size(temp.cols * 2, temp.rows * 2));
			contrastPyramid[i] = (255 * downSampleImgs[i]) / (upSampleImgs[i] + tempUpSample);
			float factor = std::min(1.0, 2 * ((double)meanInput) / cv::mean(contrastPyramid[i]).val[0]);
			//cout << factor << endl;
			contrastPyramid[i] = contrastPyramid[i] * factor;
			//contrastPyramid[i].convertTo(contrastPyramid[i], CV_8UC1);
		}

		double e2 = cv::getTickCount();
		//cout  << i << endl;
		//cout << "time " << (e2 - e1) / cv::getTickFrequency() << endl;

	}

	return contrastPyramid;
}

cv::Mat contrastPyramidReconstruction(std::vector<cv::Mat> contrastPyramid){
	//cv::Mat outputImage(contrastPyramid[0].rows, contrastPyramid[0].cols, CV_32FC1, float(0));
	int levels = contrastPyramid.size();
	cv::Mat outputImage = contrastPyramid[levels - 1].clone();
	//double e1 = cv::getTickCount();
	for (int i = levels - 1; i > 0; i--)
	{
		cv::Mat temp;
		pyrUp(outputImage, temp, Size(outputImage.cols * 2, outputImage.rows * 2));
		outputImage = temp + contrastPyramid[i - 1];
	}
	//double e2 = cv::getTickCount();
	//cout << "reconstruction time " << (e2 - e1) / cv::getTickFrequency() << endl;
	//outputImage.convertTo(outputImage, CV_8UC1, 255);

	return outputImage;
}

//cv::Mat lppyr (cv::Mat gray_image, const int level) {
//	int r =  gray_image.rows;
//	int c = gray_image.cols;
//	cv::Mat M1, G1, M1T, Y;
//	std::vector<cv::Mat> E;
//	gray_image.convertTo(M1, CV_64FC1, 1.0/255);
//	const float constant = 1.0/ 16;
//
//	std::vector <float> f{1, 4, 6, 4, 1};
//	std::vector <float> f2;
//	std::transform(f.begin(), f.end(), f.begin(), std::bind1st(std::multiplies<float>(), constant));
//	for (auto i = f.begin(); i != f.end(); ++i)
//	    std::cout << *i << ' ';
//
//	std::vector <int> zl;
//	std::vector <int> sl;
//
//	for (int i = 0; i < level; ++i) {
//		zl.push_back(r);
//		sl.push_back(c);
//
//		std::vector <int> ew;
//		if ( floor(r/2) != r/2 )
//			ew.push_back(1);
//		else
//			ew.push_back(0);
//
//		if ( floor(c/2) != c/2 )
//			ew.push_back(1);
//		else
//			ew.push_back(0);
//
//		if (ew[0] == 1 || ew[1] == 1)
//			M1 = abd(M1, ew);
//
//		// Perform filtering
//		M1 = es2(M1, 2);
//		cv::filter2D(M1, G1, -1, f, cv::Point(-1,-1),0, cv::BORDER_DEFAULT );
//		cv::filter2D(G1, G1, -1, f, cv::Point(-1,-1),0, cv::BORDER_DEFAULT );
//
//
//		std::transform(f.begin(), f.end(), f2.begin(), std::bind1st(std::multiplies<float>(), 2));
//		cv::filter2D(G1, M1T, -1, f2, cv::Point(-1,-1),0, cv::BORDER_DEFAULT );
//		cv::filter2D(M1T, M1T, -1, f2, cv::Point(-1,-1),0, cv::BORDER_DEFAULT );
//
//		M1 = dec2(G1);
//
//		E.push_back((M1 - M1T));
//		M1 = dec2(G1);
//	}
//
//	for (int i = level; i > 0; --i) {
//		M1 = es2(undec2(M1),2);
//		cv::filter2D(M1, M1T, -1, f2, cv::Point(-1,-1),0, cv::BORDER_DEFAULT );
//		cv::filter2D(M1T, M1T, -1, f2, cv::Point(-1,-1),0, cv::BORDER_DEFAULT );
//
//		M1 = M1T + E[i];
//
//		cv::Rect rect(0, 0, zl[i], sl[i]);
//		cv::resize(M1, M1(rect), cv::Size(rect.width, rect.height));
//	}
//
//	Y = M1;
//	return Y;
//}
//
//cv::Mat es2(cv::Mat gray_image, const int n ) {
//	int r =  gray_image.rows;
//	int c = gray_image.cols;
//	cv::Mat out_image = cv::Mat( (r + 2 * n), (c +  2 * n), CV_8U, double(0) );
//	gray_image.copyTo(out_image(Rect(Point(n, n), gray_image.size())));
//	std::cout << "\n Size of Gray Image is : " << out_image.size();
//
//	for (int i = 0; i < r; ++i ) {
//		for (int j = 0; j < n; ++j ) {
//			out_image.at<int>(n-1-j, n+i)   = gray_image.at<int>(j+1 , i );
//			out_image.at<int>(n+c+j, n+i) 	= gray_image.at<int>(c-1-j, i);
//		}
//	}
//
//	for (int i = 0; i < c; ++i ) {
//		for (int j = 0; j < n; ++j ) {
//			out_image.at<int>(n+i, n-1-j) = gray_image.at<int>(i, j+1  );
//			out_image.at<int>(n+i, n+r+j) = gray_image.at<int>(i, r-2-j);
//		}
//	}
//
//	return out_image;
//}
//
//cv::Mat undec2(cv::Mat gray_image) {
//	int r =  gray_image.rows;
//	int c = gray_image.cols;
//	cv::Mat out_image = cv::Mat( r * 2, c *2, CV_8U, double(0) );
//	std::cout << "\n Size of Out image is : " << out_image.size() ;
//	int k =0, l =0;
//	for ( int i = 0; i < r; ++i) {
//		for ( int j = 0; j < c; j++) {
//			out_image.at<int>(k, l) = gray_image.at<int>(j, i);
//			k += 2;
//		}
//		l +=2;
//		k = 0;
//	}
//	return out_image;
//}
//
//
//cv::Mat dec2(cv::Mat gray_image) {
//	int r =  gray_image.rows;
//	int c = gray_image.cols;
//	int r1 = ceil (r/2);
//	int c1 = ceil (c/2);
//	cv::Mat out_image = cv::Mat( r1, c1, CV_8U, double(0) );
//	std::cout << "\n Size of Out image is : " << out_image.size() ;
//	int k =0, l =0;
//	for ( int i = 0; i < r1; ++i) {
//		for ( int j = 0; j < c1; j++) {
//			out_image.at<int>(j, i) = gray_image.at<int>(k, l);
//			k += 2;
//		}
//		l +=2;
//		k = 0;
//	}
//	return out_image;
//}
//
//cv::Mat abd (cv::Mat gray_image, std::vector<int> bd ) {
//	int r =  gray_image.rows;
//	int c = gray_image.cols;
//	cv::Mat out_image = cv::Mat( (r + bd.at(0)), (c + bd.at(1)), CV_8U, double(0) );
//    gray_image.copyTo(out_image(Rect(Point(0, 0), gray_image.size())));
//
//    std::cout << "\n Size of Gray Image is : " << gray_image.size();
//    std::cout << "\n Size of Out image is : " << out_image.size() ;
//
//	for ( int i = 0; i < bd.at(0); ++i) {
//		out_image.row( r+i ) = gray_image.row(r-i-1);
//	}
//
//	for ( int i = 0; i < bd.at(1); ++i) {
//		out_image.col( c+i ) = gray_image.col(c-i-1);
//	}
//
//	for ( int i = 0; i < bd.at(0); ++i) {
//		for ( int j = 0; j < bd.at(1); ++j) {
//			out_image.at<int>(c+j, r+i) = gray_image.at<int>(c-j-1, r-i-1);
//		}
//	}
//	return out_image;
//}
