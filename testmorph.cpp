#include <opencv2/opencv.hpp>
#include "lcudamorph.h"
#include "lcudatypes.h"

#define BLOCKSIZE 64
#define BLOCKCOUNT 16

lcuda8u streldata[13] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
lcudaSize strelsizes[] = {{13,1}};
int strelbinary[] = {0};

static int getBlockSize()
{
	return BLOCKSIZE;
}

static int getBlockCount()
{
	return BLOCKCOUNT;
}

static cv::Mat generateBoard(int bs, int count)
{
	int ims = bs * count;
	cv::Mat retval(ims, ims, CV_8UC1, cv::Scalar::all(0));
	unsigned char color = 0;

	for(int i = 0; i < ims; i = i + bs){
		color=~color;
		for(int j = 0; j < ims; j = j + bs){
			cv::Mat ROI = retval(cv::Rect(i, j, bs, bs));
			ROI.setTo(cv::Scalar::all(color));
			color=~color;
		}
	}
	return retval;
}


int main()
{
	lcuda8u *indata, *outdata;
	lcudaMatrix_8u inmatrix, outmatrix;
	lcudaStrel_8u strel;
	lcudaArray_8u strelarr;

	int side = getBlockSize() * getBlockCount();

	cv::Size insize;
	cv::Mat board = generateBoard(getBlockSize(), getBlockCount());
	cv::Mat output(side, side, CV_8UC1);

	cv::imwrite("input.bmp", board);

	indata = (lcuda8u *) board.ptr();

	insize = board.size();
	inmatrix = lcudaAllocMatrix_8u(insize.width, insize.height);
	lcudaCpyToMatrix_8u(indata, inmatrix);

	strelarr = lcudaAllocArray_8u(13);
	strel.data = strelarr;
	strel.sizes = strelsizes;
	strel.numStrels = 1;
	strel.isFlat = 1;
	strel.binary = strelbinary;

	outmatrix = lcudaAllocMatrix_8u(insize.width, insize.height);
	lcudaDilate_8u(inmatrix, outmatrix, strel);

	outdata = output.ptr();
	lcudaCpyFromMatrix_8u(outmatrix, outdata);

	cv::imwrite("output.bmp", output);

	lcudaFreeArray_8u(strelarr);
	lcudaFreeMatrix_8u(inmatrix);
	lcudaFreeMatrix_8u(outmatrix);
	return 0;
}
