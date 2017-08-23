/************************************************************************
* Copyright(c) 2012  Yang Xian
* All rights reserved.
*
* File:	main.cpp
* Brief: ¶Ô±ÈORBºÍFREAKÌØÕ÷ÃèÊö×ÓµÄËã·¨Ð§¹û£¬»ùÓÚOpenCV2.4.2
* Version: 1.0
* Author: Yang Xian
* Email: yang_xian521@163.com
* Date:	2012/07/10
* History:
************************************************************************/
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>

using namespace cv;
using namespace std;

int main(void)
{
    // ORB
    Mat img_1_Orb = imread("/Users/lan/Desktop/TarReg/ORB/ORB_FREAK/C3_A1_P1_D12.jpg");
    Mat img_2_Orb = imread("/Users/lan/Desktop/TarReg/ORB/ORB_FREAK/C3_A1_P1_D13.jpg");

    ORB orb;
    vector<KeyPoint> keyPoints_1_Orb, keyPoints_2_Orb;
    Mat descriptors_1_Orb, descriptors_2_Orb;

    double t = (double)getTickCount();
    orb(img_1_Orb, Mat(), keyPoints_1_Orb, descriptors_1_Orb);
    orb(img_2_Orb, Mat(), keyPoints_2_Orb, descriptors_2_Orb);
    t = 1000 * ((double)getTickCount() - t)/getTickFrequency();
    cout << "ORB time [ms]: " << t << endl;

    BruteForceMatcher<Hamming> matcher_Orb;
    vector<DMatch> matches_Orb;
    matcher_Orb.match(descriptors_1_Orb, descriptors_2_Orb, matches_Orb);

    double max_dist = 0;
    double min_dist = 100;
    //-- Quick calculation of max and min distances between keypoints
    for (int i=0; i<descriptors_1_Orb.rows; i++)
    {
        double dist = matches_Orb[i].distance;
        if (dist < min_dist) min_dist = dist;
        if(dist > max_dist) max_dist = dist;
    }
    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);
    //-- Draw only "good" matches (i.e. whose distance is less than 0.6*max_dist )
    //-- PS.- radiusMatch can also be used here.
    vector<DMatch> good_matches_Orb;
    for (int i=0; i<descriptors_1_Orb.rows; i++)
    {
        if(matches_Orb[i].distance < 0.15*max_dist)
        {
            good_matches_Orb.push_back(matches_Orb[i]);
        }
    }

    Mat img_matches_Orb;
    drawMatches(img_1_Orb, keyPoints_1_Orb, img_2_Orb, keyPoints_2_Orb,
                good_matches_Orb, img_matches_Orb, Scalar::all(-1), Scalar::all(-1),
                vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
    // FREAK
    Mat imgA_Freak = imread("/Users/lan/Desktop/TarReg/ORB/ORB_FREAK/C3_A1_P1_D12.jpg");
    Mat imgB_Freak = imread("/Users/lan/Desktop/TarReg/ORB/ORB_FREAK/C3_A1_P1_D13.jpg");
    vector<KeyPoint> keypointsA_Freak, keypointsB_Freak;
    Mat descriptorsA_Freak, descriptorsB_Freak;
    vector<DMatch> matches_Freak;

    // DETECTION
    // Any openCV detector such as
    SurfFeatureDetector detector_Freak(2000,4);

    // DESCRIPTOR
    // Our proposed FREAK descriptor
    // (roation invariance, scale invariance, pattern radius corresponding to SMALLEST_KP_SIZE,
    // number of octaves, optional vector containing the selected pairs)
    // FREAK extractor(true, true, 22, 4, std::vector<int>());
    FREAK freak;

    // MATCHER
    // The standard Hamming distance can be used such as
    // BruteForceMatcher<Hamming> matcher;
    // or the proposed cascade of hamming distance using SSSE3
    BruteForceMatcher<Hamming> matcher_Freak;

    // detect
    t = (double)getTickCount();
    detector_Freak.detect(imgA_Freak, keypointsA_Freak);
    detector_Freak.detect(imgB_Freak, keypointsB_Freak);
    t = 1000 * ((double)getTickCount() - t)/getTickFrequency();
    cout << "FREAK detection time [ms]: " << t/1.0 << endl;

    // extract
    t = (double)getTickCount();
    freak.compute(imgA_Freak, keypointsA_Freak, descriptorsA_Freak);
    freak.compute(imgB_Freak, keypointsB_Freak, descriptorsB_Freak);
    t = 1000 * ((double)getTickCount() - t)/getTickFrequency();
    cout << "FREAK extraction time [ms]: " << t << endl;

    // match
// 	t = (double)getTickCount();
    matcher_Freak.match(descriptorsA_Freak, descriptorsB_Freak, matches_Freak);
// 	t = ((double)getTickCount() - t)/getTickFrequency();
// 	std::cout << "matching time [s]: " << t << std::endl;

    max_dist = 0;
    min_dist = 100;
    //-- Quick calculation of max and min distances between keypoints
    for (int i=0; i<descriptorsA_Freak.rows; i++)
    {
        double dist = matches_Freak[i].distance;
        if (dist < min_dist) min_dist = dist;
        if(dist > max_dist) max_dist = dist;
    }
    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);
    //-- Draw only "good" matches (i.e. whose distance is less than 0.7*max_dist )
    //-- PS.- radiusMatch can also be used here.
    vector<DMatch> good_matches_Freak;
    for (int i=0; i<descriptorsA_Freak.rows; i++)
    {
        if(matches_Freak[i].distance < 0.7*max_dist)
        {
            good_matches_Freak.push_back(matches_Freak[i]);
        }
    }

    // Draw matches
    Mat imgMatch_Freak;
    drawMatches(imgA_Freak, keypointsA_Freak, imgB_Freak, keypointsB_Freak, good_matches_Freak, imgMatch_Freak,
                Scalar::all(-1), Scalar::all(-1),
                vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    // display
    imshow("MatchORB", img_matches_Orb);
    imshow("matchFREAK", imgMatch_Freak);

    waitKey(0);
    return 0;
}