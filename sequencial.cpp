#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

void smooth(const cv::Mat& img, cv::Mat& out)
{
    float n = 1.0f/25.0f;
    float K[5][5];
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            K[i][j] = n;
        }
    }

    out.create(img.rows, img.cols, img.type());

    for(int x = 1; x < img.rows-4; ++x)
    {
        for(int y = 1; y < img.cols-4; ++y)
        {
            int p = (K[0][0] * img.at<uchar>(x-2,y-2)) + (K[0][1] * img.at<uchar>(x-1,y-2)) + (K[0][2] * img.at<uchar>(x,y-2)) + (K[0][3] * img.at<uchar>(x+1,y-2)) + (K[0][4] * img.at<uchar>(x+2,y-2)) +
                    (K[1][0] * img.at<uchar>(x-2,y-1)) + (K[1][1] * img.at<uchar>(x-1,y-1)) + (K[1][2] * img.at<uchar>(x,y))  + (K[1][3] * img.at<uchar>(x+1,y-1)) + (K[1][4] * img.at<uchar>(x+2,y-1)) +
                    (K[2][0] * img.at<uchar>(x-2,y))   + (K[2][1] * img.at<uchar>(x-1,y))   + (K[2][2] * img.at<uchar>(x,y))  + (K[2][3] * img.at<uchar>(x+1,y)) + (K[2][4] * img.at<uchar>(x+2,y)) +
                  (K[3][0] * img.at<uchar>(x-2,y+1)) + (K[3][1] * img.at<uchar>(x-1,y+1)) + (K[3][2] * img.at<uchar>(x,y+1)) + (K[3][3] * img.at<uchar>(x+1,y+1)) + (K[3][4] * img.at<uchar>(x+2,y+1)) +
                  (K[4][0] * img.at<uchar>(x-2,y+2)) + (K[4][1] * img.at<uchar>(x-1,y+2)) + (K[4][2] * img.at<uchar>(x,y+2)) + (K[4][3] * img.at<uchar>(x+1,y+2)) + (K[4][4] * img.at<uchar>(x+2,y+2));

            out.at<uchar>(x,y) = p;
        }
    }
}

int main(int argc, char *argv[])
{
    if(argc < 3) {
        std::cout << "Forma de executar:\n./init_sequencial img1 img2" << std::endl;
        std::cout << "\nonde img1 - imagem de entrada"
                     "\n     img2 - imagem de saida" << std::endl;
        return 1;
    }
    cv::Mat img = cv::imread(argv[1], CV_LOAD_IMAGE_UNCHANGED);
    if(!img.data) {
        std::cerr << "Erro ao abrir imagem." << std::endl;
        return 1;
    }

    std::vector<cv::Mat> ch(img.channels());
    cv::split(img, ch);

    std::vector<cv::Mat> out(img.channels());
    time_t t1, t2;
    t1 = time(NULL);
    for(int i = 0; i < ch.size(); ++i)
        smooth(ch[i], out[i]);
    t2 = time(NULL);

    std::cout << "\nTempo: " << t2 - t1 << " s" << std::endl;
    
    cv::Mat m;
    cv::merge(out, m);
    cv::imwrite(argv[2], m);
    
    return 0;
}
