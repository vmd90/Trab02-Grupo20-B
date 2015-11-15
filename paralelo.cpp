#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <omp.h>
#include <mpi.h>

#define MASTER_TO_SLAVE_TAG  1
#define SLAVE_TO_MASTER_TAG  10

double t1, t2;

void smooth(const cv::Mat& img, cv::Mat& out) {
    int x, y;
    float n = 1.0f / 25.0f;
    float K[5][5];
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            K[i][j] = n;
        }
    }

    out.create(img.rows, img.cols, img.type());

#pragma omp parallel for private(y) 
    for (x = 1; x < img.rows - 4; ++x) {
        for (y = 1; y < img.cols - 4; ++y) {
            int p = (K[0][0] * img.at<uchar>(x - 2, y - 2)) + (K[0][1] * img.at<uchar>(x - 1, y - 2)) + (K[0][2] * img.at<uchar>(x, y - 2)) + (K[0][3] * img.at<uchar>(x + 1, y - 2)) + (K[0][4] * img.at<uchar>(x + 2, y - 2)) +
                    (K[1][0] * img.at<uchar>(x - 2, y - 1)) + (K[1][1] * img.at<uchar>(x - 1, y - 1)) + (K[1][2] * img.at<uchar>(x, y)) + (K[1][3] * img.at<uchar>(x + 1, y - 1)) + (K[1][4] * img.at<uchar>(x + 2, y - 1)) +
                    (K[2][0] * img.at<uchar>(x - 2, y)) + (K[2][1] * img.at<uchar>(x - 1, y)) + (K[2][2] * img.at<uchar>(x, y)) + (K[2][3] * img.at<uchar>(x + 1, y)) + (K[2][4] * img.at<uchar>(x + 2, y)) +
                    (K[3][0] * img.at<uchar>(x - 2, y + 1)) + (K[3][1] * img.at<uchar>(x - 1, y + 1)) + (K[3][2] * img.at<uchar>(x, y + 1)) + (K[3][3] * img.at<uchar>(x + 1, y + 1)) + (K[3][4] * img.at<uchar>(x + 2, y + 1)) +
                    (K[4][0] * img.at<uchar>(x - 2, y + 2)) + (K[4][1] * img.at<uchar>(x - 1, y + 2)) + (K[4][2] * img.at<uchar>(x, y + 2)) + (K[4][3] * img.at<uchar>(x + 1, y + 2)) + (K[4][4] * img.at<uchar>(x + 2, y + 2));

            out.at<uchar>(x, y) = p;
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Falta o argumento da imagem." << std::endl;
        return 1;
    }

    int N = omp_get_max_threads();
    //std::cout << "Num threads: " << N;
    omp_set_num_threads(N);

    cv::Mat img;
    std::vector<cv::Mat> channels;
    std::vector<uchar> buffer;

    //------------------ Inicia sessao MPI
    MPI_Status status;
    int rank, numtasks, tag = 1;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    if (rank == 0) {
        img = cv::imread(argv[1], CV_LOAD_IMAGE_UNCHANGED);
        if (!img.data) {
            std::cerr << "Erro ao abrir imagem." << std::endl;
            return 1;
        }
        int rows = img.rows;
        int cols = img.cols;

        channels.resize(img.channels()); // canais de cor (RGB) ou Grayscale
        cv::split(img, channels);
        buffer.resize(img.rows*img.cols);
        
        //t1 = time(NULL);
        t1 = MPI_Wtime();

        // Enviando o tamanho da imagem (vetor) e a imagem
        for (int i = 0; i < channels.size(); ++i) {
            int type = channels[i].type();
            memcpy(&buffer[0], channels[i].data, rows*cols*sizeof(uchar));

            std::cout << "Processo 0 enviando para " << i + 1 << " tipo: " << type << std::endl;
            MPI_Send(&type, 1, MPI_INT, i + 1, MASTER_TO_SLAVE_TAG, MPI_COMM_WORLD);
            std::cout << "Processo 0 enviando para " << i + 1 << " rows: " << rows << std::endl;
            MPI_Send(&rows, 1, MPI_INT, i + 1, MASTER_TO_SLAVE_TAG+1, MPI_COMM_WORLD);
            std::cout << "Processo 0 enviando para " << i + 1 << " cols: " << cols << std::endl;
            MPI_Send(&cols, 1, MPI_INT, i + 1, MASTER_TO_SLAVE_TAG+2, MPI_COMM_WORLD);
            std::cout << "Processo 0 enviando para " << i + 1 << " data: " << channels[i].data[0] << std::endl;
            MPI_Send(&buffer[0], rows*cols, MPI_UNSIGNED_CHAR, i + 1, MASTER_TO_SLAVE_TAG+3, MPI_COMM_WORLD);
        }

        std::vector<cv::Mat> ch;

        ch.resize(img.channels());
        for (int i = 0; i < img.channels(); ++i) {
            MPI_Recv(&buffer[0], rows*cols, MPI_UNSIGNED_CHAR, i + 1, SLAVE_TO_MASTER_TAG, MPI_COMM_WORLD, &status);
            std::cout << "Processo 0 recebendo de "<< i+1 << std::endl;
            ch[i] = cv::Mat(rows, cols, channels[i].type(), &buffer[0]);
        }

        //t2 = time(NULL);
        t2 = MPI_Wtime();
        
        std::cout << "Tempo: " << t2 - t1 << "s" << std::endl;

        cv::Mat m;
        cv::merge(ch, m);
        cv::imwrite(argv[2], m);

    } else {
        std::vector<uchar> v_out;
        int rows, cols, type;

        MPI_Recv(&type, 1, MPI_INT, 0, MASTER_TO_SLAVE_TAG, MPI_COMM_WORLD, &status);
        std::cout << "Processo " << rank << " recebendo tipo: " << type << std::endl;
        MPI_Recv(&rows, 1, MPI_INT, 0, MASTER_TO_SLAVE_TAG+1, MPI_COMM_WORLD, &status);
        std::cout << "Processo " << rank << " recebendo rows: " << rows << std::endl;
        MPI_Recv(&cols, 1, MPI_INT, 0, MASTER_TO_SLAVE_TAG+2, MPI_COMM_WORLD, &status);
        std::cout << "Processo " << rank << " recebendo cols: " << cols << std::endl;

        v_out.resize(rows * cols);
        MPI_Recv(&v_out[0], rows*cols, MPI_UNSIGNED_CHAR, 0, MASTER_TO_SLAVE_TAG+3, MPI_COMM_WORLD, &status);
        std::cout << "Processo " << rank << " recebendo data: " << v_out[0] << std::endl;

        cv::Mat in_img(rows, cols, type, &v_out[0]);
        cv::Mat out_img;
        smooth(in_img, out_img);

        std::cout << "Processo " << rank << " completou smooth, enviando resultado" << std::endl;
        // Enviando resultado para processo 0
        MPI_Send(&out_img.data[0], rows*cols, MPI_UNSIGNED_CHAR, 0, SLAVE_TO_MASTER_TAG, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    //-------------------- Fim do MPI
    return 0;
}
