#include <opencv2/opencv.hpp>
#include <iostream>

// Função para aplicar a convolução
cv::Mat convolve(const cv::Mat& image, const cv::Mat& kernel) {
    cv::Mat result;
    cv::filter2D(image, result, -1, kernel, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
    return result;
}

// Função para calcular o Laplaciano da imagem
cv::Mat computeLaplacian(const cv::Mat& image) {
    cv::Mat laplacian;
    cv::Laplacian(image, laplacian, CV_32F, 1, 1, 0, cv::BORDER_CONSTANT);
    return laplacian;
}

// Função principal para deblurring
cv::Mat deblurImage(const cv::Mat& blurred, const cv::Mat& kernel, float lambda, int iterations, float learning_rate) {
    // Inicialize a imagem limpa como a imagem borrada
    cv::Mat clean = blurred.clone();
    clean.convertTo(clean, CV_32F);
    
    // Garantir que o kernel esteja no formato adequado
    cv::Mat kernelFlipped;
    kernel.convertTo(kernel, CV_32F);
    cv::flip(kernel, kernelFlipped, -1);  // Inverte o kernel para a convolução inversa

    for (int iter = 0; iter < iterations; ++iter) {
        // Convolução de G * F
        cv::Mat blurredEstimate = convolve(clean, kernel);
        cv::normalize(blurredEstimate, blurredEstimate, 0, 255, cv::NORM_MINMAX);

        // Gradiente da função de erro
        cv::Mat grad = convolve(blurredEstimate - blurred, kernelFlipped);

        // Regularização Laplaciana
        cv::Mat laplacian = computeLaplacian(clean);

        // Atualize a imagem limpa
        clean -= learning_rate * (grad + lambda * laplacian);
    }

    // Converta a imagem de volta para o formato original
    clean.convertTo(clean, blurred.type());
    cv::normalize(clean,clean,0,255,cv::NORM_MINMAX);
    return clean;
}

int main() {
    // Carregar a imagem limpa (em tons de cinza)
    cv::Mat clean = cv::imread("lena.png", cv::IMREAD_GRAYSCALE);
    if (clean.empty()) {
        std::cerr << "Erro ao carregar a imagem limpa!" << std::endl;
        return -1;
    }

    clean.convertTo(clean, CV_32F);

    // Criar a máscara (kernel de borramento) - pode ser personalizada
    int kernelSize = 5;
    cv::Mat mask = cv::getGaussianKernel(kernelSize, -1, CV_32F) *
                     cv::getGaussianKernel(kernelSize, -1, CV_32F).t();

    // Aplicar o borramento sintético
    cv::Mat blurred = convolve(clean, mask);
    cv::normalize(blurred, blurred, 0, 255, cv::NORM_MINMAX);

    // Parâmetros do algoritmo
    float lambda = 0.01;          // Fator de regularização
    int iterations = 700;        // Número de iterações
    float learning_rate = 0.1f;  // Taxa de aprendizado

    // Recuperar a imagem limpa a partir da imagem borrada
    cv::Mat recovered = deblurImage(blurred, mask, lambda, iterations, learning_rate);
    

    // Convertendo para uchar para exibir    
    recovered.convertTo(recovered,CV_8U);
    clean.convertTo(clean, CV_8U);
    blurred.convertTo(blurred, CV_8U);  // Convertendo para tipo adequado para exibição
    // Salvar e exibir as imagens
    cv::imwrite("recovered_image.jpg", recovered);
    cv::imshow("Clean Image", clean);
    cv::imshow("Blurred Image", blurred);
    cv::imshow("Recovered Image", recovered);
    cv::waitKey(0);

    return 0;
}

