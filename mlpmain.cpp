#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <omp.h>
// BIBLIOTECA COM CLASSES E FUNÇÕES DO MLP
#include "mlp.hpp"
#pragma omp num_threads(4)
using namespace std;

int main()
{
    int i, j, qtTestCases = 0, qtTrainCases = 0;
    float vec[inLength] = {0};
    FILE *trainDataset, *testDataset;
    mlp mlp; // classe contendo os pesos, biases, resultados, informações e funções do MLP

    // Acessando arquivo contendo trainDataset
    trainDataset = fopen(trainFile, "r");
    if (trainDataset == NULL) {
        printf("error opening trainDataset file\n");
        return -1;
    }

    // Contando quantidade de linhas do trainDataset = quantidade de "casos treino" para o MLP
    qtTrainCases = countLines(trainDataset);

    // Matrizes de entradas e de saídas para treinar o MLP
    float X[qtTrainCases][inLength], Y[qtTrainCases][outLength];

    // Preenchendo matrizes com dados do trainDataset
    #pragma omp parallel for private(j) // Paralelização do loop sobre os casos de treino
    for (i = 0; i < qtTrainCases; i++) {
        for (j = 0; j < inLength; j++) {
            fscanf(trainDataset, "%f", &X[i][j]);
        }
        for (j = 0; j < outLength; j++) {
            fscanf(trainDataset, "%f", &Y[i][j]);
        }
    }
    fclose(trainDataset);

    // Treinando MLP (a função backpropagation já está paralelizada internamente)
    mlp.backpropagation(X, Y, qtTrainCases);

    // Testando MLP
    testDataset = fopen(testFile, "r");
    if (testDataset == NULL) {
        cout << "error opening testDataset file\n";
        return -1;
    }

    qtTestCases = countLines(testDataset);

    #pragma omp parallel for private(j, vec) // Paralelização do loop sobre os casos de teste
    for (i = 0; i < qtTestCases; i++) {
        for (j = 0; j < inLength; j++) {
            fscanf(testDataset, "%f", &vec[j]);
        }
        mlp.forward(vec);
        mlp.printResult();
    }
    fclose(testDataset);

    return 0;
}
