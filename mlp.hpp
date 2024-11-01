#ifndef MLP_HPP
#define MLP_HPP

//DEFINES
#define inLength 4
#define hidLength 5
#define outLength 3
#define trainFile "train/train_images3.txt"
#define testFile "test/test_images3.txt"
#define learningRate 0.1
#define threshold 0.01

//FUNCTIONS
int countLines(FILE* file);

//CLASSES
class mlp{
private:
	float matH[hidLength][inLength+1]; //pesos+bias de cada neuronio da camada H em uma linha
	float matO[outLength][hidLength+1]; //pesos+bias de cada neuronio da camada O em uma linha
	float outResult[outLength]; //resultados obtidos em cada neuronio da camada O (apos aplicar funcao de ativacao)
	float hidResult[hidLength]; //resultados obtidos em cada neuronio da camada H (apos aplicar funcao de ativacao)
public:
	mlp();
	void printResult();
	float activFunc(float z);
	float activFuncDeriv(float z);
	void forward(float* inVector);
	void backpropagation(float X[][inLength], float Y[][outLength], int qtTrainCases);
};

#endif