import numpy
import pandas
import json
import matplotlib.pyplot as plt
'''
Componentes da rede:
Camadas e quantidade de neurônios
Inicialização aleatória de pesos e bias
Função de ativação dos neurônios
Função custo
Descida do gradiente por meio do algoritmo do backpropagation: Utilizou a derivada da cross-entropy function
Técnicas de decisão das camadas e neurônios: A maioria dos problemas demandam uma camada oculta e uma das regras para definir o número de neurônios é fazer a média entre a camada de entrada e de saída
'''
def Sigmoid(x): #Sua própria derivada é S(a)*(1-S(a)), como S(a) é o atributo output do neurônio fica o*(1-o), essa função de ativação pertence aos neurônios da camada oculta.
    return 1/(1+numpy.exp(-x)) #numpy.exp(var) é igual a e^var

def Softmax(x, Sum): #Sua própria derivada também é o*(1-o), essa função foi utilizada nos neurônios de saída. A variável sum é igual ao somatório de potências e^(an), onde an é a a ativação (a soma ponderada somada ao bias) de cada neurônio da saída, e x que é igual a e^aj em que aj é a ativação do neurônio correspondente.
    return numpy.exp(x)/Sum

def Cross_Entropy(y,Y,N,λ,Sum): #Função de erro da entropia cruzada regularizada (L2). Na parte do backwards propagation é usada a derivada do erro dessa função para determinar a influência dos pesos sendo importante na atualização deles.
    sum = numpy.sum([y*numpy.log(x)+(1-y)*numpy.log(1-x) for x, y in zip(y, Y)]) #Compressão de lista com os valores desejados representados pela lista Y, e os valores preditos representados por y. Cada elemento da lista será igual a Yj*ln(yj) + (1-Yj)*ln(1-yj) em que Yj e yj são as saídas esperadas para o neurônio j, numpy.sum(var) é a soma de todos os números em var.
    return ((-1)*(sum/N)+(λ/(2*N))*Sum) #N é o número de dados de treinamento do mini batch (mini lote), uma pequena porção de dados do conjunto de dados de treinamento total. Sum é a soma dos quadrados de todos os pesos da rede.

class Neuron(object): #Neuronio
    
    def __init__(self,VectorWeights = [], Bias = 0):
        self.weights = VectorWeights #Pesos que conectam o neurônio j da camada L cada neurônio da camada L-1. [W0j,W1j,...,Wnj]
        self.bias = Bias #Bias do neurônio
        self.output = 0 #Serve para o backpropagation, é a saída do neurônio.
        self.ErrorDerivative = numpy.zeros((len(VectorWeights)+1)) #Serve para guardar as derivadas parcias (no final de tudo as derivadas parciais em todos os neurônios representam o gradiente da função) dos pesos e bias já multiplicados por 1/N (N é a quantidada de dados no mini lote).
        self.weightsForward = [] #Pesos que conectam o neurônio j da camada L com cada neurônio da camada k+1 [Wj0,Wj1,Wj2,...,Wjn]. Esse termo é muito útil para agilizar o cálculo da derivada de todos os pesos do neurônio usando o produto escalar (vetorial).
        self.weigthsSquared = numpy.sum(numpy.array(VectorWeights)**2) #no numpy, um vetor A**n é [a0**n,a1**n,...,ai**n], numpy.sum faz a soma de todos os elementos. Isso é importante para a função de custo regularizada.

    def Operar(self,o): #Função do neurônio da camada oculta
        self.output = Sigmoid(numpy.dot(o, self.weights) + self.bias) #a é a ativação do neurônio j em que a é a soma entre produtos de cada uma das saídas dos neurônios da camada anterior com os pesos, ao valor do bias. a = W0j*O0 + W1j*O1 + ... + Wnj*On + bj. o numpy.dot() faz justamente essa operação entre os vetores weigths = [W0j,W1j,...,Wnj] e O = [O0,O1,...,On], sendo assim numpy.dot(a, weights) + b = W0j*O0 + ... + Wnj*Oj + b.
    
    def Saida(self, o, S): #Função do neurônio da camada de saída
        self.output = Softmax(numpy.dot(o, self.weights) + self.bias, S) #mesma lógica da função Operar, porém com o Softmax.

class Layer(object): #Camada

    def __init__(self,VectorNeurons):
        self.Neurons = VectorNeurons #Neurônios da camada
        self.QuantityNeurons = len(VectorNeurons) #Quantidade de neurônios
        self.Output = [] #Saída de cada neurônio da camada. Seu formato é [O0,O1,...,On]
        self.δ = [] #Guarda um termo para calcular a derivada do erro pelo peso Wij (conecta o neurônio i da camada k-1 ao j da camada k) para uso no backpropagation

class Network(object): #Rede Neural

    def __init__(self, Neurons = [], et = 0.1, Bool = False, lamb = 0): #Neurons é uma lista cujo o tamanho indica a quantidade de camadas e cada número indica a quantidade de neurônios cada neurônio têm o conjunto de pesos que os conectam com os neurônios da camada anterior (exceto a camada de entrada). et é a taxa de aprendizado, controla o quão rápido a rede aprende (quão grande é a alteração nos pesos e bias), lamb é a variável da regularização que controla o tamanho dos pesos da rede deixando-os menores e impedindo a rede de "memorizar" os dados de treinamento, Bool é um arquivo com pesos, bias e camadas já definidos para a rede.
        if Bool: #Bool é na realidade um arquivo Json com os pesos e biases salvos
            self.LoadWeightsBiases(Bool)
        else:
            self.InputLayer = Layer([Neuron() for var in range(Neurons[0])]) #Camada de entrada
            self.HiddenLayers = [Layer([Neuron(numpy.random.randn(Neurons[var-1]),numpy.random.randn()) for var1 in range(Neurons[var])]) for var in range(1,len(Neurons)-1)] #Camadas ocultas. numpy.random.randn() gera um número aletatório de uma distribuição gaussiana, quando existem um número n dentro da função ela cria um array com tamanho n e números aleatórios de uma distribuição gaussiana.
            self.OutputLayer = Layer([Neuron(numpy.random.randn(Neurons[-2]),numpy.random.randn()) for var in range(Neurons[-1])]) #Camada de saída
        self.eta = et #Taxa de aprendizado
        self.lamb = lamb #Lambda a variável que impede a 'memorização' da rede, pois limita os valores dos pesos, forçando-os a serem menores e assim tornando a rede menos sensível a qualquer entrada.
        self.weightSum = self.UpgradeForwardWeights() #Armazenar os pesos que conectam um neurônio j da camada k a cada neurônio da camada k+1 na ordem em que esses neurônios estão camada k+1 no vetor weights forward do neurônio j. Além disso, retorna o valor da soma dos quadrados de todos os pesos da rede.

    def UpgradeForwardWeights(self):
        Sum = 0 #Soma do quadrado dos pesos de toda a rede.
        for var in range(self.HiddenLayers[-1].QuantityNeurons): #Itera cada neurônio da última camada oculta
            IN = []
            for var1 in self.OutputLayer.Neurons: #Itera nos neurônios da camada de saída
                IN += [var1.weights[var]] #Constrói o array com o peso Wji que conecta o neurônio i da camada de saída com o neurõnio j da última camada oculta
                Sum += var1.weigthsSquared
            self.HiddenLayers[-1].Neurons[var].weightsForward = IN #Atribui esse vetor ao weigthsForward do neurônio j
        for var in range(len(self.HiddenLayers)-2,-1,-1): #Itera em cada camada oculta da penúltima para a última
            for var1 in range(self.HiddenLayers[var].QuantityNeurons): #Itera em cada neurônio da camada oculta L
                IN = []
                for var2 in self.HiddenLayers[var+1].Neurons: #Itera em cada neurônio da camada oculta L+1
                    IN += [var2.weights[var1]] #Constrói o array que conecta com os pesos Wji que conectam o neurônio j da camada L com o i da camada L+1
                    Sum += var2.weigthsSquared
                self.HiddenLayers[var].Neurons[var1].weightsForward = IN #Atribui o array ao weigthsForward
        return Sum

    def FeedForward(self, InputVector, Bool = False, N = 1): #Vetor de entrada e Vetor de neurônios da camada de entrada devem ter mesmo tamanho, o booleano ativa a mudança dos pesos durante o FeedForward
        self.InputLayer.Output = numpy.array(InputVector) #Atribui cada valor da imagem do número a saída de um neurônio da camada de entrada.
        Transfer = []
        for var in self.HiddenLayers[0].Neurons: #Itera em cada neurônio da primeira camada oculta
            if (Bool): # 1 Booleano que confirma se o a rede está em um novo mini-lote. Caso verdadeiro, haverá uma atualização dos pesos
                var.weights += (-1)*self.eta*var.ErrorDerivative[:-1] - (self.eta*(self.lamb/N))*var.weights #2 Essa atualização está sendo feita de uma forma vetorial. Individualmente, W = w - eta*derivada_do_erro_da_rede_no_peso_nos_N_dados_de_treinamento(dE(X,θ)/dwij) - eta*(lambda/tamanho_do_mini-lote)*w, em que W é o novo valor do peso, w é o antigo, eta é a taxa de aprendizado, lambda é a variável da regularização (impede da rede "memorizar" os dados de treinamento, a regularização aplicada foi a L2).
                var.bias += (-1)*self.eta*var.ErrorDerivative[-1] #3 A expressão para o bias é B = b - eta*derivada_do_bias_no_erro_da_rede_nos_N_dados(dE(X,θ)/dbj), B é o novo valor, b é o antigo valor do bias, eta é a taxa de aprendizagem. A derivada do bias, assim como a do peso, indicam o quão influente a grandeza é no erro da rede, quanto mais influente, mais alterado.
                var.ErrorDerivative = numpy.zeros(len(var.weights)+1) #4 Derivada de cada peso é reinicializada para 0.
            var.Operar(self.InputLayer.Output) #Neurônio da camada processa sua saída ao receber os valores de todos os outros da camada de entrada.
            Transfer += [var.output] #A saída do neurônio é incluída em um vetor.
        self.HiddenLayers[0].Output = numpy.array(Transfer) #No final dos cálculos esse vetor é atribuído para a saída da camada para o processamento continuar nas outras
        for var in range(1,len(self.HiddenLayers)): #Faz os calculos entra a camada L-1 e L em cada neurônio da L
            Transfer = []
            for var1 in self.HiddenLayers[var].Neurons: #Itera cada neurônio de cada camada oculta L para fazer a operação.
                if (Bool): #1
                    var1.weights += (-1)*self.eta*var1.ErrorDerivative[:-1] - (self.eta*(self.lamb/N))*var1.weights #2
                    var1.bias += (-1)*self.eta*var1.ErrorDerivative[-1] #3
                    var1.ErrorDerivative = numpy.zeros(len(var1.weights)+1) #4
                var1.Operar(self.HiddenLayers[var-1].Output) #Neurônio recebe a saída de todos os neurônios da camada anterior e processa.
                Transfer += [var1.output] #Saída de cada neurônio é armazenada em um vetor
            self.HiddenLayers[var].Output = numpy.array(Transfer) #Vetor é atribuído a variável Output da camada.
        Transfer = []
        Activations = numpy.sum([numpy.exp(numpy.dot(self.HiddenLayers[-1].Output, var.weights)+var.bias) for var in self.OutputLayer.Neurons]) #Faz a soma de todas as potências do tipo e^(aj) em que aj é a ativação (aj = W0j*O0 + ... + Wnj*On + b) do neurônio j da camada de saída.
        for var in self.OutputLayer.Neurons: #Itera nos neurônios da camada de saída.
            if (Bool): #1
                var.weights += (-1)*self.eta*var.ErrorDerivative[:-1] - (self.eta*(self.lamb/N))*var.weights #2
                var.bias += (-1)*self.eta*var.ErrorDerivative[-1] #3
                var.ErrorDerivative = numpy.zeros(len(var.weights)+1) #4
            var.Saida(self.HiddenLayers[-1].Output, Activations) #Envia a ativação do neurônio j (Output da última camada oculta) da camada de saída com aquela soma de exponenciações para usar a função Softmax
            Transfer += [var.output] #Saída de cada neurônio é armazenada em um vetor
        self.OutputLayer.Output = numpy.array(Transfer) #O vetor é atribuído ao Output da camada de saída
        if (Bool): #1
            self.Sum = self.UpgradeForwardWeights() #Como os pesos são atualizados, é necessário passar os seus novos valores para o atributo weigthForward de cada neurônio assim como a soma dos quadrados
                
    def Backwards_Propagation(self,y,Y,N): # Y é um vetor de saída desejada e y é um vetor de saída da rede.
        for var in range(self.OutputLayer.QuantityNeurons): #Calcula derivadas parciais dos pesos de cada neurônio da camada de saída.
            self.OutputLayer.δ += [(Y[var]-y[var])] #Calcula o valor δj do neurônio j da saída. Esse termo surge na derivada do erro em relação ao peso, não acho que consigo explicar toda a lógica do cálculo no código, mas um bom material é o da briliant sobre backpropagation, o deeplearning book em português (mostra o caso da função de entropia cruzada), além de alguns materias do khan academy para entender a matemática por trás disso (derivada parcial, regra da cadeia, função multivariável e gradiente). Em suma, por causa da regra da cadeia, a fórmula da derivada do erro para neurônios da saída é dE/dwij = δj * Oi (para cada caso de treinamento), em que Oi é a saída do neurônio i da última camada oculta, δj = dE/daj, no caso da função de entropia cruzada, δj = de/daj = (Yj-yj). Essa função foi escolhida porque a derivada dela envolve apenas o variação entre a saída desejada e a saída obtida e a saída do neurônio anterior, quando essa variação é muito grande, o valor de de/dwij é muito maior, o que corrige os erros com mais velocidade. (Eu comparei o MSE e a entropia cruzada, nas primeiras 10 épocas da MSE, épocas são iterações sobre os dados de treinamento a precisão evoluiu para 10%-12%, nas primeiras 10 épocas da entropia cruzada a precisão foi de 40%-50%).
            self.OutputLayer.Neurons[var].ErrorDerivative += (-1)*(1/N)*self.OutputLayer.δ[var]*numpy.array(list(self.HiddenLayers[-1].Output)+[1]) #Calcula dE(X,θ)/dwij para cada peso do neurônio (pois usa os vetores de pesos para fazer todas as multiplicações de uma só vez), e também calcula dE(X,θ)/dbj (por causa da concatenação da lista dos pesos e da lista com o valor 1, sim, dE/dbj = δj). E(X,θ) é a fórmula do erro para um conjunto de N dados (X) que usa E em cada exemplo de treinamento (O theta é o hiperparâmetro: um peso ou bias), ou seja a derivada de E(X,θ) depende da derivada de cada E individual para ambos pesos e bias, é o erro médio da rede em um conjunto N de dados, sendo assim, dE(X,θ)/dwij e dE(X,θ)/dwbj calcula a influência média dos pesos e bias sobre o erro da rede no pequeno conjunto (mini-lote) dos dados de treinamento.
        for var in range(len(self.HiddenLayers)-1,-1,-1): #Itera em cada camada oculta
            if (var == len(self.HiddenLayers)-2): #Esvaziar δ para uso posterior
                self.OutputLayer.δ = []
            elif (var < len(self.HiddenLayers)-2): #Esvaziar δ para uso posterior
                self.HiddenLayers[var+2].δ = []
            for var1 in range(self.HiddenLayers[var].QuantityNeurons): #Itera em cada neurônio de cada camada oculta
                if (var == 0 and len(self.HiddenLayers) == 1): #Caso a camada seja a primeira e a última camada da rede
                    self.HiddenLayers[var].δ += [(numpy.dot(self.OutputLayer.δ, self.HiddenLayers[var].Neurons[var1].weightsForward))*(self.HiddenLayers[var].Neurons[var1].output*(1-self.HiddenLayers[var].Neurons[var1].output))] #[A] No caso da camada oculta ser a primeira e a última. O cálculo do δj vai envolver o δ de cada neurônio da camada de saída.
                    self.HiddenLayers[var].Neurons[var1].ErrorDerivative += ((-1)*(1/N)*self.HiddenLayers[var].δ[var1])*numpy.array(list(self.InputLayer.Output)+[1]) #[B] #No caso da saída, vai ser preciso acessar a camada de entrada.
                elif (var == len(self.HiddenLayers)-1): #Caso a camada seja a última da rede.
                    self.HiddenLayers[var].δ += [(numpy.dot(self.OutputLayer.δ, self.HiddenLayers[var].Neurons[var1].weightsForward))*(self.HiddenLayers[var].Neurons[var1].output*(1-self.HiddenLayers[var].Neurons[var1].output))] #[A] No caso da camada ser só a última também será preciso usar δ de cada neurônio da camada de saída.
                    self.HiddenLayers[var].Neurons[var1].ErrorDerivative += ((-1)*(1/N)*self.HiddenLayers[var].δ[var1])*numpy.array(list(self.HiddenLayers[var-1].Output)+[1]) #[B] Caso geral.
                elif (var == 0): #Caso a camada seja a primeira da rede
                    self.HiddenLayers[var].δ += [(numpy.dot(self.HiddenLayers[var+1].δ, self.HiddenLayers[var].Neurons[var1].weightsForward))*(self.HiddenLayers[var].Neurons[var1].output*(1-self.HiddenLayers[var].Neurons[var1].output))]#[A] Caso geral.
                    self.HiddenLayers[var].Neurons[var1].ErrorDerivative += ((-1)*(1/N)*self.HiddenLayers[var].δ[var1])*numpy.array(list(self.InputLayer.Output)+[1]) #[B] No caso da camada ser a primeira camada oculta, será preciso acessar a saída de cada neurônio da camada de saída.
                else: #Caso geral de uma camada K
                    self.HiddenLayers[var].δ += [(numpy.dot(self.HiddenLayers[var+1].δ, self.HiddenLayers[var].Neurons[var1].weightsForward))*(self.HiddenLayers[var].Neurons[var1].output*(1-self.HiddenLayers[var].Neurons[var1].output))] #[A] Calcula δj de cada neurônio. Nas camadas ocultas dE/dwij = δj * Oi e dE/dbj = δj. Entretanto, no caso das camadas ocultas, a lógica por trás disso também é explicado no material da briliant e também envolve a regra da cadeia (nesse caso é a específica das funções multivariáveis), é igual a um somatório no formato δn(k+1) * Wjn multiplicado pela derivada da função de ativação do neurônio em aj A'(aj), δn(k+1) é dE/daj do neurônio n da camada posterior (k+1) e Wjn é o peso que liga o neurônio j da camada k ao neurônio n (por isso o vetor weigthsForward existe). Para acelerar o somatório o numpy.dot() faz uma soma entre os vetores δ(k+1) (contém todos os dE/dan(k+1) de cada neurônio) e o vetor weightsForward de maneira que numpy.dot([δ0,δ1,...,δn],[Wj0,W1,...,Wjn]) = δ0*Wj0+δ1*Wj1+...+δj*Wjn (é o que é chamado de produto escalar em vetorial). Quanto a derivada da sigmoide essa é dS/daj = S(aj)*(1-S(aj)). Curiosidade: com a softmax o mesmo ocorre dSo/daj = So(aj)*(1-So(aj)).
                    self.HiddenLayers[var].Neurons[var1].ErrorDerivative += ((-1)*(1/N)*self.HiddenLayers[var].δ[var1])*numpy.array(list(self.HiddenLayers[var-1].Output)+[1]) #[B] Calcula dE(X,θ)/dwij = δj * Oi (mesma fórmula da camada de saída), para cada peso do neurônio j e o dE(X,θ)/dbj = δj * Oi (mesma fórmula da camada de saída) por meio do mesmo processo feito nos neurônios da camada de saída, que utiliza o vetor de pesos do neurônio concatenado a uma lista com o valor 1.
        if len(self.HiddenLayers) == 1: #Para o caso em que há uma camada oculta.
            self.OutputLayer.δ = []
            self.HiddenLayers[0].δ = []
        else: #Para casos em que há várias camadas ocultas
            self.HiddenLayers[1].δ = [] #Esvaziar δ para uso posterior
            self.HiddenLayers[0].δ = [] #Esvaziar δ para uso posterior

    def Training(self,Train_data,epochs,mini_batch_size, test_data = False,Test_Validation_or_Test_Cost = True): #Começa o treinamento, Train_data é uma lista de tuplas com a 'imagem' e o vetor que mostra a saída esperada para cada neurônio da saída. Epochs indicam a quantidade de vezes que a rede vai treinar no conjunto de dados, mini_batch_size indica a quantidade de números de cada mini-lote (pequeno conjunto de dados formado por alguns dos dados de treinamento), test_data é  uma outra lista que contém tuplas ela contém a 'imagem' e pode conter que número aquela imagem representa (para verificar a precisão da rede) ou quais os resultados de cada neurônio da saída (usado para avaliar os valores de eta e lambda que ainda permitiam a rede errar menos nos dados de teste), a última variável quando True ,valor normal, indica que a rede vai usar o test_data para verificar a precisão, caso contrário, ela verificará o custo no test_data e preparar um gráfico da evolução do custo com as epochs.
        
        n = len(Train_data) #Quantidade dos dados de treinamento.
        Train_data = list(Train_data)
        mini_batches = [] #mini-lotes
        F = False
        A = [] #Vai conter a quantidade de acertos, erros, porcentagem da precisão e custo do teste.
        for ep in range(epochs): #itera as épocas
            numpy.random.shuffle(Train_data) #Reorganiza as tuplas de treinamento aleatoriamente na lista.
            mini_batches = [Train_data[var:var+mini_batch_size:] for var in range(0,n,mini_batch_size)] #Constrói cada mini-lote.
            for mini_batch in mini_batches: #itera em cada mini-lote.
                for inp, out in mini_batch: #inp é formado pelos valores dos píxels da imagem do número, out é a lista que contém as saídas desejadas para cada neurônio de saída.
                    self.FeedForward(inp, F, mini_batch_size) #Aplica o FeedForward em cada imagem, F é o boleano que controla as atualizações do peso e mini_batch_size é o valor N utilizado na atualização dos pesos.
                    F = False
                    self.Backwards_Propagation(self.OutputLayer.Output, out, mini_batch_size) #Aplica o backpropagation. O primeiro argumento é uma lista com as saídas de cada neurônio da camada de saída, out (saídas desejadas) e mini_batch_size é o N, ambos servem para calcular as influências dos hiperparâmetros.
                F = True
            print("Epoch {} complete.".format(ep+1)) #Quando a epoch acaba.
            if (test_data and not Test_Validation_or_Test_Cost): #Para o caso em que o test_data serve para avaliar o custo da rede em relação aos dados de teste, sendo feito a cada época.
                test_data = list(test_data)
                A += self.Classify(test_data, Test_Validation_or_Test_Cost) #Caso a opção de avaliação de precisão seja utilizada o último elemento, o correspondente ao custo, é igual a 0
        if (test_data and Test_Validation_or_Test_Cost): #Para o caso em que o test_data serve para testar a precisão da rede depois do treinamento em um conjunto de dados do qual ela não treinou.
            test_data = list(test_data)
            A += self.Classify(test_data, Test_Validation_or_Test_Cost) #Quando a opção de avaliação de custo é ativada, os três primeiros índices são iguais a 0.
        if (A[-1] != 0 and len(A) != 0): #Caso o valor no último índice seja diferente de 0, o custo.
            plt.plot([var for var in range(1,epochs+1)],A[3::4], label = "λ = {}/ η = {}".format(self.lamb,self.eta)) #Gera um gráfico de linha com o custo por época. A legenda inclui os valores do lambda e da taxa de aprendizado.
            plt.ylabel("Cost function of test") #Coloca a legenda sobre o eixo y
            plt.xlabel("Epochs") #Coloca a legenda sobre o eixo x
        elif (len(A) > 0):
            print("Casos classificados corretamente: {}\nCasos classificados erroneamente: {}\nPorcentagem: {}".format(A[0],A[1],A[2]))  
        if (Test_Validation_or_Test_Cost):
            self.SaveWeightsBiases() #Função que guarda os pesos e bias pós treinamento.

    def Classify(self,test, Bo = True): #Classifica um conjunto de dados e verifica se a classificação para cada imagem está correta.
        
        Correct = 0
        Cos = 0
        for inp, out in test: #inp é a imagem e out é o resultado esperado
            self.FeedForward(inp) 
            if (Bo): #cálculo da quantidade de previsões corretas
                if (numpy.argmax(self.OutputLayer.Output) == out): #numpy.argmax() retorna o índice do maior número em uma lista. Se a rede acha que a imagem é de um 5, a saída dela será bem próximo de [0,0,0,0,0,1,0,0,0,0] o numpy.argmax() retornará 5, o índice do valor mais próximo de 1.
                    Correct += 1
            else: #Cálculo do custo quando ao teste caso Bo seja Falso.
                Cos += Cross_Entropy(self.OutputLayer.Output, out, len(test), self.lamb, self.weightSum) #Calcula o erro baseado na função de entropia cruzada regularizada, Output é a saída da rede, out é a saída desejada, self.lamb é o lambda da regularização, weightSum é a soma dos quadrados de todos os pesos (parte da regularização da função).
        return [Correct, len(test)-Correct,Correct/len(test)*100,Cos]


    def SaveWeightsBiases(self): #Guarda os valores de pesos e bias de cada neurônio.
        with open("PesosBias.json","w") as Arc:
            J = {
                "InputLayer" : self.InputLayer.QuantityNeurons,
                "HiddenLayers" : [[list(var.weights)+[var.bias] for var in var1.Neurons] for var1 in self.HiddenLayers],
                "OutputLayer" : [list(var.weights)+[var.bias] for var in self.OutputLayer.Neurons]
            }
            json.dump(J, Arc)

    def LoadWeightsBiases(self, J): #Carrega os valores de pesos e bias de um arquivo em um objeto Network.
        self.InputLayer = Layer([Neuron() for var in range(J["InputLayer"])])
        self.HiddenLayers = [Layer([Neuron(numpy.array(var[:-1]),var[-1]) for var in var1]) for var1 in J["HiddenLayers"]]
        self.OutputLayer = Layer([Neuron(numpy.array(var[:-1]),var[-1]) for var in J["OutputLayer"]])

with open("ExpectedTrain.json","r") as A:
    T = json.load(A)
with open("ExpectedTest.json","r") as A:
    Te = json.load(A)
P = json.load(open("PesosBias.json","r"))
# 784,30,10 para (lamb,et,epoch) apresentou o melhor desempenho em ordem 1 (0.1,0.6,30), 2 treino (0.0001,0.1,40), 3 treino (0.0001,0.0005,27) tem menor valor da função de custo para teste.
# 2,4 minutos (2 minutos e 24 segundos) por epoch de 50000 dados.
N1 = Network(Bool=P,lamb=0,et=0)
Data = pandas.read_csv("archive/mnist_train.csv") #[1] Lê os dados do csv
Data = numpy.array(Data) #[2] Transforma um csv em uma matriz.
Data = numpy.transpose(Data) #[3] Pela estrutura do csv os números que estão representados em cada imagem estão na coluna 1. Essa função faz a matriz transposta.
Ans = Data[0][:] #[4] Coloca todas as respostas de cada imagem (que agora estão na primeira linha) em uma variável
Data = numpy.delete(Data,0,axis=0) #[5] apaga a primeira linha
Data = numpy.transpose(Data) * (1/255) #[6] multiplica todos os valores do array por (1/255). Isso serve para facilitar os cálculos do numpy.
DataT = pandas.read_csv("archive/mnist_test.csv") #[1]
DataT = numpy.array(DataT) #[2]
DataT = numpy.transpose(DataT) #[3]
Answer = DataT[0][::] #[4]
DataT = numpy.delete(DataT,0,0) #[5]
DataT = numpy.transpose(DataT) * (1/255) #[6]
Tr = [var for var in zip(Data, T["ET"])] #Constrói uma lista em que cada elemento será uma tupla do tipo (Eln,Eln) sendo o primeiro elemento do primeiro array com o primeiro elemento do segundo, o segundo do primeiro com o segundo do segundo...
Tr = Tr[:50000] #Dados de treinamento
Tes = [var for var in zip(DataT, Answer)] #Precisão do teste
Tess = [var for var in zip(DataT, Te["ETr"])] #Custo do teste (ajuda a regularizar as variáveis eta e lambda)
Trr = [var for var in zip(Data, Ans)] #Precisão nos dados de treinamento
Val = Trr[50000:60000] #Dados de validação (precisão)
Trr = Trr[:50000] #Dados do treinamento (precisão)
Result = N1.Classify(Val)
print("Casos classificados corretamente: {}\nCasos classificados erroneamente: {}\nPrecisão: {}".format(Result[0],Result[1],Result[2])) #Pode ser Trr (números usados para o treinamento), Tes (números do teste, que serviram para ver se a rede não estava apenas "memorizando" os dados de treinamento) ou Val (números que servem para validar o resultado da rede)
'''plt.legend()
plt.show()'''
#Detalhe interessante que percebi é que quando fiz os testes de custo para determinar os valores, quando eu fiz os testes com a MSE (Mean Squared Error) e a função de entropia cruzada, percebi que quanto menor o valor de lambda menor o custo, mas também mais lento a rede aprende e portanto ela ajusta os pesos de maneira mais lenta, caso o lambda seja um pouco maior, o custo será maior e mais rápido a rede aprende.
