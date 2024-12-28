English Description Below:

Custom Layer Normalization

Neste projeto, desenvolvi um pipeline de aprendizado de máquina utilizando TensorFlow, com foco no treinamento e validação de uma rede neural em um conjunto de dados.

O objetivo principal foi a criação de uma camada personalizada de normalização (custom Layer Normalization) e sua comparação com a função integrada (built-in function) do TensorFlow.

A seguir, descrevo os principais passos e componentes do meu projeto:

1. Normalização Personalizada

Criei uma camada personalizada de normalização (LayerNormalization) para padronizar os dados.

Desenvolvi a funcionalidade utilizando operações de TensorFlow, ajustando parâmetros como alpha e beta para controle do desvio padrão e da média.

Comparei os resultados da minha implementação com a camada integrada de normalização do TensorFlow, garantindo equivalência com um erro insignificante.

2. Pré-processamento dos Dados

Utilizei o conjunto de dados Fashion MNIST, incluindo divisão em subconjuntos de treino, validação e teste.

Escalei os valores das imagens para o intervalo [0, 1] e ajustei os dados de entrada para o formato apropriado.

3. Estrutura da Rede Neural

Configurei a arquitetura do modelo com duas partes principais:

Camadas inferiores (lower layers): 

Responsáveis por extração de recursos com camadas densas e ativação ReLU.

Camadas superiores (upper layers): 

Realizei a classificação com uma camada de saída softmax para 10 classes.

Implementei a rede neural utilizando o módulo tf.keras.Sequential.

4. Otimização e Treinamento

Utilizei dois otimizadores diferentes:

SGD com momento e nesterov=True: 

Para as camadas inferiores.

Nadam: 

Para as camadas superiores.

Treinei a rede neural utilizando o algoritmo de backpropagation com gradientes calculados via tf.GradientTape.

5. Treinamento em Mini-lotes

Implementei a seleção aleatória de lotes de dados durante o treinamento, garantindo uma boa generalização do modelo.

Realizei múltiplas iterações (épocas) e utilizei barras de progresso para acompanhar o treinamento.


6. Métricas de Avaliação

Monitorei a perda de treino e a acurácia categórica esparsa durante cada iteração.

Validei o desempenho do modelo utilizando o conjunto de validação após cada época.


7. Gerenciamento de Estado

Resetava as métricas ao final de cada época para garantir precisão na avaliação subsequente.

8. Ferramentas e Bibliotecas
   
Bibliotecas: time, tensorflow, numpy, tqdm, collections

Utilizei o tqdm para criar barras de progresso detalhadas que acompanham o treinamento por etapa e por época.






Custom Layer Normalization

In this project, I developed a machine learning pipeline using TensorFlow, focusing on training and validating a neural network on a dataset.

The primary goal was to create a custom Layer Normalization layer and compare its performance with TensorFlow's built-in function.

Below are the main steps and components of my project:

1. Custom Normalization

I designed a custom Layer Normalization layer to standardize the data.

I implemented the functionality using TensorFlow operations, adjusting parameters like alpha and beta for controlling the standard deviation and mean.

I compared the results of my implementation with TensorFlow's built-in normalization layer, ensuring equivalence with negligible error.

2. Data Preprocessing

I used the Fashion MNIST dataset, including splitting into training, validation, and test subsets.

I scaled image values to the range [0, 1] and adjusted the input data format as required.

3. Neural Network Structure

I configured the model architecture with two main parts:

Lower layers: 

I extracted features using dense layers with ReLU activation.

Upper layers: 

I performed classification using a softmax output layer for 10 classes.

Built the neural network using the tf.keras.Sequential module.

4. Optimization and Training

I utilized two different optimizers:

SGD with momentum and nesterov=True: 

Applied to the lower layers.

Nadam: 

Applied to the upper layers.

I trained the neural network using backpropagation with gradients computed via tf.GradientTape.

5. Mini-batch Training

I Implemented random selection of data batches during training to ensure good model generalization.

I conducted multiple iterations (epochs) and used progress bars to track training performance.

6. Evaluation Metrics

I monitored training loss and sparse categorical accuracy during each iteration.

I validated model performance on the validation set after each epoch.

7. State Management

I reset metrics at the end of each epoch to ensure accurate subsequent evaluations.

8. Tools and Libraries

Libraries: 

time, tensorflow, numpy, tqdm, collections.

I utilized tqdm to create detailed progress bars for tracking training at both step and epoch levels.
