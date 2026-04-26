import os
import librosa
import numpy as np

print("\n" + "="*50)
print(f"{'Hyperparams Settings':^50}")
print("="*50)
print("Press ENTER for Default values.\n")

def get_int(prompt, default):
    answer = input(f"{prompt} [Default: {default}]: ").strip()
    return int(answer) if answer else default

def get_float(prompt, default):
    answer = input(f"{prompt} [Default: {default}]: ").strip()
    return float(answer) if answer else default

# Recebendo os dados do usuário
neu_qnt = get_int("Number of hidden layer neurons", 128)
lamb = get_float("Lambda value (activation slope)", 0.5)
alpha = get_float("Learning rate (Alpha)", 0.07)
stop_criterion = get_float("Stopping criterion Error (MSE)", 0.005)
max_epochs = get_int("Maximum number of epochs", 5000)

print(f"\nStarting with: Neurons={neu_qnt} | Lambda={lamb} | Alpha={alpha} | Stop<={stop_criterion}")
print("-" * 50)

data_path = './dataset/isolated-guitar-chords/data/Train'
target_chords = ['C', 'Dm', 'Em', 'F', 'G', 'Am', 'Bm']
class_quantity = len(target_chords)

X_list = []
d_list = []

for chord in target_chords:
    folder_path = os.path.join(data_path, chord)
    if not os.path.exists(folder_path):
        continue
    
    chord_index = target_chords.index(chord)
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.wav'):
            file_path = os.path.join(folder_path, file_name)
            signal, sr = librosa.load(file_path, sr=22050)
            mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
            mfccs_mean = np.mean(mfccs.T, axis=0)
            X_list.append(mfccs_mean)
            
            output_array = np.zeros(class_quantity)
            output_array[chord_index] = 1.0
            d_list.append(output_array)

# declarações iniciais
X = np.array(X_list) # entradas dos MFCCs
d = np.array(d_list) # saídas desejadas (one-hot encoding)
x_max_abs = np.max(np.abs(X), axis=0)

X = X / x_max_abs # normalização dos dados de entrada 

bias_colum_x = np.ones((X.shape[0], 1)) 
biased_X = np.hstack((X, bias_colum_x)) # adiciona a coluna de bias às entradas

neu_qnt = 128 # quantidade de neurônios na camada oculta
lamb = 0.5 # parâmetro de inclinação da função de ativação
alpha = 0.03 # taxa de aprendizado
stop_criterion = 0.005 # critério de parada para o erro médio quadrático
W = np.random.uniform(-0.5, 0.5, (biased_X.shape[1], neu_qnt)) # pesos da camada de entrada para a camada oculta (incluindo bias)
Wz = np.random.uniform(-0.5, 0.5, (neu_qnt + 1, class_quantity)) # pesos da camada oculta para a camada de saída (incluindo bias)

for epoch in range(5000): # número de épocas para treinamento
    # muda com as epochs
    S = np.dot(biased_X, W) # produto entre as entradas e os pesos da camada oculta 

    Z = (1 - np.exp(-lamb * S)) / (1 + np.exp(-lamb * S)) # saída intermediária da camada oculta

    bias_colum_z = np.ones((Z.shape[0], 1))
    biased_Z = np.hstack((Z, bias_colum_z)) # adiciona a coluna de bias à saída da camada oculta

    T = np.dot(biased_Z, Wz) # produto entre a saída da camada oculta e os pesos da camada de saída

    stable_T = T - np.max(T, axis=1, keepdims=True) # subtrai o valor máximo de cada linha para estabilidade numérica

    exp_T = np.exp(stable_T)

    Y = exp_T / np.sum(exp_T, axis=1, keepdims=True) # saída da camada de saída usando softmax

    Ey = d - Y # erro da camada de saída

    Deltay = Ey # para softmax com cross-entropy, o delta da camada de saída é simplesmente o erro (d - Y)

    Wz_no_bias = Wz[:-1, :] # remove a última linha de Wz para calcular o erro da camada oculta sem considerar os pesos de bias

    Ez = np.dot(Deltay, Wz_no_bias.T) # erro da camada oculta

    Tetha_z = 0.5 * lamb * (1 - Z ** 2) # derivada da função de ativação para a camada oculta
    Delta_z = Ez * Tetha_z # delta da camada oculta

    Wz += alpha * np.dot(biased_Z.T, Deltay) # atualização dos pesos da camada de saída
    W += alpha * np.dot(biased_X.T, Delta_z) # atualização dos pesos da camada oculta

    mean_squared_error = np.mean(Ey ** 2) # cálculo do erro médio quadrático para monitoramento do treinamento
    
    if epoch % 100 == 0:
        print("Mean Squared Error:", mean_squared_error)

    if mean_squared_error <= stop_criterion:
        print(f"Training completed at epoch {epoch} with Mean Squared Error: {mean_squared_error}")
        break
    
    
os.makedirs('pesosSalvos', exist_ok=True)
np.save('pesosSalvos/W.npy', W)
np.save('pesosSalvos/Wz.npy', Wz)
np.save('pesosSalvos/x_max_abs.npy', x_max_abs)
print("\nCurrent weights successfully saved in the 'pesosSalvos' folder.")