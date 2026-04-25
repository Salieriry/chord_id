import os
import librosa
import numpy as np

acertos = 0

# caminho dos dados de treinamento
data_path = './dataset/isolated-guitar-chords/data/Test'

target_chords = ['C', 'Dm', 'Em', 'F', 'G', 'Am', 'Bm']
class_quantity = len(target_chords)

W = np.load('W.npy')
Wz = np.load('Wz.npy')
x_max_abs = np.load('x_max_abs.npy')

print("Weights loaded successfully!")

X_list = []
Gabarito_list = []

for chord in target_chords:
    folder_path = os.path.join(data_path, chord)
    
    if not os.path.exists(folder_path):
        print(f"Folder for chord '{chord}' not found. Skipping.")
        continue
    
    chord_index = target_chords.index(chord)
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.wav'):
            file_path = os.path.join(folder_path, file_name)
            signal, sr = librosa.load(file_path, sr=22050)
            mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
            mfccs_mean = np.mean(mfccs.T, axis=0)
            
            X_list.append(mfccs_mean)
            Gabarito_list.append(chord_index)

            
            
X = np.array(X_list) # entradas dos MFCCs
X = X / x_max_abs # normalização dos dados de entrada

bias_colum_x = np.ones((X.shape[0], 1)) 
biased_X = np.hstack((X, bias_colum_x)) # adiciona a coluna de bias às entradas

lamb = 0.5

S = np.dot(biased_X, W) # produto entre as entradas e os pesos da camada oculta 

Z = (1 - np.exp(-lamb * S)) / (1 + np.exp(-lamb * S)) # saída intermediária da camada oculta

bias_colum_z = np.ones((Z.shape[0], 1))
biased_Z = np.hstack((Z, bias_colum_z)) # adiciona a coluna de bias à saída da camada oculta

T = np.dot(biased_Z, Wz) # produto entre a saída da camada oculta e os pesos da camada de saída

exp_T = np.exp(T)

Y = exp_T / np.sum(exp_T, axis=1, keepdims=True) # saída da camada de saída usando softmax

predicted_classes = np.argmax(Y, axis=1)
gabarito = np.array(Gabarito_list)
confidence_scores = np.max(Y, axis=1) * 100

confusion_matrix = np.zeros((class_quantity, class_quantity), dtype=int)

for i in range(len(predicted_classes)):
     
     
    if predicted_classes[i] == gabarito[i]:
        acertos += 1 
        
    confusion_matrix[gabarito[i], predicted_classes[i]] += 1            

print("\nCerteza da rede por amostra:")
for i in range(len(predicted_classes)):
    real_label = target_chords[gabarito[i]]
    predicted_label = target_chords[predicted_classes[i]]
    print(
        f"Amostra {i + 1:03d} | Real: {real_label:<3} | Prevista: {predicted_label:<3} | Certeza: {confidence_scores[i]:6.2f}%"
    )

print(f"Total samples: {len(predicted_classes)}, Correct predictions: {acertos}")
print(f"Accuracy: {acertos / len(predicted_classes) * 100:.2f}%")

print("\nConfusion Matrix:")
print("Columns (Predicted) ->")
print(target_chords)
print(confusion_matrix)








