import os
import librosa
import numpy as np

hits = 0

# caminho dos dados de treinamento
data_path = './dataset/isolated-guitar-chords/data/Test'

target_chords = ['C', 'Dm', 'Em', 'F', 'G', 'Am', 'Bm']
class_quantity = len(target_chords)

W = np.load('pesosSalvos/W81.npy')
Wz = np.load('pesosSalvos/Wz81.npy')
x_max_abs = np.load('pesosSalvos/x_max_abs.npy')

print("Weights loaded successfully!")

X_list = []
Chord_list = []

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
            Chord_list.append(chord_index)

            
            
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
real_chord = np.array(Chord_list)

confusion_matrix = np.zeros((class_quantity, class_quantity), dtype=int)

for i in range(len(predicted_classes)):
     
     
    if predicted_classes[i] == real_chord[i]:
        hits += 1 
        
    confusion_matrix[real_chord[i], predicted_classes[i]] += 1            


print(f"Total samples: {len(predicted_classes)}, Correct predictions: {hits}")
print(f"Accuracy: {hits / len(predicted_classes) * 100:.2f}%")

print("\n" + "="*65)
print(f"{'CONFUSION MATRIX':^65}")
print("="*65)

header = "Real \ Pred | " + " | ".join([f"{chord:>4}" for chord in target_chords])
print(header)
print("-" * len(header))


for i, row in enumerate(confusion_matrix):
    real_chord = target_chords[i]
    
    row_str = " | ".join([f"{val:>4}" for val in row])
    
    total_samples = np.sum(row)
    class_accuracy = (row[i] / total_samples * 100) if total_samples > 0 else 0
    
    print(f"{real_chord:>11} | {row_str} | Accuracy: {class_accuracy:>5.1f}%")

print("="*65)








