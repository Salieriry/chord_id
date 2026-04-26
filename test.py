import os
import librosa
import numpy as np

print("\n" + "="*50)
print(f"{'TEST CONFIGURATION':^50}")
print("="*50)

# Menu interativo
print("Choose which weights to test:")
print("[1] BEST saved weights (W81.npy / Wz81.npy)")
print("[2] RECENT weights from last training (W.npy / Wz.npy)")
choice = input("Enter 1 or 2 [Default: 2]: ").strip()

best_mode = (choice == '1')

lamb_str = input("\nWhat Lambda was used in training? [Default: 0.5]: ").strip()
lamb = float(lamb_str) if lamb_str else 0.5

print("-" * 50)

if best_mode:
    w_file = 'pesosSalvos/W81.npy'
    wz_file = 'pesosSalvos/Wz81.npy'
    print("Selected mode: TEST BEST WEIGHTS (81%)")
else:
    w_file = 'pesosSalvos/W.npy'
    wz_file = 'pesosSalvos/Wz.npy'
    print("Selected mode: TEST LATEST TRAINING")

try:
    W = np.load(w_file)
    Wz = np.load(wz_file)
    x_max_abs = np.load('pesosSalvos/x_max_abs.npy')
    print("Weights loaded successfully! Analyzing audio files...\n")
except FileNotFoundError:
    print(f"\nError: Could not find files '{w_file}' or '{wz_file}'. Please train the model first.")
    exit()

hits = 0
data_path = './dataset/isolated-guitar-chords/data/Test'
target_chords = ['C', 'Dm', 'Em', 'F', 'G', 'Am', 'Bm']
class_quantity = len(target_chords)

X_list = []
Chord_list = []

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
            Chord_list.append(chord_index)


X = np.array(X_list) # entradas dos MFCCs
X = X / x_max_abs # normalização dos dados de entrada

bias_colum_x = np.ones((X.shape[0], 1)) 
biased_X = np.hstack((X, bias_colum_x)) # adiciona a coluna de bias às entradas

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

print(f"Total Samples: {len(predicted_classes)}")
print(f"PCorrect predictions: {hits}")
print(f"Accuracy: {hits / len(predicted_classes) * 100:.2f}%")

print("\n" + "="*65)
print(f"{'CONFUSION MATRIX':^65}")
print("="*65)

header = "Real \ Pred | " + " | ".join([f"{chord:>4}" for chord in target_chords])
print(header)
print("-" * len(header))

for i, row in enumerate(confusion_matrix):
    real_chord_name = target_chords[i]
    row_str = " | ".join([f"{val:>4}" for val in row])
    
    total_samples = np.sum(row)
    class_accuracy = (row[i] / total_samples * 100) if total_samples > 0 else 0
    
    print(f"{real_chord_name:>11} | {row_str} | Accuracy: {class_accuracy:>5.1f}%")

print("="*65)