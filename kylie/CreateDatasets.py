import h5py
import numpy as np
import torch
f1 = h5py.File('flowphantom_complicated_combined.hdf5','r')

all_data = np.array(f1['all_data'])
np.random.shuffle(all_data)
training_data = all_data[:134624]
validation_data = all_data[134624:163472]
test_data = all_data[163472:]

training_oxygenations = []
training_spectra = []
validation_oxygenations = []
validation_spectra = []
test_oxygenations = []
test_spectra = []

for i in range(len(training_data)):
    print(i)
    target = list(training_data[i])
    training_oxygenations.append(target[0])
    training_spectra.append(list(target[1:]))
for i in range(len(validation_data)):
    print(i)
    target = list(validation_data[i])
    validation_oxygenations.append(target[0])
    validation_spectra.append(list(target[1:]))
for i in range(len(test_data)):
    print(i)
    target = list(test_data[i])
    test_oxygenations.append(target[0])
    test_spectra.append(list(target[1:]))

training_spectra=np.array(training_spectra)
validation_spectra=np.array(validation_spectra)
test_spectra=np.array(test_spectra)
training_spectra=torch.tensor(training_spectra)
validation_spectra=torch.tensor(validation_spectra)
test_spectra=torch.tensor(test_spectra)

training_oxygenations=torch.tensor(training_oxygenations)
validation_oxygenations = torch.tensor(validation_oxygenations)
test_oxygenations = torch.tensor(test_oxygenations)

print(np.shape(training_spectra))
print(np.shape(validation_spectra))
print(np.shape(test_spectra))
print(np.shape(training_oxygenations))
print(np.shape(validation_oxygenations))
print(np.shape(test_oxygenations))

torch.save(training_spectra, 'training_spectra.pt')
torch.save(validation_spectra, 'validation_spectra.pt')
torch.save(test_spectra, 'test_spectra.pt')
torch.save(training_oxygenations, 'training_oxygenations.pt')
torch.save(validation_oxygenations, 'validation_oxygenations.pt')
torch.save(test_oxygenations, 'test_oxygenations.pt')