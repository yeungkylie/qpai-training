import numpy as np
import h5py
import simpa as sp

def generate_random_coordinates(n_samples):
    np.random.seed(471)
    selection=[]
    for n in range(n_samples+10000):
        print(n)
        image = np.random.randint(0,500)
        i, j, k = np.random.randint(0,64,3)
        coordinates = [image, i, j, k]
        selection.append(coordinates)
    return selection
# coordinates = generate_random_coordinates(500000)
# u_coordinates = np.array(np.unique(coordinates, axis=0))
# chosen_idx = np.random.choice(len(u_coordinates), 500000,replace=False)
# selection = u_coordinates[chosen_idx,:]
# print(selection.shape)
# np.savez('I:/research\seblab\data\group_folders\Kylie/all simulated data/random.npz', selection=selection)
random = np.load('I:/research\seblab\data\group_folders\Kylie/all simulated data/random.npz')
selection = random['selection']
print(selection)

for image in range(3,4):
    print(f'Creating new segmentation for image {image}')
    folder = 'I:/research\seblab\data\group_folders\Kylie/all simulated data\Heterogeneous 0-100/'
    file = "KylieHeterogeneousNoVess2_" + str(int(1e4+image)) + '.hdf5'
    data = h5py.File(folder+file,'r+')
    # with h5py.File(folder+file, 'a') as data:
    #     del data['new_segmentation']
    new_segmentation = np.zeros((64,64,64))

    image_selection = selection[selection[:,0] == image]
    # print(len(image_selection))
    for n in range(len(image_selection)):
        i,j,k = image_selection[n,1:4]
        new_segmentation[i,j,k] = 3
    # print(new_segmentation)
    data.create_dataset("new_segmentation", (64,64,64), data=new_segmentation)
    # print(data['new_segmentation'])
    print(f"The number of pixels match: {len(image_selection) == np.count_nonzero(new_segmentation)}")
