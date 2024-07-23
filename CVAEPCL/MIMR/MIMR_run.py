import h5py
import os
import numpy as np
from MIMR_class import MIMR_model

def run_MIMR(origin_data, bin_size=0.05, Pct=0.95):

    y_indexes = np.linspace(1, 39, num=20).astype("int")
    x_indexes = np.linspace(1, 79, num=40).astype("int")

    y_obs = []
    x_obs = []

    for i in range(0, 20, 10):
        print("start %d, end %d" % (i, i + 10))
        for j in range(0, 40, 20):
            print("start %d, end %d" % (j, j + 20))

            y_indexes_temp = y_indexes[i:i + 10]
            x_indexes_temp = x_indexes[j:j + 20]
            template = np.zeros((10, 20))
            data = origin_data[:, :, y_indexes_temp, :][:, :, :, x_indexes_temp]
            new_data_shape = [data.shape[0], data.shape[1], 10 * 20]
            data = data.reshape(new_data_shape)
            data_max = np.max(data, axis=0)
            data_min = np.min(data, axis=0)
            data_normalized = (data - data_min) / (data_max - data_min)
            MIMR_obj = MIMR_model(origin_data=data_normalized, bin_size=bin_size, Pct=Pct)
            MIMR_obj.rank_candidate_wells()
            indexes = MIMR_obj.wells_select
            template_one_dim = template.reshape(-1)
            for k in range(len(indexes)):
                index = int(indexes[k])
                template_one_dim[index] = 1
            template = template_one_dim.reshape((10, 20))
            result = np.where(template == 1)

            y_obs.append(y_indexes_temp[result[0]])
            x_obs.append(x_indexes_temp[result[1]])

    return np.concatenate(y_obs), np.concatenate(x_obs)


if __name__ == "__main__":
    input_file1 = "./MC_dataset/MC_dataset_4000.hdf5"
    input_file2 = "./MC_dataset/MC_dataset_3000.hdf5"
    input_file3 = "./MC_dataset/MC_dataset_2000.hdf5"
    input_file4 = "./MC_dataset/MC_dataset_5000.hdf5"


    f = h5py.File(input_file1, "r")
    head1 = np.array(f["head"])
    conc1 = np.array(f["conc"])
    f.close()
    f = h5py.File(input_file2, "r")
    head2 = np.array(f["head"])
    conc2 = np.array(f["conc"])
    f.close()
    f = h5py.File(input_file3, "r")
    head3 = np.array(f["head"])
    conc3 = np.array(f["conc"])
    f.close()
    f = h5py.File(input_file4, "r")
    head4 = np.array(f["head"])
    conc4 = np.array(f["conc"])
    f.close()


    conc1 = conc1.reshape((conc1.shape[0] * conc1.shape[1], conc1.shape[2], conc1.shape[3], conc1.shape[4]))


    ## save Head_output and Conc_output in hdf5 file
    save_dir = "./MIMR_results/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    # set hyperparameters
    Pcts_head = [0.9, 0.99, 0.999]

    head_data = np.concatenate((head2, head3, head4), axis=0)
    for Pct in Pcts_head:
        y_obs, x_obs = run_MIMR(origin_data=head_data, Pct=Pct)
        hf = h5py.File(save_dir + './MIMR_head_Pct{}.hdf5'.format(str(Pct)), 'w')
        hf.create_dataset('y_obs', data=y_obs, dtype='f', compression='gzip')
        hf.create_dataset('x_obs', data=x_obs, dtype='f', compression='gzip')
        hf.close()

    Pcts_conc = [0.8, 0.85, 0.90]
    for Pct in Pcts_conc:
        y_obs, x_obs = run_MIMR(origin_data=conc1, Pct=Pct)
        hf = h5py.File(save_dir + './MIMR_conc_Pct{}.hdf5'.format(str(Pct)), 'w')
        hf.create_dataset('y_obs', data=y_obs, dtype='f', compression='gzip')
        hf.create_dataset('x_obs', data=x_obs, dtype='f', compression='gzip')
        hf.close()
