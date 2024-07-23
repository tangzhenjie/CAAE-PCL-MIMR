import numpy as np
import copy
from tqdm import tqdm

class MIMR_model(object):
    def __init__(self, origin_data, bin_size, lambda_1=0.8, lambda_2=0.2, Pct=0.95):
        """
        origin_data: the normalized head or conc value in [0, 1], shape[num of samples, 6, num of candidate monitoring wells]
        bin_size: the fixed bin size
        """
        self.origin_data = origin_data
        self.bin_size = bin_size
        self.num_candidate_wells = origin_data.shape[-1]
        self.lambda_1 = lambda_1  # λ1
        self.lambda_2 = lambda_2  # λ2
        self.Pct = Pct

        # discrete data using bin_size
        self.discrete_data_vector = ((np.floor((2 * origin_data + bin_size) / (2 * bin_size)) * bin_size) * 100).astype('int8')

        # convert vector to scalar eg.[num_sample, 6, num_candidate]->[num_sample, num_candidate]
        self.vector_to_scalar()

    def vector_to_scalar(self):
        dt = np.dtype(
            (np.void, self.discrete_data_vector.dtype.itemsize * self.discrete_data_vector.shape[1]))

        T_discrete_data_vector = np.transpose(self.discrete_data_vector, axes=(0, 2, 1))
        self.discrete_data_scalar = np.ascontiguousarray(T_discrete_data_vector).view(dt)[:, :, 0]

        # T_discrete_data_scalar_new_dtype = np.ascontiguousarray(T_discrete_data_vector).view(dt)[:, :, 0]
        # unique_total_array, count_total_array = np.unique(T_discrete_data_scalar_new_dtype, return_counts=True)
        #
        # self.discrete_data_scalar = -np.ones(shape=T_discrete_data_scalar_new_dtype.shape, dtype=np.int32)  # shape [num_sample, wells]
        # scalars = [i for i in range(len(unique_total_array))]
        # for i in range(len(unique_total_array)):
        #     self.discrete_data_scalar[T_discrete_data_scalar_new_dtype == unique_total_array[i]] = scalars[i]

        return

    def marginal_entropy(self):
        self.marginal_entropy_matrix = np.ones([self.num_candidate_wells, 1])
        for i in range(self.num_candidate_wells):
            unique_array, unique_count = np.unique(self.discrete_data_scalar[:, i], return_counts=True)
            probabilities = unique_count / np.sum(unique_count)
            self.marginal_entropy_matrix[i, 0] = -np.sum(np.log2(probabilities) * probabilities)
        return

    def total_joint_entropy(self):
        dt = np.dtype((np.void, self.discrete_data_scalar.dtype.itemsize * self.discrete_data_scalar.shape[1]))
        array_Total = np.ascontiguousarray(self.discrete_data_scalar).view(dt)
        unique_total_array, count_total_array = np.unique(array_Total, return_counts=True)
        percentage_joint = count_total_array / np.sum(count_total_array)
        self.total_joint_entropy = -np.sum(np.log2(percentage_joint) * percentage_joint)
        return

    def matrix_based_joint_entropy(self, matrix):
        dt = np.dtype((np.void, matrix.dtype.itemsize * matrix.shape[1]))
        array_Total = np.ascontiguousarray(matrix).view(dt)
        unique_total_array, count_total_array = np.unique(array_Total, return_counts=True)
        percentage_matrix = count_total_array / np.sum(count_total_array)
        entropy_matrix_based = -np.sum(np.log2(percentage_matrix) * percentage_matrix)
        return entropy_matrix_based

    def selected_joint_entropy(self, combine_matrix_i):
        joint_entropy_S = np.ones([combine_matrix_i.shape[-1], 1])
        for i in range(combine_matrix_i.shape[-1]):
            combine_matrix_i_iter = combine_matrix_i[:, :, i].T
            joint_entropy_S[i, 0] = self.matrix_based_joint_entropy(combine_matrix_i_iter)
        return joint_entropy_S

    def transinformation_relation_based(self, entropy_H_conbine, single_matrix, combine_matrix, marginal_entropy_rest):
        """
        entropy_H_conbine: [num_F, 1]
        single_matrix: [1, num_sample, num_F]
        combine_matrix: [1+num_S, num_sample, num_F]
        marginal_entropy_rest: [num_F, 1]
        """
        num_F = single_matrix.shape[-1]
        Sum_H_S = (num_F-1) * entropy_H_conbine
        Sum_M_F = np.sum(marginal_entropy_rest) - marginal_entropy_rest
        if len(Sum_M_F.shape) == 1:
            Sum_M_F = Sum_M_F[:, None]
        entropy_J_F_matrix = np.zeros([num_F, num_F])
        for i in range(num_F-1):
            combine_matrix_i_iter = combine_matrix[:, :, i].T
            for j_0 in range(num_F - i - 1):
                j = j_0 + i + 1
                single_matrix_j_iter = single_matrix[:, :, j].T
                t_matrix_j = np.concatenate((single_matrix_j_iter, combine_matrix_i_iter), axis=1)
                entropy_J_F_matrix[i, j] = entropy_J_F_matrix[j, i] = self.matrix_based_joint_entropy(t_matrix_j)
        Sum_J_F = np.sum(entropy_J_F_matrix, axis=1)
        if len(Sum_J_F.shape) == 1:
            Sum_J_F = Sum_J_F[:, None]
        transinformation_array = Sum_H_S + Sum_M_F - Sum_J_F
        return transinformation_array

    def rank_candidate_wells(self):
        self.marginal_entropy()
        self.total_joint_entropy()

        object_value_i = np.ones([1, 4])

        wells_index = np.arange(self.num_candidate_wells)
        self.wells_select = np.array([])

        for i in tqdm(range(self.num_candidate_wells)):
            # 循环该轮，找到目前最大目标值的 wells, 并判断是否满足条件
            if i == 0:
                # 用于存储F集合的采样数据[1, num_sample, num_wells_F]
                single_matrix_i = np.expand_dims(self.discrete_data_scalar, axis=0)

                # 用于存储F集合+S集合的采样数据[1 + num_wells_S, num_sample, num_wells_F]  也就是该轮循环所需计算的所有数据
                combine_matrix_i = np.expand_dims(self.discrete_data_scalar, axis=0)

                # 用于存储1 + num_wells_S边际熵的和 [num_wells_F, 1]
                sum_marginal_entropy_S = self.marginal_entropy_matrix

                # 用于存储num_wells_F的边际熵 [num_wells_F, 1]
                mariginal_entropy_i = self.marginal_entropy_matrix

            """
            计算目标函数
            """
            # 计算所有这次候选wells + S的联合熵
            entropy_H_conbine_i = self.selected_joint_entropy(combine_matrix_i)
            entropy_C_correlation_i = sum_marginal_entropy_S - entropy_H_conbine_i
            entropy_T_i = self.transinformation_relation_based(entropy_H_conbine=entropy_H_conbine_i,  # [num_F, 1]
                                                               single_matrix=single_matrix_i,  # [1, num_sample, num_F]
                                                               combine_matrix=combine_matrix_i,  # [1+num_S, num_sample, num_F]
                                                               marginal_entropy_rest=mariginal_entropy_i) # [num_F, 1]

            MIMR_objective = self.lambda_1 * (entropy_H_conbine_i + entropy_T_i) \
                             - self.lambda_2 * entropy_C_correlation_i

            # 选出该轮循环的最大目标函数的井
            max_idex = np.where(MIMR_objective == np.max(MIMR_objective))[0]

            if len(max_idex) > 1:
                raise Exception("More than 1 candidate well appears with equal maximum value!")
            else:
                max_idex = max_idex[0]

            """
            暂存结果
            """
            self.wells_select = np.append(self.wells_select, wells_index[max_idex])
            wells_index = np.delete(wells_index, max_idex)

            object_value_i[0, 0] = entropy_H_conbine_i[max_idex]
            object_value_i[0, 1] = entropy_T_i[max_idex]
            object_value_i[0, 2] = entropy_C_correlation_i[max_idex]
            object_value_i[0, 3] = MIMR_objective[max_idex]

            if i == 0:
                self.final_objects = copy.deepcopy(object_value_i)
            else:
                self.final_objects = np.concatenate((self.final_objects, object_value_i), axis=0)

            """
            判断是否停止
            """
            judge_C = entropy_H_conbine_i[max_idex]

            print(entropy_H_conbine_i[max_idex], self.total_joint_entropy)
            if judge_C >= self.Pct * self.total_joint_entropy:
                break

            if len(self.wells_select) == self.num_candidate_wells:
                break

            """
            修改中间变量
            """
            matrix_i_pickout = single_matrix_i[:, :, max_idex][:, :, None]
            combine_matrix_i_rest = np.delete(combine_matrix_i, max_idex, axis=-1)
            matrix_i_pickout_tile = np.tile(matrix_i_pickout, [1, 1, combine_matrix_i_rest.shape[-1]])
            combine_matrix_i = np.concatenate((combine_matrix_i_rest, matrix_i_pickout_tile), axis=0)

            single_matrix_i = np.delete(single_matrix_i, max_idex, axis=-1)

            new_S = mariginal_entropy_i[max_idex, :]
            mariginal_entropy_i = np.delete(mariginal_entropy_i, max_idex)
            if len(mariginal_entropy_i.shape) == 1:
                mariginal_entropy_i = mariginal_entropy_i[:, None]

            sum_marginal_entropy_S = np.delete(sum_marginal_entropy_S, max_idex)
            if len(sum_marginal_entropy_S.shape) == 1:
                sum_marginal_entropy_S = sum_marginal_entropy_S[:, None]

            sum_marginal_entropy_S = new_S + sum_marginal_entropy_S        # 已选中点的边际熵和















