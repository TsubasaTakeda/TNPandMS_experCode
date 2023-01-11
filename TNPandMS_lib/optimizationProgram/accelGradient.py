
import numpy as np
import time
import os
from sqlalchemy import null
import pandas as pd


class FISTA:

    # 初期設定
    def __init__(self):
        # 結果格納用
        self.iter = null
        self.sol = null
        self.sol_obj = null
        self.num_call_nbl = null
        self.time = null
        # 演算等に必要なデータ
        self.obj_func = null
        self.nbl_func = null
        self.lips = null
        self.x_init = null
        self.conv_judge = null
        self.output_iter = 1000

    # 目的関数と勾配関数を設定(関数の引数はベクトルのみ)
    def set_obj_func(self, obj_func):
        self.obj_func = obj_func
    def set_nbl_func(self, nbl_func):
        self.nbl_func = nbl_func

    # 計算に必要な各種設定をsetting
    def set_lips(self, lips):
        self.lips = lips
    def set_output_iter(self, iter):
        self.output_iter = iter
    def set_conv_judge(self, conv):
        self.conv_judge = conv
    def set_x_init(self, x_init):
        self.x_init = x_init

    # FISTA(初期解, 勾配関数, 勾配のリプシッツ定数, 収束判定値(勾配要素の絶対値の最大値), 結果出力の頻度)
    def exect_FISTA(self):

        is_Executed = 0

        print("\n\n")

        if self.x_init is null:
            print("init sol is not set.")
            is_Executed = -1
        if self.nbl_func is null:
            print("nabla function is not set.")
            is_Executed = -1
        if self.lips is null:
            print("lipschitz constant is not set.")
            is_Executed = -1
        if self.conv_judge is null:
            print("convergence judgement (>0) is not set.")
            is_Executed = -1

        if is_Executed:
            print("FISTA cannot be executed.\n\n")
            return -1
        
        print('start Accel Gradient Method (FISTA)!')

        start_time = time.process_time()

        # 初期設定（反復回数・勾配関数の呼び出し回数を初期化）
        iteration = 0
        num_call_nbl = 0

        # 解の更新で使用する値
        t = 1

        # 初期解の設定
        now_sol = self.x_init
        temp_sol = now_sol

        if len(now_sol) > 5:
            print('iteration:', iteration, ' now_sol:', now_sol[:5])
        else:
            print('iteration:', iteration, ' now_sol:', now_sol)

        while 1:

            iteration += 1

            # 勾配計算（一時的な解の）
            temp_nbl = self.nbl_func(temp_sol)
            num_call_nbl += 1

            # 暫定解の更新
            prev_sol = now_sol
            now_sol = temp_sol - temp_nbl/self.lips
            temp_t = (1 + (1 + 4*t**2)**(1/2))/2
            temp_sol = now_sol + ((t - 1)/temp_t) * (now_sol - prev_sol)
            t = temp_t

            # 収束判定の準備
            now_nbl = self.nbl_func(now_sol)
            num_call_nbl += 1
            nbl_max = np.max(now_nbl)
            nbl_min = np.min(now_nbl)

            if iteration % self.output_iter == 0:
                if len(now_sol) > 5:
                    print('iteration:', iteration, ' now_sol:', now_sol[:5], ' max_nabla:', nbl_max, ' min_nabla:', nbl_min)
                else:
                    print('iteration:', iteration, ' now_sol:', now_sol, ' max_nabla:', nbl_max, ' min_nabla:', nbl_min)

            # 収束判定
            if nbl_max < self.conv_judge and -nbl_min < self.conv_judge:
                break

        end_time = time.process_time()

        self.sol = now_sol
        self.iter = iteration
        self.time = end_time - start_time
        self.num_call_nbl = num_call_nbl

        if self.obj_func is null:
            return 0
        else:
            self.sol_obj = self.obj_func(self.sol)

        return 0


class FISTA_BACK:

    # 初期設定
    def __init__(self):
        # 結果格納用
        self.iter = null
        self.sol = null
        self.sol_obj = null
        self.num_call_nbl = null
        self.num_call_obj = null
        self.lips = null
        self.time = null
        # 演算等に必要なデータ
        self.obj_func = null
        self.nbl_func = null
        self.lips_init = null
        self.x_init = null
        self.conv_judge = null
        self.back_para = null
        self.output_iter = 1000

    # 目的関数と勾配関数を設定(関数の引数はベクトルのみ)
    def set_obj_func(self, obj_func):
        self.obj_func = obj_func
    def set_nbl_func(self, nbl_func):
        self.nbl_func = nbl_func

    # 計算に必要な各種設定をsetting
    def set_lips_init(self, lips_init):
        self.lips_init = lips_init
    def set_output_iter(self, iter):
        self.output_iter = iter
    def set_conv_judge(self, conv):
        self.conv_judge = conv
    def set_back_para(self, para):
        if para <= 1.0:
            print("backtracking parameter is rather than 1.")
            return 0
        self.back_para = para
    def set_x_init(self, x_init):
        self.x_init = x_init

    # FISTA with backtracking
    def exect_FISTA_back(self):

        is_Executed = 0

        print("\n\n")

        if self.x_init is null:
            print("init sol is not set.")
            is_Executed = -1
        if self.obj_func is null:
            print("objective function is not set.")
            is_Executed = -1
        if self.nbl_func is null:
            print("nabla function is not set.")
            is_Executed = -1
        if self.lips_init is null:
            print("initial lipschitz constant (>0) is not set.")
            is_Executed = -1
        if self.back_para is null:
            print("backtracking parameter is not set.")
            is_Executed = -1
        if self.conv_judge is null:
            print("convergence judgement (>0) is not set.")
            is_Executed = -1

        if is_Executed:
            print("FISTA with backtracking cannot be executed.\n\n")
            return -1

        print('start Accel Gradient Method (FISTA with backtracking)!')

        start_time = time.process_time()

        # 初期設定（反復回数・勾配関数の呼び出し回数を初期化）
        iteration = 0
        num_call_obj = 0
        num_call_nbl = 0
        now_lips = self.lips_init

        # 解の更新で使用する値
        t = 1

        # 初期解の設定
        now_sol = self.x_init

        temp_sol = now_sol

        if len(now_sol) > 5:
            print('iteration:', iteration, ' now_sol:', now_sol[:5])
        else:
            print('iteration:', iteration, ' now_sol:', now_sol)

        while 1:

            iteration += 1

            # 勾配計算（一時的な解の）
            temp_nbl = self.nbl_func(temp_sol)
            num_call_nbl += 1

            # backtracking
            [now_lips, temp_call_obj, temp_call_nbl] = self.backtracking(temp_sol, now_lips)
            num_call_obj += temp_call_obj
            num_call_nbl += temp_call_nbl

            # 暫定解の更新
            prev_sol = now_sol
            now_sol = temp_sol - temp_nbl/now_lips
            # print(now_lips)
            # print(temp_nbl)
            if self.nbl_func(prev_sol) @ (now_sol - prev_sol) > 0:
                t = 1.0
            num_call_nbl += 1
            temp_t = (1.0 + (1.0 + 4.0*t**2.0)**(1.0/2.0))/2.0
            temp_sol = now_sol + ((t - 1.0)/temp_t) * (now_sol - prev_sol)
            t = temp_t

            # 収束判定の準備
            now_nbl = self.nbl_func(now_sol)
            num_call_nbl += 1
            nbl_max = np.max(now_nbl)
            nbl_min = np.min(now_nbl)

            if iteration % self.output_iter == 0:
                if len(now_sol) > 5:
                    print('iteration:', iteration, ' now_sol:',
                          now_sol[:5], ' max_nabla:', nbl_max, ' min_nabla:', nbl_min)
                else:
                    print('iteration:', iteration, ' now_sol:', now_sol,
                          ' max_nabla:', nbl_max, ' min_nabla:', nbl_min)

            # 収束判定
            if nbl_max < self.conv_judge and -nbl_min < self.conv_judge:
                break

        end_time = time.process_time()

        self.sol = now_sol
        self.iter = iteration
        self.time = end_time - start_time
        self.num_call_nbl = num_call_nbl
        self.num_call_obj = num_call_obj
        self.sol_obj = self.obj_func(self.sol)

        print('finish accel gradient method')

        return 0


class FISTA_PROJ_BACK:

    # 初期設定
    def __init__(self):
        # 結果格納用
        self.iter = null
        self.sol = null
        self.sol_obj = null
        self.num_call_nbl = null
        self.num_call_obj = null
        self.num_call_conv = null
        self.num_call_proj = null
        self.lips = null
        self.para_time = null
        self.total_time = null
        # 演算等に必要なデータ
        self.obj_func = null
        self.nbl_func = null
        self.proj_func = null
        self.conv_func = null
        self.lips_init = null
        self.x_init = null
        self.conv_judge = null
        self.back_para = null
        self.output_iter = 1000
        # 結果格納用のデータセット
        self.output_data = null
        self.output_root = null

    # 目的関数と勾配関数を設定(関数の引数はベクトルのみ)
    def set_obj_func(self, obj_func):
        self.obj_func = obj_func

    def set_nbl_func(self, nbl_func):
        self.nbl_func = nbl_func

    def set_proj_func(self, proj_func):
        self.proj_func = proj_func

    def set_conv_func(self, conv_func):
        self.conv_func = conv_func

    # 計算に必要な各種設定をsetting
    def set_lips_init(self, lips_init):
        self.lips_init = lips_init

    def set_output_iter(self, iter):
        self.output_iter = iter

    def set_conv_judge(self, conv):
        self.conv_judge = conv

    def set_back_para(self, para):
        if para <= 1.0:
            print("backtracking parameter is rather than 1.")
            return 0
        self.back_para = para

    def set_x_init(self, x_init):
        self.x_init = x_init

    def set_output_root(self, root):
        self.output_root = root

    # FISTA with backtracking
    def exect_FISTA_proj_back(self):

        is_Executed = 0

        print("\n\n")

        if self.x_init is null:
            print("init sol is not set.")
            is_Executed = -1
        if self.obj_func is null:
            print("objective function is not set.")
            is_Executed = -1
        if self.nbl_func is null:
            print("nabla function is not set.")
            is_Executed = -1
        if self.proj_func is null:
            print("projection function is not set.")
            is_Executed = -1
        if self.conv_func is null:
            print("convergence function is not set.")
            is_Executed = -1
        if self.lips_init is null:
            print("initial lipschitz constant (>0) is not set.")
            is_Executed = -1
        if self.back_para is null:
            print("backtracking parameter is not set.")
            is_Executed = -1
        if self.conv_judge is null:
            print("convergence judgement (>0) is not set.")
            is_Executed = -1

        if is_Executed:
            print("FISTA with backtracking cannot be executed.\n\n")
            return -1

        print('start Accel Gradient Projection Method (FISTA with backtracking)!')

        # 初期状態の諸々を計算
        [now_obj, temp_para_time, temp_total_time] = self.obj_func(self.x_init)
        [now_conv, temp_para_time, temp_total_time] = self.conv_func(self.x_init)

        output_data = pd.DataFrame([[0, 0.0, 0.0, now_obj, now_conv, self.lips_init, 0, 0, 0, 0]], columns=['Iteration', 'pararel_time', 'total_time', 'now_obj', 'now_conv', 'now_lips', 'num_call_obj', 'num_call_nbl', 'num_call_proj', 'num_call_conv'])
        if self.output_root != null:
            output_data.to_csv(os.path.join(self.output_root, 'result.csv'))


        # 初期設定（反復回数・勾配関数の呼び出し回数を初期化）
        iteration = 0
        num_call_obj = 0
        num_call_nbl = 0
        num_call_conv = 0
        num_call_proj = 0
        now_lips = self.lips_init

        # 解の更新で使用する値
        t = 1

        # 初期解の設定
        now_sol = self.x_init

        temp_sol = now_sol

        # 計算時間格納用変数
        para_time = 0.0
        total_time = 0.0

        if len(now_sol) > 5:
            print('iteration:', iteration, ' now_sol:', now_sol[:5], ' now_obj:', now_obj, ' convergence:', now_conv)
        else:
            print('iteration:', iteration, ' now_sol:', now_sol, ' now_obj:', now_obj, ' convergence:', now_conv)


        while 1:

            iteration += 1

            # 勾配計算（一時的な解の）
            [temp_nbl, temp_para_time, temp_total_time] = self.nbl_func(temp_sol)
            para_time += temp_para_time
            total_time += temp_total_time
            num_call_nbl += 1

            # backtracking
            [now_lips, temp_call_obj, temp_call_nbl, temp_call_proj, temp_para_time, temp_total_time] = self.backtracking(temp_sol, now_lips)
            para_time += temp_para_time
            total_time += temp_total_time
            num_call_obj += temp_call_obj
            num_call_nbl += temp_call_nbl
            num_call_proj += temp_call_proj

            # 暫定解の更新
            prev_sol = now_sol
            [now_sol, temp_para_time, temp_total_time] = self.proj_func(temp_sol - temp_nbl/now_lips)
            para_time += temp_para_time
            total_time += temp_total_time
            num_call_proj += 1
            [prev_nbl, temp_para_time, temp_total_time] = self.nbl_func(prev_sol)
            para_time += temp_para_time
            total_time += temp_total_time

            start_time = time.process_time()
            if prev_nbl @ (now_sol - prev_sol) > 0:
                t = 1.0
            num_call_nbl += 1
            temp_t = (1.0 + (1.0 + 4.0*t**2.0)**(1.0/2.0))/2.0
            temp_sol = now_sol + ((t - 1.0)/temp_t) * (now_sol - prev_sol)
            end_time = time.process_time()
            para_time += end_time - start_time
            total_time += end_time - start_time


            [temp_sol, temp_para_time, temp_total_time] = self.proj_func(temp_sol)
            para_time += temp_para_time
            total_time += temp_total_time
            t = temp_t
            # 収束判定の準備
            [conv, temp_para_time, temp_total_time] = self.conv_func(now_sol)
            para_time += temp_para_time
            total_time += temp_total_time
            num_call_conv += 1

            if iteration % self.output_iter == 0:
                
                [now_obj, dammy_para_time, dammy_total_time]  = self.obj_func(now_sol)

                if len(now_sol) > 5:
                    print('iteration:', iteration, ' now_sol:', now_sol[:5], ' now_obj:', now_obj, ' convergence:', conv)
                else:
                    print('iteration:', iteration, ' now_sol:', now_sol, ' now_obj:', now_obj, ' convergence:', conv)

                # num_call_obj += 1
                add_df = pd.DataFrame([[iteration, para_time, total_time, now_obj, conv, now_lips, num_call_obj, num_call_nbl, num_call_proj, num_call_conv]], columns=output_data.columns)
                # print(add_df)
                output_data = output_data.append(add_df)
                if self.output_root != null:
                    output_data.to_csv(os.path.join(self.output_root, 'result.csv'))

            # 収束判定
            if conv < self.conv_judge:
                break

        self.sol = now_sol
        self.iter = iteration
        self.para_time = para_time
        self.total_time = total_time
        self.num_call_nbl = num_call_nbl
        self.num_call_obj = num_call_obj
        self.num_call_proj = num_call_proj
        self.num_call_conv = num_call_conv
        [self.sol_obj, dummy_para_time, dummy_total_time] = self.obj_func(self.sol)
        
        if iteration % self.output_iter != 0:
            add_df = pd.DataFrame([[self.iter, self.para_time, self.total_time, self.sol_obj, conv, now_lips, self.num_call_obj, self.num_call_nbl, self.num_call_proj, self.num_call_conv]], columns=output_data.columns)
            output_data = output_data.append(add_df)
            self.output_data = output_data
            if self.output_root != null:
                output_data.to_csv(os.path.join(self.output_root, 'result.csv'))
            np.savetxt(os.path.join(self.output_root, 'sol.csv'), self.sol)

        print('finish accel gradient projection method')

        return 0



    def backtracking(self, now_sol, now_lips):

        para_time = 0.0
        total_time = 0.0
        
        [now_obj, temp_para_time, temp_total_time] = self.obj_func(now_sol)
        para_time += temp_para_time
        total_time += temp_total_time
        [now_nbl, temp_para_time, temp_total_time] = self.nbl_func(now_sol)
        para_time = temp_para_time
        total_time += temp_total_time
        num_call_obj = 1
        num_call_nbl = 1
        num_call_proj = 0

        start_time = time.process_time()
        # now_nbl_nolm = np.dot(now_nbl, now_nbl)

        iota = 0
        while 1:

            [temp_sol, temp_para_time, temp_total_time] = self.proj_func(now_sol - now_nbl/(self.back_para**iota * now_lips))
            temp_sol = now_sol - now_nbl/(self.back_para**iota * now_lips)
            num_call_proj += 1
            end_time = time.process_time()
            para_time += end_time - start_time
            total_time += end_time - start_time

            [F, temp_para_time, temp_total_time] = self.obj_func(temp_sol)
            para_time += temp_para_time
            total_time += temp_total_time
            num_call_obj += 1

            start_time = time.process_time()
            now_nbl_nolm = np.dot(temp_sol-now_sol, temp_sol-now_sol)
            Q = now_obj + np.dot(now_nbl, temp_sol - now_sol) + now_nbl_nolm * (self.back_para**iota*now_lips/2.0)

            # print('F: ', F, 'Q: ', Q)

            if F <= Q:
                break

            iota += 1

        end_time = time.process_time()
        para_time += end_time - start_time
        total_time += end_time - start_time

        return [self.back_para**iota*now_lips, num_call_obj, num_call_nbl, num_call_proj, para_time, total_time]



# テスト用

if __name__ == '__main__':

    # 目的関数の設定
    def obj_func(x):

        obj = 0

        for i in range(len(x)):
            obj += x[i]**2

        return obj

    # 降下方向ベクトル関数を設定
    def nbl_func(x):

        nbl_vec = x*2.0

        return nbl_vec

    x_init = np.array([-20, 2, 3, 7, 8, 9, 0])

    fista = FISTA()
    fista.set_x_init(x_init)
    fista.set_obj_func(obj_func)
    fista.set_nbl_func(nbl_func)
    fista.set_lips(1000)
    fista.set_conv_judge(0.03)
    fista.exect_FISTA()

    print('\n\n')

    print('sol: ', fista.sol)
    print('sol_obj: ', fista.sol_obj)
    print('iteration: ', fista.iter)
    print('elapsed_time: ', fista.time)
    print('num_call_nabla: ', fista.num_call_nbl)

    fista_back = FISTA_BACK()
    fista_back.set_x_init(x_init)
    fista_back.set_obj_func(obj_func)
    fista_back.set_nbl_func(nbl_func)
    fista_back.set_lips_init(0.01)
    fista_back.set_back_para(1.1)
    fista_back.set_conv_judge(0.03)
    fista_back.set_output_iter(1)
    fista_back.exect_FISTA_back()

    print('\n\n')

    print('sol: ', fista_back.sol)
    print('sol_obj: ', fista_back.sol_obj)
    print('iteration: ', fista_back.iter)
    print('elapsed_time: ', fista_back.time)
    print('num_call_nabla: ', fista_back.num_call_nbl)
    print('num_call_obj: ', fista_back.num_call_obj)
