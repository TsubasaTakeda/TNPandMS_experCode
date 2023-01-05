from sqlalchemy import null
import pandas as pd
import os
import time
import sys
import numpy as np


class MSA:

    def __init__(self):
        # 結果格納用
        self.iter = null
        self.sol = null
        self.sol_obj = null
        self.num_call_obj = null
        self.num_call_dir = null
        self.num_call_conv = null
        self.para_time = null
        self.total_time = null
        # 演算等に必要なデータ
        self.obj_func = null
        self.dir_func = null
        self.x_init = null
        self.conv_judge = null
        self.output_iter = 1000
        # 結果格納用のデータセット
        self.output_data = null
        self.output_root = null

    # 目的関数と降下方向関数を設定(関数の引数はベクトルのみ，outputは[結果, para_time, total_time])
    def set_obj_func(self, obj_func):
        self.obj_func = obj_func

    def set_dir_func(self, dir_func):
        self.dir_func = dir_func

    # 計算等に必要な各種設定をsetting
    def set_output_iter(self, iter):
        self.output_iter = iter

    def set_x_init(self, x_init):
        self.x_init = x_init
    
    def set_conv_judge(self, conv):
        self.conv_judge = conv

    def set_output_root(self, root):
        self.output_root = root



    # 初期解・目的関数・勾配関数・収束判定値(目的関数のdelta)・解を表示するタイミング
    # 点列の記録verもほしい
    def exect_MSA(self):

        is_Executed = 0

        print("\n\n")

        if self.x_init is null:
            print("init sol is not set.")
            is_Executed = -1
        if self.obj_func is null:
            print("objective function is not set.")
            is_Executed = -1
        if self.dir_func is null:
            print("dir function is not set.")
            is_Executed = -1
        if self.conv_judge is null:
            print("convergence judgement (>0) is not set.")
            is_Executed = -1

        if is_Executed:
            print("MSA cannot be executed.\n\n")
            return -1

        print('start MSA!')

        # 初期状態の諸々を計算
        [now_obj, temp_para_time, temp_total_time] = self.obj_func(self.x_init)
        now_conv = sys.float_info.max

        output_data = pd.DataFrame([[0, 0.0, 0.0, now_obj, now_conv, 0, 0]], columns=['Iteration', 'pararel_time', 'total_time', 'now_obj', 'now_conv', 'num_call_obj', 'num_call_dir'])
        if self.output_root != null:
            output_data.to_csv(os.path.join(self.output_root, 'result.csv'))


        # 初期設定（反復回数・目的関数，降下方向関数，射影関数の呼び出し回数を初期化）
        iteration = 0
        num_call_obj = 0
        num_call_dir = 0

        # 計算時間格納用変数
        para_time = 0.0
        total_time = 0.0
        
        now_sol = self.x_init
        # [now_obj, temp_para_time, temp_total_time] = self.obj_func(now_sol)

        if len(now_sol) > 5:
            print('iteration:', iteration, ' now_sol:', now_sol[:5], ' now_obj:', now_obj, ' convergence:', now_conv)
        else:
            print('iteration:', iteration, ' now_sol:', now_sol, ' now_obj:', now_obj, ' convergence:', now_conv)


        while 1:

            iteration += 1
            
            prev_sol = now_sol
            prev_obj = now_obj

            [dir_vec, temp_para_time, temp_total_time] = self.dir_func(prev_sol)
            para_time += temp_para_time
            total_time += temp_total_time
            num_call_dir += 1

            now_sol = prev_sol + dir_vec/(iteration)
            # print('MSA_iteration: ', iteration)
            # print('dir_vec[:30]: ', dir_vec[:30])
            # print('prev_sol[:30]: ', prev_sol[:30])
            # print('now_sol[:30]: ', now_sol[:30])
            [now_obj, temp_para_time, temp_total_time] = self.obj_func(now_sol)  
            para_time += temp_para_time 
            total_time += temp_total_time
            num_call_obj += 1

            # print(now_sol)
            # print(prev_sol)

            now_conv = prev_obj - now_obj

            if iteration % self.output_iter == 0:
                
                # [now_obj, dammy_para_time, dammy_total_time]  = self.obj_func(now_sol)

                if len(now_sol) > 5:
                    print('iteration:', iteration, ' now_sol:', now_sol[:5], ' now_obj:', now_obj, ' convergence:', now_conv)
                else:
                    print('iteration:', iteration, ' now_sol:', now_sol, ' now_obj:', now_obj, ' convergence:', now_conv)

                # num_call_obj += 1
                add_df = pd.DataFrame([[iteration, para_time, total_time, now_obj, now_conv, num_call_obj, num_call_dir]], columns=output_data.columns)
                # print(add_df)
                output_data = output_data.append(add_df)
                if self.output_root != null:
                    output_data.to_csv(os.path.join(self.output_root, 'result.csv'))

            start_time = time.process_time()

            if now_conv < self.conv_judge and now_conv > - self.conv_judge:
                break

            end_time = time.process_time()
            para_time += end_time - start_time
            total_time += end_time - start_time

        end_time = time.process_time()
        para_time += end_time - start_time
        total_time += end_time - start_time

        self.sol = now_sol
        self.iter = iteration
        self.para_time = para_time
        self.total_time = total_time
        self.num_call_obj = num_call_obj
        self.num_call_dir = num_call_dir
        self.sol_obj = now_obj

        
        add_df = pd.DataFrame([[iteration, para_time, total_time, now_obj, now_conv, num_call_obj, num_call_dir]], columns=output_data.columns)
        output_data = output_data.append(add_df)
        self.output_data = output_data
        if self.output_root != null:
            output_data.to_csv(os.path.join(self.output_root, 'result.csv'))
            np.savetxt(os.path.join(self.output_root, 'sol.csv'), self.sol)

        print('finish MSA')

        return 0






# テスト用

# if __name__ == '__main__':

    # # 目的関数の設定
    # def obj_func(x):

    #     obj = 0
        
    #     for i in range(len(x)):
    #         obj += x[i]**2

    #     return obj

    # # 降下方向ベクトル関数を設定
    # def dir_func(x):

    #     dir_vec = -x*1.9
        
    #     return dir_vec



    # x_init = np.array([-20, 2, 3])
        

    # msa = MSA()
    # msa.exect_MSA(x_init, obj_func, dir_func, 0.003, 2)

    # print('\n\n')

    # print('sol: ', msa.sol)
    # print('sol_obj: ', msa.sol_obj)
    # print('iteration: ', msa.iter)
    # # print(msa.num_call_dir)

