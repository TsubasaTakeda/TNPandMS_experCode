from sqlalchemy import null
import pandas as pd
import os
import time
import sys
import numpy as np
import linprog as lp

# 黄金分割法
def goldenRatioSearch(function, range, tolerance):

    para_time = 0.0
    total_time = 0.0


    gamma = (-1+np.sqrt(5))/2
    a = range[0]
    b = range[1]
    p = b-gamma*(b-a)
    q = a+gamma*(b-a)
    [Fa, temp_para_time, temp_total_time] = function(a)
    [Fb, temp_para_time, temp_total_time] = function(b)
    [Fp, temp_para_time, temp_total_time] = function(p)
    para_time += temp_para_time
    total_time += temp_total_time
    [Fq, temp_para_time, temp_total_time] = function(q)
    para_time += temp_para_time 
    total_time += temp_total_time

    num_call_obj = 4

    while 1:

        if Fp <= Fq:
            b = q
            q = p
            Fb = Fq
            Fq = Fp
            p = b-gamma*(b-a)
            [Fp, temp_para_time, temp_total_time] = function(p)
            para_time += temp_para_time
            total_time += temp_total_time
            num_call_obj += 1
        else:
            a = p
            p = q
            Fa = Fp
            Fp = Fq
            q = a+gamma*(b-a)
            [Fq, temp_para_time, temp_total_time] = function(q)
            para_time += temp_para_time
            total_time += temp_total_time
            num_call_obj += 1
        
        conv = Fa - Fb
        if conv < 0.0:
            conv = -conv

        if conv < tolerance:
            break

    alpha = (a+b)/2
    return alpha, para_time, total_time, num_call_obj


class FrankWolf:

    def __init__(self):
        # 結果格納用
        self.iter = null
        self.sol = null
        self.sol_obj = null
        self.num_call_obj = null
        self.num_call_nbl = null
        self.para_time = null
        self.total_time = null
        # 演算等に必要なデータ
        self.obj_func = null
        self.nbl_func = null
        self.x_init = null
        self.conv_judge = null
        self.output_iter = 1000
        self.B = null
        self.B_eq = null
        self.b = null
        self.b_eq = null
        self.lb = null
        self.ub = null
        # 結果格納用のデータセット
        self.output_data = null
        self.output_root = null

    # 目的関数と降下方向関数を設定(関数の引数はベクトルのみ，outputは[結果, para_time, total_time])
    def set_obj_func(self, obj_func):
        self.obj_func = obj_func

    def set_nbl_func(self, nbl_func):
        self.nbl_func = nbl_func

    # 計算等に必要な各種設定をsetting
    def set_output_iter(self, iter):
        self.output_iter = iter

    def set_x_init(self, x_init):
        self.x_init = x_init
    
    def set_conv_judge(self, conv):
        self.conv_judge = conv

    def set_output_root(self, root):
        self.output_root = root
    def set_B(self, B):
        self.B = B

    def set_B_eq(self, B_eq):
        self.B_eq = B_eq

    def set_b(self, b):
        self.b = b
    
    def set_b_eq(self, b_eq):
        self.b_eq = b_eq

    def set_lb(self, lb):
        self.lb = lb

    def set_ub(self, ub):
        self.ub = ub
        
    def dir_func(self, now_sol, now_nbl):

        para_time = 0.0
        total_time = 0.0
        
        [model, temp_sol, temp_time] = lp.linprog(now_nbl, self.B, self.b, self.B_eq, self.b_eq, self.lb, self.ub)
        para_time += temp_time
        total_time += temp_time

        dir_vec = temp_sol - now_sol

        return dir_vec, para_time, total_time


    # ステップサイズを決定する関数
    def calc_stepSize(self, now_sol, dir_vec):

        # for i in range(now_sol.shape[0]):
        #     if now_sol[i] < 0.0:
        #         print('now_sol!!', i, now_sol[i])
        #     if (now_sol + dir_vec)[i] < 0.0:
        #         print('dir_sol!!', i, (now_sol + dir_vec)[i])

        def stepFunc(step_size):
            temp_sol = now_sol + dir_vec*step_size
            [temp_obj, para_time, total_time] = self.obj_func(temp_sol)
            return temp_obj, para_time, total_time

        [stepSize, para_time, total_time, num_call_obj] = goldenRatioSearch(stepFunc, [0.0, 1.0], self.conv_judge)

        return stepSize, para_time, total_time, num_call_obj



    def exect_FW(self):

        is_Executed = 0

        print("\n\n")

        if self.x_init is null:
            print("init sol is not set.")
            is_Executed = -1
        if self.obj_func is null:
            print("objective function is not set.")
            is_Executed = -1
        if self.nbl_func is null:
            print("nbl function is not set.")
            is_Executed = -1
        if self.conv_judge is null:
            print("convergence judgement (>0) is not set.")
            is_Executed = -1

        if is_Executed:
            print("Frank-Wolf cannot be executed.\n\n")
            return -1

        print('start Frank-Wolf!')

        # 初期状態の諸々を計算
        [now_obj, temp_para_time, temp_total_time] = self.obj_func(self.x_init)
        # [now_nbl, temp_para_time, temp_total_time] = self.nbl_func(self.x_init)
        # now_conv = np.max(now_nbl)
        now_conv = sys.float_info.max

        output_data = pd.DataFrame([[0, 0.0, 0.0, now_obj, now_conv, 0, 0]], columns=['Iteration', 'pararel_time', 'total_time', 'now_obj', 'now_conv', 'num_call_obj', 'num_call_nbl'])
        if self.output_root != null:
            output_data.to_csv(os.path.join(self.output_root, 'result.csv'))

        # 初期設定（反復回数・目的関数，降下方向関数，射影関数の呼び出し回数を初期化）
        iteration = 0
        num_call_obj = 0
        num_call_nbl = 0

        # 計算時間格納用変数
        para_time = 0.0
        total_time = 0.0
        
        now_sol = self.x_init

        if len(now_sol) > 5:
            print('iteration:', iteration, ' now_sol:', now_sol[:5], ' now_obj:', now_obj, ' convergence:', now_conv)
        else:
            print('iteration:', iteration, ' now_sol:', now_sol, ' now_obj:', now_obj, ' convergence:', now_conv)


        while 1:

            iteration += 1
            
            prev_sol = now_sol
            prev_obj = now_obj

            [prev_nbl, temp_para_time, temp_total_time] = self.nbl_func(prev_sol)
            para_time += temp_para_time
            total_time += temp_total_time
            num_call_nbl += 1

            [dir_vec, temp_para_time, temp_total_time] = self.dir_func(prev_sol, prev_nbl)
            para_time += temp_para_time 
            total_time += temp_total_time

            [stepSize, temp_para_time, temp_total_time, temp_num_call_obj] = self.calc_stepSize(prev_sol, dir_vec)
            para_time += temp_para_time
            total_time += temp_total_time
            num_call_obj += temp_num_call_obj

            now_sol = prev_sol + dir_vec * stepSize
            [now_obj, temp_para_time, temp_total_time] = self.obj_func(now_sol)  
            para_time += temp_para_time 
            total_time += temp_total_time
            num_call_obj += 1

            # now_conv = np.max(prev_nbl)
            now_conv = prev_obj - now_obj
            if now_conv < 0.0:
                now_conv = -now_conv

            if iteration % self.output_iter == 0:
                
                # [now_obj, dammy_para_time, dammy_total_time]  = self.obj_func(now_sol)

                if len(now_sol) > 5:
                    print('iteration:', iteration, ' now_sol:', now_sol[:5], ' now_obj:', now_obj, ' convergence:', now_conv)
                else:
                    print('iteration:', iteration, ' now_sol:', now_sol, ' now_obj:', now_obj, ' convergence:', now_conv)

                # num_call_obj += 1
                add_df = pd.DataFrame([[iteration, para_time, total_time, now_obj, now_conv, num_call_obj, num_call_nbl]], columns=output_data.columns)
                # print(add_df)
                output_data = output_data.append(add_df)
                if self.output_root != null:
                    output_data.to_csv(os.path.join(self.output_root, 'result.csv'))

            if now_conv < self.conv_judge:
                break

        self.sol = now_sol
        self.iter = iteration
        self.para_time = para_time
        self.total_time = total_time
        self.num_call_obj = num_call_obj
        self.num_call_nbl = num_call_nbl
        self.sol_obj = now_obj

        
        add_df = pd.DataFrame([[iteration, para_time, total_time, now_obj, now_conv, num_call_obj, num_call_nbl]], columns=output_data.columns)
        output_data = output_data.append(add_df)
        self.output_data = output_data
        if self.output_root != null:
            np.savetxt(os.path.join(self.output_root, 'sol.csv'), self.sol)
            if iteration % self.output_iter != 0:
                output_data.to_csv(os.path.join(self.output_root, 'result.csv'))

        print('finish Frank-Wolf')

        return 0



class FrankWolf_MSA:

    def __init__(self):
        # 結果格納用
        self.iter = null
        self.sol = null
        self.sol_obj = null
        self.num_call_obj = null
        self.num_call_nbl = null
        self.para_time = null
        self.total_time = null
        # 演算等に必要なデータ
        self.obj_func = null
        self.nbl_func = null
        self.x_init = null
        self.conv_judge = null
        self.output_iter = 1000
        self.B = null
        self.B_eq = null
        self.b = null
        self.b_eq = null
        self.lb = null
        self.ub = null
        # 結果格納用のデータセット
        self.output_data = null
        self.output_root = null

    # 目的関数と降下方向関数を設定(関数の引数はベクトルのみ，outputは[結果, para_time, total_time])
    def set_obj_func(self, obj_func):
        self.obj_func = obj_func

    def set_nbl_func(self, nbl_func):
        self.nbl_func = nbl_func

    # 計算等に必要な各種設定をsetting
    def set_output_iter(self, iter):
        self.output_iter = iter

    def set_x_init(self, x_init):
        self.x_init = x_init
    
    def set_conv_judge(self, conv):
        self.conv_judge = conv

    def set_output_root(self, root):
        self.output_root = root
    def set_B(self, B):
        self.B = B

    def set_B_eq(self, B_eq):
        self.B_eq = B_eq

    def set_b(self, b):
        self.b = b
    
    def set_b_eq(self, b_eq):
        self.b_eq = b_eq

    def set_lb(self, lb):
        self.lb = lb

    def set_ub(self, ub):
        self.ub = ub
        
    def dir_func(self, now_sol, now_nbl):

        para_time = 0.0
        total_time = 0.0
        
        [model, temp_sol, temp_time] = lp.linprog(now_nbl, self.B, self.b, self.B_eq, self.b_eq, self.lb, self.ub)
        para_time += temp_time
        total_time += temp_time

        dir_vec = temp_sol - now_sol

        return dir_vec, para_time, total_time

    def exect_FW(self):

        is_Executed = 0

        print("\n\n")

        if self.x_init is null:
            print("init sol is not set.")
            is_Executed = -1
        if self.obj_func is null:
            print("objective function is not set.")
            is_Executed = -1
        if self.nbl_func is null:
            print("nbl function is not set.")
            is_Executed = -1
        if self.conv_judge is null:
            print("convergence judgement (>0) is not set.")
            is_Executed = -1

        if is_Executed:
            print("Frank-Wolf cannot be executed.\n\n")
            return -1

        print('start Frank-Wolf!')

        # 初期状態の諸々を計算
        [now_obj, temp_para_time, temp_total_time] = self.obj_func(self.x_init)
        # [now_nbl, temp_para_time, temp_total_time] = self.nbl_func(self.x_init)
        # now_conv = np.max(now_nbl)
        now_conv = sys.float_info.max

        output_data = pd.DataFrame([[0, 0.0, 0.0, now_obj, now_conv, 0, 0]], columns=['Iteration', 'pararel_time', 'total_time', 'now_obj', 'now_conv', 'num_call_obj', 'num_call_nbl'])
        if self.output_root != null:
            output_data.to_csv(os.path.join(self.output_root, 'result.csv'))

        # 初期設定（反復回数・目的関数，降下方向関数，射影関数の呼び出し回数を初期化）
        iteration = 0
        num_call_obj = 0
        num_call_nbl = 0

        # 計算時間格納用変数
        para_time = 0.0
        total_time = 0.0
        
        now_sol = self.x_init

        if len(now_sol) > 5:
            print('iteration:', iteration, ' now_sol:', now_sol[:5], ' now_obj:', now_obj, ' convergence:', now_conv)
        else:
            print('iteration:', iteration, ' now_sol:', now_sol, ' now_obj:', now_obj, ' convergence:', now_conv)


        while 1:

            iteration += 1
            
            prev_sol = now_sol
            prev_obj = now_obj

            [prev_nbl, temp_para_time, temp_total_time] = self.nbl_func(prev_sol)
            para_time += temp_para_time
            total_time += temp_total_time
            num_call_nbl += 1

            [dir_vec, temp_para_time, temp_total_time] = self.dir_func(prev_sol, prev_nbl)
            para_time += temp_para_time 
            total_time += temp_total_time

            stepSize = 2.0 / (iteration + 2.0)

            now_sol = prev_sol + dir_vec * stepSize
            [now_obj, temp_para_time, temp_total_time] = self.obj_func(now_sol)  
            para_time += temp_para_time 
            total_time += temp_total_time
            num_call_obj += 1

            # now_conv = np.max(prev_nbl)
            now_conv = prev_obj - now_obj
            if now_conv < 0.0:
                now_conv = -now_conv

            if iteration % self.output_iter == 0:
                
                # [now_obj, dammy_para_time, dammy_total_time]  = self.obj_func(now_sol)

                if len(now_sol) > 5:
                    print('iteration:', iteration, ' now_sol:', now_sol[:5], ' now_obj:', now_obj, ' convergence:', now_conv)
                else:
                    print('iteration:', iteration, ' now_sol:', now_sol, ' now_obj:', now_obj, ' convergence:', now_conv)

                # num_call_obj += 1
                add_df = pd.DataFrame([[iteration, para_time, total_time, now_obj, now_conv, num_call_obj, num_call_nbl]], columns=output_data.columns)
                # print(add_df)
                output_data = output_data.append(add_df)
                if self.output_root != null:
                    output_data.to_csv(os.path.join(self.output_root, 'result.csv'))

            if now_conv < self.conv_judge:
                break

        self.sol = now_sol
        self.iter = iteration
        self.para_time = para_time
        self.total_time = total_time
        self.num_call_obj = num_call_obj
        self.num_call_nbl = num_call_nbl
        self.sol_obj = now_obj

        
        add_df = pd.DataFrame([[iteration, para_time, total_time, now_obj, now_conv, num_call_obj, num_call_nbl]], columns=output_data.columns)
        output_data = output_data.append(add_df)
        self.output_data = output_data
        if self.output_root != null:
            np.savetxt(os.path.join(self.output_root, 'sol.csv'), self.sol)
            if iteration % self.output_iter != 0:
                output_data.to_csv(os.path.join(self.output_root, 'result.csv'))

        print('finish Frank-Wolf')

        return 0
