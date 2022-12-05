

class MSA:

    # def __init__(self):

    # 初期解・目的関数・勾配関数・収束判定値(目的関数のdelta)・解を表示するタイミング
    # 点列の記録verもほしい
    def exect_MSA(self, x_init, obj_func, dir_func, conv_judge, output_iter):

        # 初期設定（反復回数・目的関数，降下方向関数，射影関数の呼び出し回数を初期化）
        iteration = 0
        num_call_obj = 0
        num_call_dir = 0
        conv = conv_judge*2
        
        now_sol = x_init
        now_obj = obj_func(now_sol)

        print('iteration:', iteration, ' now_sol:', now_sol, ' now_obj', now_obj)


        while 1:

            iteration += 1
            
            prev_sol = now_sol
            prev_obj = now_obj

            dir_vec = dir_func(prev_sol)
            num_call_dir += 1

            now_sol = prev_sol + dir_vec/iteration
            now_obj = obj_func(now_sol)  
            num_call_obj += 1

            # print(now_sol)
            # print(prev_sol)

            conv = prev_obj - now_obj

            if iteration%output_iter == 0:
                print('iteration:', iteration, ' now_sol:', now_sol, ' now_obj:', now_obj, ' conv:', conv)


            if conv < conv_judge and conv > -conv_judge:
                break

        self.sol = now_sol
        self.sol_obj = now_obj
        self.iter = iteration
        # self.time = 0
        self.num_call_obj = num_call_obj
        self.num_call_dir = num_call_dir

        return 0





if __name__ == '__main__':
    
    import numpy as np

    # 目的関数の設定
    def obj_func(x):

        obj = 0
        
        for i in range(len(x)):
            obj += x[i]**2

        return obj

    # 降下方向ベクトル関数を設定
    def dir_func(x):

        dir_vec = -x*1.9
        
        return dir_vec



    x_init = np.array([-20, 2, 3])
        

    msa = MSA()
    msa.exect_MSA(x_init, obj_func, dir_func, 0.003, 2)

    print('\n\n')

    print('sol: ', msa.sol)
    print('sol_obj: ', msa.sol_obj)
    print('iteration: ', msa.iter)
    # print(msa.num_call_dir)

