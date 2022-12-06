
import numpy as np
from sqlalchemy import null

class AccelGradient:

    def __init__(self):
        self.iter = null
        self.sol = null
        self.sol_obj = null
        self.num_call_nbl = null

    # FISTA(初期解, 勾配関数, 勾配のリプシッツ定数, 収束判定値(勾配要素の絶対値の最大値), 結果出力の頻度)
    def exect_FISTA(self, x_init, nbl_func, lip_cont, conv_judge, output_iter):

        # 初期設定（反復回数・勾配関数の呼び出し回数を初期化）
        iteration = 0
        num_call_nbl = 0

        # 解の更新で使用する値
        t = 1

        # 初期解の設定
        now_sol = x_init
        temp_sol = now_sol

        print('start accel gradient method!')
        if len(now_sol) > 5:
            print('iteration:', iteration, ' now_sol:', now_sol[:5])
        else:
            print('iteration:', iteration, ' now_sol:', now_sol)

        while 1:

            iteration += 1

            # 勾配計算（一時的な解の）
            temp_nbl = nbl_func(temp_sol)
            num_call_nbl += 1

            # 暫定解の更新
            prev_sol = now_sol
            now_sol = temp_sol - temp_nbl/lip_cont
            temp_t = (1 + (1 + 4*t**2)**(1/2))/2
            temp_sol = now_sol + ((t - 1)/temp_t) * (now_sol - prev_sol)
            t = temp_t

            # 収束判定の準備
            now_nbl = nbl_func(now_sol)
            num_call_nbl += 1
            nbl_max = np.max(now_nbl)
            nbl_min = np.min(now_nbl)

            if iteration % output_iter == 0:
                if len(now_sol) > 5:
                    print('iteration:', iteration, ' now_sol:', now_sol[:5], ' max_nabla:', nbl_max, ' min_nabla:', nbl_min)
                else:
                    print('iteration:', iteration, ' now_sol:', now_sol, ' max_nabla:', nbl_max, ' min_nabla:', nbl_min)

            # 収束判定
            if nbl_max < conv_judge and -nbl_min < conv_judge:
                break

        self.sol = now_sol
        self.iter = iteration
        # self.time = 0
        self.num_call_nbl = num_call_nbl

        return 0





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

    msa = AccelGradient()
    msa.exect_FISTA(x_init, nbl_func, 200, 0.003, 5)

    msa.sol_obj = obj_func(msa.sol)

    print('\n\n')


    print('sol: ', msa.sol)
    print('sol_obj: ', msa.sol_obj)
    print('iteration: ', msa.iter)
    print('num_call_nabla: ', msa.num_call_nbl)
