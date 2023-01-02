import gurobipy as gp
import numpy as np
from sqlalchemy import null


# 線計画問題を解くプログラム
# min c x
# s.t., B x <= b, 
#       B_eq x = b_eq
def linprog(c, B, b, B_eq = null, b_eq = null):

    if len(c) != B.shape[1]:
        print('目的関数の係数ベクトルcと，制約条件の係数行列Bについて，変数の数が合っていません．')
        print('c.shape: ', c.shape)
        print('B.shape: ', B.shape)
        return -1
    elif B.shape[0] != len(b):
        print('係数行列Bと，制約ベクトルbについて，制約条件の数が合っていません．')
        print('B.shape: ', B.shape)
        print('b.shpae: ', b.shape)
        return -1
    elif B_eq is not null:
        if len(c) != B_eq.shape[1]:
            print('目的関数の係数ベクトルcと，制約条件の係数行列B_eqについて，変数の数が合っていません．')
            print('c.shape: ', c.shape)
            print('B_eq.shape: ', B_eq.shape)
            return -1
        elif B_eq.shape[0] != len(b_eq):
            print('係数行列B_eqと，制約ベクトルb_eqについて，制約条件の数が合っていません．')
            print('B_eq.shape: ', B_eq.shape)
            print('b_eq.shpae: ', b_eq.shape)
            return -1


    # モデルの宣言
    model = gp.Model(name="Sample")

    # 変数の宣言
    x = {i: model.addVar() for i in range(len(c))}

    # 目的関数の設定
    model.setObjective(gp.quicksum(c[i]*x[i] for i in range(len(c))), gp.GRB.MINIMIZE)

    # 制約条件の設定
    for i in range(B.shape[0]):
        model.addConstr(gp.quicksum(B[i, j] * x[j] for j in range(B.shape[1])) <= b[i])
    if B_eq is not null:
        for i in range(B_eq.shape[0]):
            model.addConstr(gp.quicksum(B_eq[i, j] * x[j] for j in range(B_eq.shape[1])) == b_eq[i])

    # 解を求める計算
    print("↓点線の間に、Gurobi Optimizerからログが出力")
    print("-" * 40)

    model.optimize()

    print("-" * 40)
    print()

    sol = null

    if model.Status == gp.GRB.OPTIMAL:
        sol = np.array([x[i].X for i in x.keys()])

    return model, sol


if __name__ == '__main__':

    from scipy import sparse

    # 目的関数の係数ベクトル
    c = np.array([-2, -1])

    # 制約条件の係数行列
    # B = np.array([[1, 1], [1, 2], [-1, 0], [1, 0], [-1, 0], [1, 0]])
    B = sparse.coo_matrix((6, 2))
    B = B.tocsr()
    B[0, 0] = 1
    B[0, 1] = 1
    B[1, 0] = 1
    B[1, 1] = 2
    B[2, 0] = -1
    B[3, 0] = 1
    B[4, 1] = -1
    B[5, 1] = 1
    b = np.array([4, 6, 0, 3, 0, 5])

    B_eq = np.array([[1, 2]])
    b_eq = np.array([1])



    [model, sol] = linprog(c, B, b, B_eq, b_eq)

    # 最適解が得られた場合、結果を出力
    if model.Status == gp.GRB.OPTIMAL:
        # 目的関数の値
        val_opt = model.ObjVal
        print(f"最適解は x = {sol}")
        print(f"最適値は {val_opt}")
        print(f"実行時間は {model.Runtime}")
        # print(type(sol))
