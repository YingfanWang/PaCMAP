import pacmap
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Initialize
    pacmap.PaCMAP()
    # print
    print(pacmap.PaCMAP())
    np.random.seed(0)
    sample_data = np.random.normal(size=(2000, 4000))
    instance1 = pacmap.PaCMAP(n_components = 2, n_neighbors = 10, lr = 1, random_state = 20, apply_pca = True)
    instance1_out = instance1.fit_transform(sample_data, init="pca")
    instance2 = pacmap.PaCMAP(n_components = 2, n_neighbors = 10, lr = 1, random_state = 20, apply_pca = True)
    instance2_out = instance2.fit_transform(sample_data)
    print('Experiment finished successfully.')

    print(instance1_out[:3, :3])
    print(instance2_out[:3, :3])

    try:
        assert(np.sum(np.abs(instance1_out-instance2_out))<1e-8)
        print("The output is deterministic.")
    except AssertionError:
        print("The output is not deterministic.")
        try:
            assert(np.sum(np.abs(instance1.pair_FP.astype(int)-instance2.pair_FP.astype(int)))<1e-8)
            assert(np.sum(np.abs(instance1.pair_MN.astype(int)-instance2.pair_MN.astype(int)))<1e-8)
        except AssertionError:
            print('The pairs are not deterministic')
            for i in range(5000):
                if np.sum(np.abs(instance1.pair_FP[i] - instance2.pair_FP[i])) > 1e-8:
                    print("FP")
                    print(i)
                    print(instance1.pair_FP[i])
                    print(instance1.pair_FP[i])
                    break
            for i in range(5000):
                if np.sum(np.abs(instance1.pair_MN[i] - instance2.pair_MN[i])) > 1e-8:
                    print('MN')
                    print(i)
                    print(instance1.pair_MN[i])
                    print(instance2.pair_MN[i])
                    break
