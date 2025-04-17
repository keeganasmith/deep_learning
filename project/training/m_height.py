import numpy as np
def calculate_m_height(x, m, P):
    result = np.matmul(x, P)
    result = np.append(result, x)
    result = np.abs(result)
    result = np.sort(result)
    if(result[-1] == 0 and result[m] == 0):
        return 0
    if(result[-1] == 0):
        return 100000000
    return result[-1] / result[-m - 1]

if __name__ == "__main__":
    x = [-1.2, 3.8]
    P = [[.4759809, .9938236, .819425], [-.8960798, -.7442706, .3345122]]
    result = calculate_m_height(x, 2, P)
    print(result)
        

