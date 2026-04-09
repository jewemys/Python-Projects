def add_vectors(v1, v2):
    if len(v1) != len(v2):
        raise ValueError("Vectors must be the same length.")

    result = []
    for i in range(len(v1)):
        result.append((v1[i] + v2[i])) # From start index to end index we will add the value in both indexes with each other

    return result # Returns the final added vector

def dot_product(v1, v2):
    # This is checking if v1 is a matrix (list of lists)
    if all(isinstance(x, list) for x in v1):
        m = len(v1)
        n = len(v1[0])
    else:
    # This is checking if v1 is a vector (1 row of data)
        m = 1
        n = len(v1)
        v1 = [v1] # Wrap in list to make 1D array 2D


    # This is checking if v2 is a matrix (list of lists)
    if all(isinstance(x, list) for x in v2):
        p = len(v2)
        q = len(v2[0])
    else:
    # This is checking if v2 is a vector (1 row of data)
        v2 = [[x] for x in v2] # Wrap in list to make 1D array 2D
        p = len(v2)
        q = len(v2[0])

    if n != p:
        raise ValueError("The inputs provided cannot be multiplied, make sure n (m x n) is equal to p (p x q)")

    result_rows = m;
    result_cols = q;

    result = [[0 for i in range(result_cols)] for j in range(result_rows)]

    # Using m x q since those 2 variables are going to be the dimensions of our final output
    for i in range(m):
        for j in range(q):
            total = 0
            for k in range(n):
                total += v1[i][k] * v2[k][j]
            result[i][j] = total

    return result

final = dot_product([[1, 2, 3]], [4, 5, 2])
print(final) # works