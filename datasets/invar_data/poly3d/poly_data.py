import numpy as np
import os

# set up the parameters for generating the polynomials and data points
def make_data(num_polys):
    num_data_points = 4
    poly_degree_range = (-1, 3)
    max_poly_terms = 4
    max_poly_coef = 10
    max_data_value = 256
    np.random.seed(1)

    # initialize arrays to store the polynomials and data points
    polys = np.zeros((num_polys, max_poly_terms))
    data = np.zeros((num_polys, 2*num_data_points+1, 2), dtype=int)

    # generate the polynomials and data points
    i = 0
    while i < num_polys:
        print(f"Generating polynomial {i + 1}...")
        # choose a random degree for the polynomial
        degree = np.random.randint(*poly_degree_range)

        # choose a random number of terms for the polynomial, up to the maximum
        num_terms = np.random.randint(2, max_poly_terms + 1)

        # choose random coefficients for each term
        coefs = np.random.randint(-max_poly_coef, max_poly_coef + 1, size=num_terms)

        # set up the polynomial expression as a string
        poly_str = ' + '.join([f'{coefs[j]} * x**{j}' for j in range(num_terms)])

        # add the polynomial expression to the array
        polys[i, :num_terms] = coefs

        # generate the data points for the polynomial
        j = 0

        for x in range(-1*num_data_points, num_data_points+1):
            print(f"Generating data point {x + num_data_points + 1}...")
            # evaluate the polynomial at the chosen x value
            y = -1 * np.polyval(np.flip(coefs), x)

            # if the y value is less than or equal to the maximum and the
            # polynomial value is less than or equal to the maximum, store
            # the data point and break out of the loop
            if y <= max_data_value and  -1 * max_data_value <= y:
                data[i, j] = [x, y]
                j = j + 1
        # if more than half of data[i] are 0, then discard this data
        if j > num_data_points:
            i = i + 1

    # print the polynomials and data points if desired
    labels = []
    print("Polynomials:")
    for i in range(num_polys):
        poly_str = ' + '.join([f'{polys[i, j]} * x**{j}' for j in range(max_poly_terms) if polys[i, j] != 0])
        # convert polys[i, j] to one-hot encoding
        one_hot = np.zeros(max_poly_terms)
        one_hot[np.where(polys[i, :] != 0)] = 1
        labels.append(one_hot)
        print(f"Polynomial {i + 1}: {poly_str}")
        print(one_hot)

    datas = []
    print("Data points:")
    for i in range(num_polys):
        print(f"Polynomial {i + 1}:")
        one_data = []
        for j in range(2*num_data_points+1):
            if data[i, j, 1] <= max_data_value and data[i, j, 1] != 0:
                print(f"({data[i, j, 0]}, {data[i, j, 1]})")
                one_data.append([data[i, j, 0], data[i, j, 1]])
        datas.append(one_data)
        # shape of datas: (num_polys, 2*num_data_points+1, 2)
        # shape of labels: (num_polys, max_poly_terms)
    # if one list is empty, then discard this data and the corresponding label
    # print data length
    print("data len: ", len(datas))
    print("label len: " ,len(labels))
    new_datas = []
    new_labels = []
    for i in range(len(datas)):
        if len(datas[i]) != 0:
            new_datas.append(datas[i])
            new_labels.append(labels[i])
    datas = new_datas
    labels = new_labels
    return datas, labels

# Step 1: Preprocess the data
def preprocess_data(data, d_model):
    # normalize the data
    orig_data = data
    for i in range(len(data)):
        print("before norm: ", i, data[i])
        mean = np.mean(data[i], axis=0)
        std = np.std(data[i], axis=0)
        data[i] = (data[i] - mean)
        # if all the elements of data[i] are the same, then skip divide by std
        if std[0] != 0:
            data[i][:, 0] = data[i][:, 0] / std[0]
        if std[1] != 0:
            data[i][:, 1] = data[i][:, 1] / std[1]           
        #convert to float32
        data[i] = data[i].astype(np.float32)
        # if the length of each data is smaller than d_model, pad it with zeros
        # print("after norm: ", data[i])
        if len(data[i]) < d_model:
            for j in range(d_model - len(data[i])):
                data[i] = np.concatenate((data[i], np.array([[float(0), float(0)]])), axis=0)
        #print("after padding: ", data[i])
        if len(data[i]) > d_model:
            data[i] = data[i][:d_model]
        #print("after truncating: ", data[i])
        # print the shape
        #print("shape:", i, data[i].shape)
        print("data:", i, data[i])
        data[i] = data[i].astype(np.float32)
        for j in range(len(data)):
            for k in range(len(data[j])):
                if np.isnan(data[j][k][0]) or np.isnan(data[j][k][1]):
                    print("Find nan: ", j, k, data[j][k])
                    return
    return data

def preprocess_label(label):
    # normalize each label
    for i in range(len(label)):
        label[i] = label[i].astype(np.float32)
        label[i] = label[i] - np.mean(label[i])
        if np.std(label[i]) != 0:
            label[i] = label[i] / np.std(label[i])
        label[i] = label[i].astype(np.float32)
    return label


# main function
if __name__ == "__main__":
    sequence_length = 9
    # if data.npy and labels.npy exist, then skip
    if os.path.exists("data.npy") == False or os.path.exists("labels.npy") == False:
        train_data = make_data(32*32)
        data = preprocess_data(train_data[0], sequence_length)
        np.save("data.npy", data)
        labels = preprocess_label(train_data[1])
        np.save("labels.npy", labels)

    train_data = make_data(32*4)
    data = preprocess_data(train_data[0], sequence_length)
    np.save("test_data.npy", data)
    labels = preprocess_label(train_data[1])
    np.save("test_labels.npy", labels)

