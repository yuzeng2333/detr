def get_degrees(targets, d_model):
    degrees = []
    for target in targets:
        # declare a set
        degree_set = set()
        degree_list = target['max_degree']
        for deg in degree_list:
            degree_set.add(deg)
        one_degree_list = [0] * d_model
        for deg in degree_set:
            one_degree_list[deg] = 1
        degrees.append(one_degree_list)
    # convert degrees to float
    degrees = [[float(i) for i in deg] for deg in degrees]
    return degrees