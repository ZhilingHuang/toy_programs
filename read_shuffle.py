from random import shuffle

def read_data(f_path):
    with open(f_path) as f:
        lines = f.readlines()
        data = []
        for line in lines:
            if not line or line == '':
                continue
            data.append(line)
    return data

def shuffle_and_save(f_path):
    with open(f_path + '_shuffled', 'w') as f:
        d = read_data(f_path)
        shuffle(d)
        for l in d:
            f.write(l)
            f.flush()

def transformline(l):
    if l[0] == 'M':
        return '0,0,1' + l[1:]
    elif l[0] == 'F':
        return '0,1,0' + l[1:]
    else:
        return '1,0,0' + l[1:]

def split_to_train_dev_test(f_path):
    with open(f_path) as f:
        data = f.readlines()
    ten_percent =  len(data) / 10
    with open(f_path + '_test', 'w') as f:
        for line in data[:ten_percent]:
            if line:
                f.write(transformline(line))
    with open(f_path + '_dev', 'w') as f:
        for line in data[ten_percent:ten_percent*2]:
            if line:
                f.write(transformline(line))
    with open(f_path + '_train', 'w') as f:
        for line in data[ten_percent*2:]:
            if line:
                f.write(transformline(line))

# shuffle_and_save('abalone_data')
# split_to_train_dev_test('abalone_data_shuffled')

shuffle_and_save('breast_cancer')

def transformline2(l):
    vals = l.split()
    for i in range(1, len(vals)):
        if i <= 9:
            vals[i] = vals[i][2:]
        else:
            vals[i] = vals[i][3:]
    return ' '.join(vals) + '\n'


def split_to_train_dev_test(f_path):
    with open(f_path) as f:
        data = f.readlines()
    percent =  len(data) / 5
    with open(f_path + '_test', 'w') as f:
        for line in data[:percent]:
            if line:
                f.write(transformline2(line))
    with open(f_path + '_dev', 'w') as f:
        for line in data[percent:percent*2]:
            if line:
                f.write(transformline2(line))
    with open(f_path + '_train', 'w') as f:
        for line in data[percent*2:]:
            if line:
                f.write(transformline2(line))

split_to_train_dev_test('breast_cancer_shuffled')

