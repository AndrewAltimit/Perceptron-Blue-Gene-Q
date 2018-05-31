SHOW_PARSED_DATA = True

filename = "gauss2d_test"
new_filename = "gauss2d_test_bytes"

data_size = 2
label_size = 3

sample_size = data_size + label_size

# Fixing Data Structure
data = []
file = open(filename, "r")


sample_count = 0
for line in file:
    parsed_line = line.strip().replace("]["," ").replace("[", "").replace("]", "").replace(",", "").split(" ")
    for i in range(len(parsed_line)):
        parsed_line[i] = int(parsed_line[i])
    data.append(parsed_line)
    sample_count += 1
    
file.close()

print("Samples Read:", sample_count)

### Writing Data to New File
# Writing Input Data File
file = open(new_filename, 'wb')
for sample in data:
    file.write(bytes(sample)) 
file.close()


if SHOW_PARSED_DATA:
    # Interpreting Input Data
    file = open(new_filename, 'rb')
    input_data = []
    for line in file:
        for element in line:
            input_data.append(element)
            
    '''
    print("Raw bytes:", input_data)
    print("Conversion:", list(input_data))
    print()
    '''
    
    for i in range(0, len(list(input_data)), sample_size):
        print("Sample:", (i // sample_size) + 1)
        print("Pixels:", list(input_data)[i:i+data_size])
        print("Label:", ''.join(str(e) for e in list(input_data)[i+data_size:i+sample_size]))
        print()
    file.close()
