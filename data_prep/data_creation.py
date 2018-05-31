# Example of grayscale pixels followed by label
#       PIXELS---------------  LABEL------
data = [[153, 21, 33, 123, 240, 1,0,0,0,0,0],
        [123, 23, 12, 44, 23,   0,1,0,0,0,0],
        [123, 26, 13, 42, 10,   0,0,1,0,0,0],
        [123, 23, 74, 44, 74,   0,0,0,1,0,0],
        [123, 23, 38, 23, 23,   0,0,0,0,1,0],
        [123, 23, 82, 88, 111,  0,0,0,0,0,1]]

data_size = 5
label_size = 6

filename = "input_other"

sample_size = data_size + label_size

# Writing Input Data File
file = open(filename, 'wb')
for sample in data:
    file.write(bytes(sample)) 
file.close()

# Interpreting Input Data
file = open(filename, 'rb')
input_data = []
for line in file:
    for element in line:
        input_data.append(element)

print("Raw bytes:", input_data)
print("Conversion:", list(input_data))
print()

for i in range(0, len(list(input_data)), sample_size):
    print("Sample:", (i // sample_size) + 1)
    print("Pixels:", list(input_data)[i:i+data_size])
    print("Label:", ''.join(str(e) for e in list(input_data)[i+data_size:i+sample_size]))
    print()
file.close()
