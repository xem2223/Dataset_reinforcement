import os

def print_file_structure(directory, indent=0):
    for item in os.listdir(directory):
        path = os.path.join(directory, item)
        print('    ' * indent + '|-- ' + item)
        if os.path.isdir(path):
            print_file_structure(path, indent + 1)

#Eyecandies Dataset structure
print_file_structure('./eyecandies/docs/assets')

#MAD(Multi-pose Anomaly Detection) Dataset structure
#print_file_structure('./PAD/assets')