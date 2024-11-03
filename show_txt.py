from collections import deque

def print_last_n_lines(filename, n=100):
    with open(filename, 'r') as file:
        last_lines = deque(file, maxlen=n)
    for line in last_lines:
        print(line, end='')

# Call the function with your file path
print_last_n_lines('trains/SMALL-A100-448-10k-OBS-SCHEDULER/log.txt')
