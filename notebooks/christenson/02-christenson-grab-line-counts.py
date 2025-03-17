import re

n = 0
lines = open("christenson-pv-english.txt", "r").readlines()
print("chap_num,xom-lines")
for line in lines:
    if re.match(r"^\s+\d+\s+(L|l)ines\s+", line):
        n += 1
        print(n, line.strip().split()[-1], sep=',')