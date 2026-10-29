files = []

#for ipart in range(6):
for ipart in [5]:
    # with open('files_2022C_part%d_bis.txt' %ipart) as f:
    # with open('files_2022D_part%d_22aug22_v1.txt' %ipart) as f:
    with open('files_2022F_prompt_v1_part%d_23oct22v2.txt' %ipart) as f:
        files += [int(''.join(ifile.split('/')[8:10])) for ifile in f.read().splitlines()]

files = list(set(files))
files.sort

print('first run', min(files))
print('last run ', max(files))


# first run 356170
# last run  357080

# first run 356170
# last run  357270
