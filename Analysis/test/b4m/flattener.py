import uproot
import pandas as pd

# load the tree
tree = uproot.open('test.root')['tree']

# Convert each branch to a NumPy array first, to avoid performance issues
arrays = tree.arrays(library='np')

# Combine arrays into a DataFrame at once
df = pd.DataFrame(arrays)

# find events with same ru:lumi:event --> these represent events with multiple 4mu candidates
# Group by column 'run' and select the row with the largest value in column 'event'
#df_flattened = df.loc[df.groupby(['run', 'lumi', 'event'])['cos2d'].idxmax()].reset_index(drop=True)
#df_flattened = df.loc[df.groupby('lumi')['event'].idxmax()].reset_index(drop=True)

df_flattened = df.loc[df['charge']==0]
df_flattened = df_flattened.loc[df_flattened.groupby(['run', 'lumi', 'event'])['vtx_prob'].idxmax()].reset_index(drop=True)

# now save file
#fout = uproot.recreate('test_cos2d.root')
fout = uproot.recreate('test_vtx_prob.root')
fout['tree'] = df_flattened
