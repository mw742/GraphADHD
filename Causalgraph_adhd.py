import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

# For building the MST
import networkx as nx

# pgmpy base classes (for creating a directed acyclic graph)
from pgmpy.base import DAG

# try parameter learning with linear Gaussians (experimental in pgmpy):
# from pgmpy.estimators import MaximumLikelihoodEstimator
# from pgmpy.factors.continuous import LinearGaussianCPD

from collections import deque

########################################
# 1. Load & Merge Data
########################################
mh_p_asr = pd.read_csv("mh_p_asr.csv")
mh_p_abcl = pd.read_csv("mh_p_abcl.csv")
adhd_phenotype = pd.read_csv("mh_p_cbcl.csv", low_memory=False)

prs_data = pd.read_csv(
    "/gpfs/group/ehlers/mwei/ABCD/QC_results_round3/PRS_caucasian_results.best",
    sep=r'\s+',
    header=0
)

# ---- READ & FILTER FOR MALES ONLY ----
# and that the file has columns: FID, IID, Sex, ...
# Read the file with no header, specifying column names manually
sex_pheno = pd.read_csv(
    "sexphenotypes_original.txt",
    sep=r"\s+",            # whitespace-delimited
    header=None,           # no header in the file
    names=["FID", "IID", "Sex"]  # custom column names
)

# Filter for rows indicating male == 1.0
sex_pheno_male = sex_pheno[sex_pheno["Sex"] == 2.0]

# Rename IID -> src_subject_id so we can merge
sex_pheno_male = sex_pheno_male.rename(columns={"IID": "src_subject_id"})


event_of_interest = "2_year_follow_up_y_arm_1"

asr_vars = ["src_subject_id", "eventname", "asr_q06_p", "asr_q90_p", "asr_scr_adhd_t"]
mh_p_asr_filtered = mh_p_asr[mh_p_asr["eventname"] == event_of_interest][asr_vars]

abcl_vars = [
    "src_subject_id", "eventname", "abcl_q06_p", "abcl_q90_p", "abcl_q124_p",
    "abcl_scr_sub_use_tobacco_t", "abcl_scr_sub_use_alcohol_t",
    "abcl_scr_sub_use_t_mean"
]
mh_p_abcl_filtered = mh_p_abcl[mh_p_abcl["eventname"] == event_of_interest][abcl_vars]

cbcl_vars = ["src_subject_id", "eventname", "cbcl_scr_dsm5_adhd_t"]
adhd_filtered = adhd_phenotype[adhd_phenotype["eventname"] == event_of_interest][cbcl_vars]

merged_df = (
    mh_p_asr_filtered
    .merge(mh_p_abcl_filtered, on=["src_subject_id", "eventname"], how="inner")
    .merge(adhd_filtered, on=["src_subject_id", "eventname"], how="inner")
)

prs_data_cleaned = prs_data[["IID", "PRS"]].rename(columns={"IID": "src_subject_id"})
merged_df = merged_df.merge(prs_data_cleaned, on="src_subject_id", how="inner")

# ---- MERGE AGAIN, KEEP ONLY MALES ----
# One way: do an 'inner' merge to keep only rows whose src_subject_id is in sex_pheno_male
merged_df = merged_df.merge(sex_pheno_male[["src_subject_id"]], on="src_subject_id", how="inner")

# Rename ADHD column
merged_df = merged_df.rename(columns={"cbcl_scr_dsm5_adhd_t": "ADHDScore"})

# Columns for the Chow-Liu–like structure
all_cols = [
    "asr_q06_p", "asr_q90_p", "asr_scr_adhd_t",
    "abcl_q06_p", "abcl_q90_p", "abcl_q124_p",
    "abcl_scr_sub_use_tobacco_t", "abcl_scr_sub_use_alcohol_t",
    "abcl_scr_sub_use_t_mean", "PRS", "ADHDScore"
]

merged_df = merged_df.dropna(subset=all_cols)
print("Shape after dropna:", merged_df.shape)

########################################
# 2. Standardize Numeric Data
########################################
scaler = StandardScaler()
scaled_data = scaler.fit_transform(merged_df[all_cols])
X = torch.tensor(scaled_data, dtype=torch.float32)
df_scaled = pd.DataFrame(X.numpy(), columns=all_cols)
print("Shape for BN learning:", df_scaled.shape)

########################################
# 3. Build a 'Chow-Liu' MST for Gaussian data
#    - For continuous variables, the mutual info between X and Y
#      can be approximated via r = corr(X, Y):
#        I(X; Y) = -0.5 * log(1 - r^2)   (if |r| < 1)
#    - Then we build a maximum spanning tree using that measure.
########################################

# Compute correlation matrix
corr_matrix = df_scaled.corr()

# Initialize an undirected graph
G = nx.Graph()
for i in range(len(all_cols)):
    for j in range(i + 1, len(all_cols)):
        var_i = all_cols[i]
        var_j = all_cols[j]
        r = corr_matrix.iloc[i, j]
        # Guard against r=1.0 or edge cases
        if abs(r) >= 1.0:
            # This implies perfectly correlated data or single unique value.
            # We can set the mutual info to a large number, or skip it if degenerate.
            mi = 999999.0
        else:
            # Mutual info for Gaussian
            mi = -0.5 * np.log(1 - r**2 + 1e-12)
        # Add edge with weight=mutual info
        G.add_edge(var_i, var_j, weight=mi)

# Build a maximum spanning tree based on that mutual info
mst = nx.maximum_spanning_tree(G, weight='weight')

########################################
# 4. Convert MST (undirected) to a Directed Acyclic Graph
#    We'll pick ADHDScore as the root, direct edges outward.
########################################
root_var = "ADHDScore"
dag = DAG()
dag.add_nodes_from(all_cols)

visited = {root_var}
queue = [root_var]
while queue:
    node = queue.pop(0)
    # For each neighbor in the MST
    for neighbor in mst[node]:
        if neighbor not in visited:
            visited.add(neighbor)
            # Direct edge: node -> neighbor
            dag.add_edge(node, neighbor)
            queue.append(neighbor)

# Now we have a DAG that is a "Chow-Liu–like" tree oriented from ADHDScore.

print("\nDirected edges in the learned MST from ADHDScore:")
print(dag.edges())


########################################
# 5. Build an Undirected Adjacency for BFS Paths
########################################

adj_undirected = {var: [] for var in all_cols}
for u, v in mst.edges():
    adj_undirected[u].append(v)
    adj_undirected[v].append(u)

########################################
# 6. Find Paths to ADHDScore (Ignore direction, just use BFS in undirected MST)
########################################
def find_path(adj_dict, start, end):
    """Return the unique path (list of node names) in an undirected tree."""
    queue = deque([start])
    visited = {start}
    parent = {start: None}

    while queue:
        node = queue.popleft()
        if node == end:
            break
        for neighbor in adj_dict[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = node
                queue.append(neighbor)

    # Build path by going backwards from 'end' to 'start'
    if end not in parent:
        return None
    path = []
    cur = end
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path

print(f"\nPaths from each variable to '{root_var}':")
for var in all_cols:
    if var == root_var:
        continue
    p = find_path(adj_undirected, var, root_var)
    print(f"  {var} -> {root_var}: {' -> '.join(p)}")

print("\nDone.")

