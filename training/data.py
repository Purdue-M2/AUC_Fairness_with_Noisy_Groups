import numpy as np

def compute_phats(df_proxy, proxy_columns):
    """Compute phat from the proxy columns for all groups.

    Args: 
      df_proxy: pandas dataframe containing proxy columns, where each proxy
        column contains values 0 or 1 indicating noisy membership in a protected group.
      proxy_columns: list of strings. Names of the proxy columns.
      
    Returns:
      phats: 2D nparray with float32 values. Shape is number of groups * number of datapoints
      Each row represents \hat{p}_j for group j. Sum of each row is approximatedly 1.
    """
    num_groups = len(proxy_columns)
    num_datapoints = len(df_proxy)
    phats = np.zeros((num_groups, num_datapoints), dtype=np.float32)
    for i in range(num_groups):
        group_name = proxy_columns[i]
        group_size = df_proxy[group_name].sum()
        proxy_col = np.array(df_proxy[group_name])
        for j in range(num_datapoints):
            if proxy_col[j] == 1:
                phats[i, j] = float(1/group_size)
    return phats