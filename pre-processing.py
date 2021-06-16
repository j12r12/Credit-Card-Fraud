def preprocessing(working_df, vs, stationary=False, log=True):

    working_df = working_df[vs]
    class_y = working_df["Class"]
    working_df = working_df.drop("Class", axis=1)

    if log:
        working_df["Amount_Log"] = np.log(working_df["Amount"] + 0.00001)
        working_df.drop("Amount", axis=1, inplace=True)
    else:
        pass

    if stationary:
        for i in working_df.columns:
            working_df[i] = working_df[i].diff()
    else:
        pass

    working_df["Class"] = class_y

    working_df.dropna(axis=0, inplace=True)

    return working_df
