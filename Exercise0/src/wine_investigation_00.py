import pandas as pd
import os


def main():

    # somehow it did not work like "../data/file.csv"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path_red = os.path.join(script_dir, '..', 'data', 'winequality-red.csv')
    csv_path_white = os.path.join(script_dir, '..', 'data', 'winequality-white.csv')

    # create the dataframes
    df_red = pd.read_csv(csv_path_red, delimiter=";")
    df_white = pd.read_csv(csv_path_white, delimiter=";")

    # print some beautiful basic info about the data sets (potentially to copy and paste)
    print("=== Red Wine DF ===")
    print('Length:', len(df_red))
    print('Shape:', df_red.shape)
    print('Info:', df_red.info())
    print('Basic statistic\n', df_red.describe())
    
    print("\n=== White Wine DF ===")
    print('Length:', len(df_white))
    print('Shape:', df_white.shape)
    print('Info:', df_white.info())
    print('Basic statistic\n', df_white.describe())

    print('\nRange of values for report')
    for col in df_white.columns:
        print(col)
        if df_white[col].min() < df_red[col].min():
            if df_white[col].max() > df_red[col].max():
                print("Range:", df_white[col].min(), "-", df_white[col].max(), "\n")
            else:
                print("Range:", df_white[col].min(), "-", df_red[col].max(), "\n")

        else:
            if df_white[col].max() > df_red[col].max():
                print("Range:", df_red[col].min(), "-", df_white[col].max(), "\n")
            else:
                print("Range:", df_red[col].min(), "-", df_red[col].max(), "\n")

    return 0




if __name__ == "__main__":
    main()