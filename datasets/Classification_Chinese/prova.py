import pandas as pd
def main() :
    values = pd.read_csv("./datasets/Classification_Chinese/compressed_dataset.csv", compression = "gzip")["label"].unique()
    print(values)
    

if __name__ == "__main__" :
    main()