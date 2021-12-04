import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("seaborn-dark")



if __name__ == "__main__":
    df = pd.read_csv("./td3.csv")
    
    plt.figure()
    sns.lineplot(x = df["Step"], y = df["Value"])
    plt.grid(True)
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.title("Training progress for Twin Delayed DDPG")
    #plt.show()

    plt.savefig("/home/rohan/reinforcement_learning/drl_hands_on/code/assets/td3.jpg")