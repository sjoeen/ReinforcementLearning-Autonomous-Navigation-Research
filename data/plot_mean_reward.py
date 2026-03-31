from tbparse import SummaryReader
import matplotlib.pyplot as plt

reader = SummaryReader("logs/ppo.log", pivot=True)
df = reader.scalars

print(df.head())

plt.plot(df["step"], df["rollout/ep_rew_mean"])
plt.xlabel("Timesteps")
plt.ylabel("Mean Episode Reward")
plt.title("Reward Over Time in Training")
plt.grid(True)
plt.savefig("reward_plot.pdf", bbox_inches="tight")
plt.show()
