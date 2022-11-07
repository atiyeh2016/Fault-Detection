import matplotlib.pyplot as plt
import matplotlib


fig = plt.figure()
ax = fig.gca()
ax.plot(loss_trend, linewidth=5)
plt.title('Train Loss')
matplotlib.rc('font', size=14)

