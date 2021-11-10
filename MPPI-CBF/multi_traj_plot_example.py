import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 200)
all_curves = np.array([np.sin(x * np.random.normal(1, 0.1)) * np.random.normal(1, 0.1) for _ in range(1000)])

confidence_interval1 = 95
confidence_interval2 = 80
confidence_interval3 = 50
for ci in [confidence_interval1, confidence_interval2, confidence_interval3]:
    low = np.percentile(all_curves, 50 - ci / 2, axis=0)
    high = np.percentile(all_curves, 50 + ci / 2, axis=0)
    plt.fill_between(x, low, high, color='r', alpha=0.2)
plt.plot(x, np.sin(x), color='b')
plt.margins(x=0)
plt.show()