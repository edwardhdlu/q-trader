from math import sin,pi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# parameters
PER = 20
RPT = 10
output_path = 'files/input/'

x = np.linspace(0, 2*pi, PER, endpoint=False)
xsin = [10+sin(i) for i in x]*RPT

plt.plot(xsin)
plt.show()

dummy_cols = pd.DataFrame(np.ones((len(xsin), 4)))
test_df = dummy_cols.join(pd.DataFrame(xsin, columns=['Val']))
test_df.to_csv(output_path+"test_sinus.csv", index=False)

xrise = [i for i in np.linspace(100, 200, PER*RPT, endpoint=False)]

plt.plot(xrise)
plt.show()

dummy_cols = pd.DataFrame(np.ones((len(xrise), 4)))
test_df = dummy_cols.join(pd.DataFrame(xrise, columns=['Val']))
test_df.to_csv(output_path+"test_rise_only.csv", index=False)
