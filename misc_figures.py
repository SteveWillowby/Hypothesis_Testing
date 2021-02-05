import matplotlib.pyplot as plt
import bigfloat
import math
from comb_and_prob_funcs import bigfloat_prob_of_count_given_p  # C p S

def str_func(i):
    return " " * i
x_axis = [str_func(i) for i in range(0, 41)]
x_axis[0] = "Category 1"
x_axis[6] = "Category 7"
x_axis[-1] = "Category 601"
y_axis = [1.0 - 600.0 * (1.0 / 1800)] + [1.0 / 1800 for i in range(0, 40)]

plt.bar(x_axis, y_axis)
plt.show()

p = 0.499893
x_axis = [i for i in range(0, 601)]
y_axis = [bigfloat_prob_of_count_given_p(i, p, 600) for i in range(0, 601)]
color = []
for i in range(0, 601):
    if y_axis[i] < 1.0 / 1800:
        color.append("red")
    else:
        color.append("blue")

y_axis_min = bigfloat.min(y_axis[0], y_axis[-1])

# y_axis = [bigfloat.log(value) - bigfloat.log(y_axis_min) for value in y_axis]

x_axis = [-1] + x_axis + [601]
y_axis = [1.0 / 1800] + y_axis + [1.0 / 1800]

plt.bar(x_axis, y_axis, color=color)
plt.ylabel("P(X | p = 0.499893)")
plt.xlabel("X")
plt.show()
