import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# plot_3d_surface():
#
# left_axis_values -- values of variable to be shown on left side of figure,
#                     in order from right to left (i.e. from near to far).
#
# right_axis_values -- values of variable to be shown on right side of figure,
#                      in order from left to right (i.e. from near to far).
#
# z_axis_function -- function which takes a (left-axis-value, right-axis-value)
#                    point and computes the surface height for that point
#
# left_axis_name -- label for the left axis
# right_axis_name -- label for the right axis
# plot_title -- appears at top of plot (technically, this is the "suptitle")
# plot_subtitle -- appears below title, at top of plot
#
# surface_cmap_name -- name of a matplotlib cmap -- determines the color of the
#                      surface based on the value of the z axis -- to see a list
#                      of the color maps, check out:
#           https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
#
# surface_alpha -- determines transparency of surface: 1.0 means no transparency
#                                                      0.0 means invisible
#
# marked_points -- gives the user the option to provide a list of
#                  (left-axis-value, right-axis-value) points, which will have
#                  their location on the surface highlighted. If using these,
#                  make sure to set the surface alpha below 1.0
#
# marked_points_c -- a color to give the marked points
#
# marked_points_alpha -- like `surface_alpha`, but for the marked points
def plot_3d_surface(left_axis_values, right_axis_values, z_axis_function, \
        left_axis_name="", right_axis_name="", plot_title="", plot_subtitle="",\
        surface_cmap_name='RdBu_r', surface_alpha=1.0, \
        marked_points=[], marked_points_c='green', marked_points_alpha=1.0):

    surface_cmap = plt.get_cmap(surface_cmap_name)

    surface_values = []
    for lav in left_axis_values:
        surface_values.append([])
        for rav in right_axis_values:
            surface_values[-1].append(float(z_axis_function((lav, rav))))

    
    # Have the "major" axis be left vals and the "secondary" axis be right vals
    right_vals_2d = np.array([[float(x) for x in right_axis_values] \
        for _ in left_axis_values])
    left_vals_2d = np.array([[float(left_axis_values[i]) for _ in right_axis_values] for \
        i in range(0, len(left_axis_values))])

    surface_values = np.array(surface_values)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    marked_x = [a for (a, _) in marked_points]
    marked_y = [b for (_, b) in marked_points]
    marked_z = [float(z_axis_function((a, b))) for (a, b) in marked_points]

    points_plot = ax.scatter(marked_x, marked_y, marked_z, \
        c=marked_points_c, alpha=marked_points_alpha)

    surface_plot = ax.plot_surface(left_vals_2d, right_vals_2d, surface_values, \
               linewidth=0, antialiased=False, \
               cmap=surface_cmap, alpha=surface_alpha)

    if left_axis_values[0] < left_axis_values[-1]:
        # ax.invert_xaxis()  # performed by the next line
        ax.set_xlim([float(left_axis_values[0]), float(left_axis_values[-1])])
    else:
        ax.set_xlim([float(left_axis_values[-1]), float(left_axis_values[0])])

    if right_axis_values[0] > right_axis_values[-1]:
        # ax.invert_yaxis()  # performed by the next line
        ax.set_ylim([float(right_axis_values[0]), float(right_axis_values[-1])])
    else:
        ax.set_ylim([float(right_axis_values[-1]), float(right_axis_values[0])])

    plt.suptitle(plot_title)
    plt.xlabel(left_axis_name)
    plt.ylabel(right_axis_name)
    plt.title(plot_subtitle)

    plt.show()

def save_figure_with_data_csv(the_plt, figure_name, x_axes, y_axes):
    plt.savefig(figure_name + ".pdf")

    num_rows = max([len(axis) for axis in x_axes])

    with open(figure_name + ".csv", 'w') as list_file:
        writer = csv.writer(list_file, delimiter=',')
        for row_idx in range(0, num_rows):
            row = []
            for half_col_idx in range(0, len(x_axes)):
                if len(x_axes[half_col_idx]) > row_idx:
                    row.append(x_axes[half_col_idx][row_idx])
                    row.append(y_axes[half_col_idx][row_idx])
                else:
                    row += ["", ""]

            writer.writerow(row)
