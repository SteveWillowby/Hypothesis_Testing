import bigfloat

# Assumes function is convex
def binary_min_finder(func, low, high, tol=bigfloat.BigFloat(2.0**(-30)), error_depth=1):
    low = bigfloat.BigFloat(low)
    high = bigfloat.BigFloat(high)
    mid = low + ((high - low) / 2.0)

    low_func = func(low)
    high_func = func(high)
    mid_func = func(mid)

    low_func >= mid_func or high_func >= mid_func

    val_func_pairs = [(low, low_func), (high, high_func), (mid, mid_func)]

    best_arg = mid
    best_func = mid_func
    if low_func < best_func:
        best_arg = low
        best_func = low_func
    if high_func < best_func:
        best_arg = high
        best_func = high_func

    iters = 0

    while high - low > tol:
        # print("  Remaining: %f" % ((high - low) - tol))
        left_mid = low + ((mid - low) / 2.0)
        right_mid = mid + ((high - mid) / 2.0)

        left_mid_func = func(left_mid)
        right_mid_func = func(right_mid)

        val_func_pairs.append((left_mid, left_mid_func))
        val_func_pairs.append((right_mid, right_mid_func))

        if left_mid_func < right_mid_func and left_mid_func < best_func:
            # print("A entails...")
            best_func = left_mid_func
            best_arg = left_mid
        elif right_mid_func < best_func:
            # print("B entails...")
            best_func = right_mid_func
            best_arg = right_mid
        else:
            pass
            # print("C is free...")

        if mid_func < left_mid_func and mid_func < right_mid_func:
            high = right_mid
            low = left_mid
            high_func = right_mid_func
            low_func = left_mid_func
        elif right_mid_func < mid_func and mid_func < left_mid_func:
            low = mid
            low_func = mid_func
            mid = right_mid
            mid_func = right_mid_func
        elif right_mid_func > mid_func and mid_func > left_mid_func:
            high = mid
            high_func = mid_func
            mid = left_mid
            mid_func = left_mid_func
        elif right_mid_func == mid_func and mid_func == left_mid_func:
            print("Wow!")
            high = mid
            high_func = mid_func
            low = mid
            low_func = mid_func
        elif right_mid_func == mid_func and mid_func < left_mid_func:
            low = mid
            low_func = mid_func
            mid = right_mid
            mid_func = right_mid_func
        elif left_mid_func == mid_func and mid_func < right_mid_func:
            high = mid
            high_func = mid_func
            mid = left_mid
            mid_func = left_mid_func
        elif right_mid_func == mid_func:
            assert mid_func > left_mid_func
            high = mid
            high_func = mid_func
            mid = left_mid
            mid_func = left_mid_func
        elif left_mid_func == mid_func:
            assert mid_func > right_mid_func
            low = mid
            low_func = mid_func
            mid = right_mid
            mid_func = right_mid_func
        else:
            assert mid_func > left_mid_func and mid_func > right_mid_func
            assert left_mid != mid and right_mid != mid

            """
            print("Error! Convexity assumption broken!")
            print("  low    \tleft_mid \tmid     \tright_mid \thigh")
            print("  %f \t%f \t%f \t%f \t%f" % (low, left_mid, mid, right_mid, high))
            print("  %f \t%f \t%f \t%f \t%f" % (low_func, left_mid_func, mid_func, right_mid_func, high_func))
            return (best_arg, best_func)
            """
            print("Error! Convexity assumption broken at a depth of %d" % error_depth)
            if float(left_mid_func) == float(right_mid_func):
                print("Results are close enough to quit.")
                if left_mid_func < right_mid_func:
                    return (left_mid, left_mid_func)
                else:
                    return (right_mid, right_mid_func)
            print("  low    \tleft_mid \tmid     \tright_mid \thigh")
            print("  %f \t%f \t%f \t%f \t%f" % (low, left_mid, mid, right_mid, high))
            print("  %f \t%f \t%f \t%f \t%f" % (low_func, left_mid_func, mid_func, right_mid_func, high_func))
            (best_arg_left, best_func_left) = binary_min_finder(func, low, mid, tol=tol, error_depth=(error_depth + 1))
            (best_arg_right, best_func_right) = binary_min_finder(func, mid, high, tol=tol, error_depth=(error_depth + 1))
            if best_func_left < best_func_right:
                return (best_arg_left, best_func_left)
            else:
                return (best_arg_right, best_func_right)

        iters += 1
    # print("Iters: %d" % iters)

    """
    val_func_pairs.sort()
    x_axis = [a for (a, b) in val_func_pairs]
    y_axis = [b for (a, b) in val_func_pairs]
    plt.plot(x_axis, y_axis)
    plt.title("Best of %f at %f" % (best_func, best_arg))
    plt.show()
    """

    return (best_arg, best_func)


# func must be convex
def min_finder_low_bound_only(func, low_arg, tol=bigfloat.BigFloat(2.0**(-30))):
    prev_prev_arg = low_arg
    prev_arg = low_arg
    curr_arg = low_arg

    prev_func = func(curr_arg)
    curr_func = prev_func

    while curr_func <= prev_func:
        prev_prev_arg = prev_arg
        prev_arg = curr_arg
        curr_arg *= 2.0

        prev_func = curr_func
        curr_func = func(curr_arg)

    # return (best_arg, best_func)
    return binary_min_finder(func, low=prev_prev_arg, high=curr_arg, tol=tol)

def max_finder_low_bound_only(func, low_arg, tol=bigfloat.BigFloat(2.0**(-30))):
    func_prime = (lambda x: -1.0 * func(x))
    (best_arg, best_func) = min_finder_low_bound_only(func_prime, low_arg, tol=tol)
    return (best_arg, -1.0 * best_func)
