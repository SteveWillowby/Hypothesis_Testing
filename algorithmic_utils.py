
# Returns the first item in a list meeting a criterion
#
# the_list -- a list of elements to search through
# the_criterion -- a function which can be applied to an element of the list
#                  returns true or false
#
# Elements in the_list should go from not meeting the criterion to meeting it
#   (or all should meet it)
def binary_search_for_first_meeting_criterion(the_list, the_criterion):
    assert len(the_list) > 0
    assert the_criterion(the_list[-1])

    low = 0
    high = len(the_list)
    mid = low + int((high - low) / 2)

    while low < high and not the_criterion(the_list[low]):
        if the_criterion(the_list[mid]):
            high = mid
            mid = low + int((high - low) / 2)
        else:
            low = mid
            mid = low + int((high - low) / 2)

    return the_list[low]

# Returns the last item in a list meeting a criterion
#
# the_list -- a list of elements to search through
# the_criterion -- a function which can be applied to an element of the list
#                  returns true or false
#
# Elements in the_list should go from meeting the criterion to not meeting it
#   (or all should meet it)
def binary_search_for_last_meeting_criterion(the_list, the_criterion):
    assert len(the_list) > 0
    assert the_criterion(the_list[0])

    low = 0
    high = len(the_list)
    mid = low + int((high - low) / 2)

    while low < high and not the_criterion(the_list[high - 1]):
        if the_criterion(the_list[mid]):
            low = mid
            mid = low + int((high - low) / 2)
        else:
            high = mid
            mid = low + int((high - low) / 2)

    return the_list[high - 1]

if __name__ == "__main__":
    criterion = (lambda x: x > 10)
    print(11 == binary_search_for_last_meeting_criterion([13, 12, 15, 16, 17, 11, 4, 5], criterion))
    print(11 == binary_search_for_last_meeting_criterion([13, 12, 15, 16, 17, 11, 4, 5, 5, 6, 10, 1, 2, 3, 4, 1, 4], criterion))
    print(11 == binary_search_for_last_meeting_criterion([11, 4, 5, 5, 6, 10, 1, 2, 3, 4, 1, 4], criterion))
    print(11 == binary_search_for_last_meeting_criterion([13, 12, 15, 16, 17, 11], criterion))
