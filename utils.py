def humanize_tuple(tup: tuple):
    if len(tup) == 1:
        return str(tup[0])
    elif len(tup) == 2:
        return f"{tup[0]} and {tup[1]}"
    else:
        return ", ".join(tuple[:-1]) + f", and {tup[-1]}"