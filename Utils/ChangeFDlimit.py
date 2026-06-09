

def set_ulimit(limit):
    import resource

    current_limits = resource.getrlimit(resource.RLIMIT_NOFILE)
    print(f"Current limits: soft={current_limits[0]}, hard={current_limits[1]}")
    soft_limit = max(limit, current_limits[0])
    hard_limit = max(limit, current_limits[1])
    soft_limit = min(soft_limit, hard_limit)

    if soft_limit == hard_limit:
        print("At max limit")
        return
    try:
        print(f"Trying to update limits: soft={soft_limit}, hard={hard_limit}")
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))
        print(f"Updated limits: soft={soft_limit}, hard={hard_limit}")
    except ValueError as e:
        if current_limits[1] == limit:
            print(f"Failed to set limits: {e}")
        else:
            try:
                set_ulimit(current_limits[1])
            except Exception as e:
                print(f"Failed to set limits: {e}")
    except PermissionError as e:
        print(f"Permission denied: {e}")


