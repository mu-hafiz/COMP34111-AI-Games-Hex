def allocate_thinking_time(
    time_used,
    move_number,
    max_time=280,
    max_moves=100,
    C=20,
    C_multiplier=0.5,
    behind=False
):
    # Increase conservativeness when behind
    C = C * C_multiplier if behind else C
    remaining_time = max_time - time_used
    return remaining_time / (C + max(max_moves - move_number, 0))

time_spent = 0
for turn in range(1, 121, 2):
    
    print(f"Turn {turn}:")
    normal_time = allocate_thinking_time(time_spent, turn)
    conservative_time = allocate_thinking_time(time_spent, turn, behind=True)
    time_spent += normal_time

    print(f"   Normal: {normal_time}")
    print(f"   When Behind: {conservative_time}")