def is_integer(var, var_name):
    if not isinstance(var, int):
        print(f'{var_name} must be integer')
        exit()
def is_positive(var, var_name):
    if not var > 0:
        print(f'{var_name} must be positive')
        exit()

def calculate_F1(true_positive=1, false_positive=1, false_negative=1):
    is_integer(true_positive, "true_positive")
    is_integer(false_positive, "false_positive")
    is_integer(false_negative, "false_negative")
    is_positive(true_positive, "true_positive")
    is_positive(false_negative, "false_negative")
    is_positive(false_positive, "false_positive")
    precision=true_positive/(true_positive+false_positive)
    recall=true_positive/(true_positive+false_negative)
    f1_score=2*((precision*recall)/(precision+recall))
    print(f'precison is {precision}')
    print(f'recall is {recall}')
    print(f'f1-score is {f1_score}')
    return precision, recall, f1_score


calculate_F1(true_positive=1, false_positive=3, false_negative=1)






