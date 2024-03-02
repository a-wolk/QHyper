def generate_qbits(prefix, n):
    return list(map(lambda v: f'{prefix}{v}', range(n)))[::-1]

def gen_lambda(func):
    def func_wrapper(*args, **kwargs):
        return lambda: func(*args, **kwargs)
    return func_wrapper