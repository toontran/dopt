from dopt import NEIOptimizer

if __name__ == "__main__":
    bounds = {
        'x1': (0,1),
        'x2': (0,1),
        'x3': (0,1),
        'x4': (0,1),
        'x5': (0,1),
        'x6': (0,1),
    }
    optimizer = NEIOptimizer("hartmann.json", bounds, device="cpu")
    optimizer.run(host=None)
