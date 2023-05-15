def get_algorithms_list():
    algorithms_desc = {
        "hyperopt.rand.suggest": "Random Search",
        "tpe.suggest": "Tree of Parzen Estimators",
        "gridsearch": "Gridsearch",
        "simulated_annealing": "Simulated Annealing"
    }
    return algorithms_desc

def get_range_generator_list():
    range_generator_dict = [
        {
            "name": "hp.choice",
            "parameters": "hp.choice(label, options)",
            "options": ("label", "options"),
            "description": "Returns one of the options, which should " +
            "be a list or tuple. The elements of options can themselves "+
            "be [nested] stochastic expressions. In this case, the "+
            "stochastic choices that only appear in some of the options "+
            "become conditional parameters."
        },
        {
            "name": "hp.pchoice",
            "parameters": "hp.pchoice(label, p_options)",
            "options": ("label", "p_options"),
            "description": "One of the option terms listed in p_options, "+
            "a list of pairs (prob, option) in which the sum of all prob "+
            "elements should sum to 1. The pchoice lets a user bias random "+
            "search to choose some options more often than others."
        },
        {
            "name": "hp.uniform",
            "parameters": "hp.uniform(label, low, high)",
            "options": ("label", "low", "high"),
            "description": "Uniformly between low and high. When optimizing, this variable is constrained to a two-sided interval."
        },
        {
            "name": "hp.quniform",
            "parameters": "hp.quniform(label, low, high, q)",
            "options": ("label", "low", "high", "q"),
            "description": "Drawn by round(uniform(low, high) / q) * q, "+
            "Suitable for a discrete value with respect to which the objective is still somewhat smooth."
        },
        {
            "name": "hp.loguniform",
            "parameters": "hp.loguniform(label, low, high)",
            "options": ("label", "low", "high"),
            "description": "Drawn by exp(uniform(low, high)). When optimizing, this variable is constrained to the interval [ e^{low}, e^{high}]."
        },
        {
            "name": "hp.qloguniform",
            "parameters": "hp.qloguniform(label, low, high, q)",
            "options": ("label", "low", "high", "q"),
            "description": "By round(exp(uniform(low, high)) / q) * q. Suitable for "+
            "a discrete variable with respect to which the objective is smooth and "+
            "gets smoother with the increasing size of the value."
        },
        {
            "name": "hp.normal",
            "parameters": "hp.normal(label, mu, sigma)",
            "options": ("label", "mu", "sigma"),
            "description": "A normally-distributed real value. When optimizing, this is an unconstrained variable."
        },
        {
            "name": "hp.qnormal",
            "parameters": "hp.qnormal(label, mu, sigma, q)",
            "options": ("label", "mu", "sigma", "q"),
            "description": "Drawn by round(normal(mu, sigma) / q) * q. Suitable for "+
            "a discrete variable that probably takes a value around mu, but is technically unbounded."
        },
        {
            "name": "hp.lognormal",
            "parameters": "hp.lognormal(label, mu, sigma)",
            "options": ("label", "mu", "sigma"),
            "description": "Drawn by exp(normal(mu, sigma)). When optimizing, this variable is constrained to be positive."
        },
        {
            "name": "hp.qlognormal",
            "parameters": "hp.qlognormal(label, mu, sigma, q)",
            "options": ("label", "mu", "sigma", "q"),
            "description": "Drawn by round(exp(normal(mu, sigma))/q)*q. Suitable for "+
            "a discrete variable with respect to which the objective is smooth and "+
            "gets smoother with the size of the variable, which is non-negative."
        },
        {
            "name": "hp.randint",
            "parameters": "hp.randint(label, upper)",
            "options": ("label", "upper"),
            "description": "Returns a random integer in the range [0, upper). "+
            "In contrast to quniform optimization algorithms should assume no "+
            "additional correlation in the loss function between nearby integer "+
            "values, as compared with more distant integer values (e.g. random seeds)."
        },
        {
            "name": "hp.uniformint",
            "parameters": "hp.uniformint(label, lower, upper)",
            "options": ("label", "lower", "upper"),
            "description": "Returns a uniform integer between lower and upper"
        }
    ]
    return range_generator_dict
