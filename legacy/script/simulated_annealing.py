from pprint import pprint
import numpy as np
import math
from hyperopt import hp, fmin, tpe, anneal, Trials, pyll
from hyperopt.base import miscs_update_idxs_vals
from hyperopt.pyll.base import dfs, as_apply
from hyperopt.pyll.stochastic import implicit_stochastic_symbols

class SimulatedAnnealingSearchError(Exception):
    pass

def validate_space_simulated_annealing(space):
    print("Checking space for simulated annealing")
    supported_stochastic_symbols = ['randint', 'quniform', 'qloguniform', 'qnormal', 'qlognormal', 'categorical', 'choice']
    for node in dfs(as_apply(space)):
        if node.name in implicit_stochastic_symbols:
            if node.name not in supported_stochastic_symbols:
                raise SimulatedAnnealingSearchError('Simulated annealing search is only possible with the following stochastic symbols: ' + ', '.join(supported_stochastic_symbols))
    print("Checking space for simulated annealing done")

def simulated_annealing(new_ids, domain, trials, seed, T0=100, T_min=1e-4, alpha=0.99, max_evals=1000):
    print("simulated_annealing(...)");
    # Build a hash set for previous trials
    hashset = set([hash(frozenset([(key, value[0]) if len(value) > 0 else ((key, None))
                                   for key, value in trial['misc']['vals'].items()])) for trial in trials.trials])

    rng = np.random.default_rng(seed)
    rval = []
    print("Enumerating new IDs")
    for i, new_id in enumerate(new_ids):
        T = T0
        x = None
        print("For each in max evals")
        for j in range(max_evals):
            newSample = False
            while not newSample:
                print("While not newSample")
                # -- sample new specs, idxs, vals
                idxs, vals = pyll.rec_eval(
                    domain.s_idxs_vals,
                    memo={
                        domain.s_new_ids: [new_id],
                        domain.s_rng: rng,
                    })
                x = dict(zip(idxs, vals))
                new_result = domain.new_result()
                new_misc = dict(tid=new_id, cmd=domain.cmd, workdir=domain.workdir)
                miscs_update_idxs_vals([new_misc], idxs, vals)

                # Compare with previous hashes
                print("h = hash");
                h = hash(frozenset([(key, value[0]) if len(value) > 0 else (
                    (key, None)) for key, value in vals.items()]))
                print("h not in hashset");
                if h not in hashset:
                    newSample = True

            rval.extend(trials.new_trial_docs([new_id],
                                            [None], [new_result], [new_misc]))
            hashset.add(h)
            print("Added hashset")

            T = T * alpha
            if T < T_min:
                break

            print("Choose a new point randomly")
            x_new = {}
            for key in x:
                x_new[key] = np.random.normal(0, T)
     
            print("x_new:")
            pprint(type(x_new))
            pprint(x_new)
            print("<<<<<<<<<<<<<<<<")

            print("Evaluate the new point")
            print("idxs_new")
            #idxs_new = {i for i in range(len(x_new))}
            idxs_new = list(x_new.keys())

            print("vals_new")

            vals_new = x_new # list(x_new.values())
            pprint(idxs_new)
            pprint(vals_new)

            new_misc = dict(tid=-1, cmd=domain.cmd, workdir=domain.workdir)
            miscs_update_idxs_vals([new_misc], idxs_new, vals_new)
            new_sample = trials.new_trial_docs([None], [None], [None], [new_misc])[0]
            loss_new = domain.evaluate(new_sample)

            print("Compare with previous hashes")
            hash_new = hash(frozenset([(key, val[0]) if len(val) > 0 else ((key, None)) for key, val in vals.items()]))
            if hash_new not in hashset:
                print("Evaluate the new sample")
                new_misc['cmd'] = 'simulated_annealing'
                loss = domain.evaluate(vals)[0]
                new_result = domain.new_result()
                new_result['loss'] = loss
                new_result['status'] = 'ok'
                rval.extend(trials.new_trial_docs([new_id], [None], [new_result], [new_misc]))
                hashset.add(hash_new)
            else:
                print("Check if we should keep this sample using simulated annealing")
                old_misc = [trial['misc'] for trial in trials.trials if hash(frozenset([(key, val[0]) if len(val) > 0 else ((key, None)) for key, val in trial['misc']['vals'].items()])) == hash_new][0]
                old_loss = [trial['result']['loss'] for trial in trials.trials if hash(frozenset([(key, val[0]) if len(val) > 0 else ((key, None)) for key, val in trial['misc']['vals'].items()])) == hash_new][0]
                T = anneal_temperature(num_samples=len(trials.trials), cooling_factor=0.1)
                delta_loss = old_loss - loss
                prob_accept = np.exp(-delta_loss / T)
                if prob_accept > np.random.random():
                    print("if prob_accept > np.random.random():")
                    new_misc['cmd'] = 'simulated_annealing'
                    new_result = domain.new_result()
                    new_result['loss'] = loss
                    new_result['status'] = 'ok'
                    rval.extend(trials.new_trial_docs([new_id], [None], [new_result], [new_misc]))
                print("Done simulated annealing 1");
            print("Done simulated annealing 2");
        print("Done simulated annealing 3");
    print("Done simulated annealing 4");
