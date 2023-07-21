from bayespso.tools import bayesian_skOpt as bs
from bayespso.tools import rosenbrock as rt
import json
import os


settings = {
    'seed': 112,
    'nparallel_eval': 4,
    'niter': 50,
    'output_dir': '/home/user/tmp/'
}



output_path = '/home/user/tmp/some_info.json'
evol_path = '/home/user/tmp/evol_info.txt'



def main():
    best_value, best_hyperparameters  = bs.optimize(
        hyperparameters, rt.rosenbrock_function, settings
    )
    print(best_value, best_hyperparameters)

if __name__ == '__main__':
    main()








# Nagu Veelkeniga sai rägitud, siis initial pointid on selles implementatsioonis 
# võimalik genereerida mingi sequence järgi:
#     - kas on sama mis seal Jones artiklis? Et sequence, mitte mingi kavalam
#         algo?

# acq. funktsiooni hinnatakse numbriliselt selles implementatsioonis, kui just
# ei kasutata LBFGS.

# Vaja veel leida nö muud settingud nendel funktsioonidel

# TODO:
#     - leida settingud
#     - kavalam meetod kuidas valida paralleelselt punkte



# strategy : string, default: "cl_min"
#     Method to use to sample multiple points (see also `n_points`
#     description). This parameter is ignored if n_points = None.
#     Supported options are `"cl_min"`, `"cl_mean"` or `"cl_max"`.

#     - If set to `"cl_min"`, then constant liar strategy is used
#        with lie objective value being minimum of observed objective
#        values. `"cl_mean"` and `"cl_max"` means mean and max of values
#        respectively. For details on this strategy see:

#        https://hal.archives-ouvertes.fr/hal-00732512/document


