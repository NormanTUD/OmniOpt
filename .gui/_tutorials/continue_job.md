# Continue jobs

<!-- How to continue jobs after they have been run already -->

<div id="toc"></div>

## Continue an old job with the same options

Continuing an old job with the same options as previously, but with awareness of the hyperparameter-constellations
that have already been tested, is as simple as this, assuming your job is in `runs/my_experiment/0`:

```
./omniopt --continue runs/my_experiment/0
```

This will start a new run with the same settings as the old one, but load all already tried out data points, and
continue the search from there.

## Continue an old job with the changed options

In continued runs, some options can be changed. For example,

```
./omniopt --continue runs/my_experiment/0 --time=360
```

Will run the continuation for 6 hours ( = 360 minutes), independent of how long the old job ran.

It is also possible to change parameter borders, though narrowing is not currently supported, i.e.
the parameters need to be equal or wider than in the previous run. You cannot remove parameters here or add
names of parameters that have previously not existed, though.

Also, parameter names and types must stay the same. This code, for example, runs a continued run but changes
the `epochs` parameter that was previously defined and could have been, for example, between 0 and 10. In
this new run, all old values will be considered, but new values of epochs can be between 0 and 1000.

```
./omniopt --continue runs/my_experiment/0 --parameter epochs range 0 1000 int
```

All parameters that are not specified here are taken out of the old run, and thus stay in the same borders.

## In which folder will a run continue in?

It will create a new folder. Imagine there are already the subfolders `0`, `1` and `2` for your
experiment. If you continue the job `0`, it's job data will be in the subfolder `3` then, since it is the first
non-existing folder for that project..

## How to continue shared run

You can also continue runs from an OmniOpt-Share-URL, like:

```
./omniopt --continue https://imageseg.scads.de/omniax/share.php?user_id=s3811141&experiment_name=__main__tests__&run_nr=14 --time=360
```

You can set the variables as if this was a normal continued run.

## Caveat

It is currently not possible to decrease the search space on a continued run. Attempts to do that will be ignored and the
original limits will automatically be restored.
