# <img class='emoji_nav' src='emojis/world.svg' /> Live-share control files

<!-- Enable or disable live-sharing of running OmniOpt2 jobs with marker files -->

<!-- Category: Plotting and Sharing Results -->

<div id="toc"></div>

## Why control files?

Normally live-sharing is enabled when a run starts, with `--live_share`. Sometimes you
only notice later that a run should be shared, or you want to stop sharing a job that is
already running. Restarting the job just to change that is usually not an option.

OmniOpt2 therefore watches two marker files while it runs:

| File | Effect |
| --- | --- |
| `liveshare` | Enable live-sharing for the matching runs |
| `stop_liveshare` | Disable live-sharing for the matching runs |

The files only need to exist. Their content is ignored.

```bash
touch liveshare
```

No restart is required. OmniOpt2 checks the marker files about once per second.

## Where to put the files

The location of the marker file decides which runs it affects. More specific locations
win when several scopes exist, but `stop_liveshare` always wins over `liveshare`.

| Location | Affects |
| --- | --- |
| `runs/<experiment>/<run-number>/liveshare` | Only this single run |
| `runs/<experiment>/liveshare` | All runs of this experiment, including future ones |
| `liveshare` in the current working directory | All runs, including future ones |
| `liveshare` next to the `omniopt` script | All runs, including future ones |

For example, if the job is `runs/my_experiment/0`:

```bash
# Only runs/my_experiment/0
touch runs/my_experiment/0/liveshare

# Every run of my_experiment (running and future)
touch runs/my_experiment/liveshare

# Every run of every experiment (running and future)
touch liveshare
```

The same paths work for `stop_liveshare`:

```bash
touch runs/my_experiment/0/stop_liveshare
touch runs/my_experiment/stop_liveshare
touch stop_liveshare
```

## Deleting a marker does not undo it

The marker files are triggers, not permanent settings:

- Deleting `liveshare` **does not** stop sharing.
- Deleting `stop_liveshare` **does not** start sharing again.

This is intentional: create the opposite marker when you want to change the state.

```bash
touch liveshare          # start sharing
rm liveshare             # still shared
touch stop_liveshare     # stop sharing
rm stop_liveshare        # still stopped
touch liveshare          # start sharing again
```

## Choosing between scopes

Because `stop_liveshare` always wins, a global stop file pauses every run until it is
removed, even if a more specific `liveshare` file exists. This makes the stop file a
reliable emergency brake.

If you want to keep a global `liveshare` but stop only one experiment or run, use the
more specific `stop_liveshare` file. The global share stays enabled for all other runs.

## What happens in the terminal?

When a marker enables sharing, OmniOpt2 immediately uploads the current state and prints
the share URL. While a tqdm progress bar is visible, the message is written above the
bar, and the bar continues below it:

```text
 60%|██████░░░░| 3/5 [00:00<00:00, 29.30it/s]
LiveShare activated (global marker: /home/me/OmniOpt/liveshare)
LiveShare URL: https://imageseg.scads.de/omniax/share?user_id=me&experiment_name=my_experiment&run_nr=0
 60%|██████░░░░| 3/5 [00:00<00:00, 29.30it/s]
```

The activation message is printed only once per transition, not on every poll. After the
initial upload, live-share updates are pushed periodically, exactly like a run started
with [`--live_share`](tutorials?tutorial=oo_share).

## Relationship to `--live_share`

- A run started with `--live_share` is shared as usual.
- A run started **without** `--live_share` can be shared later with a `liveshare` file.
- A `stop_liveshare` file can stop a run that was started with `--live_share`.

## Caveats

<div class="caveat warning">
The running OmniOpt2 process must be able to see the marker files. On a cluster, create
them on the same shared filesystem that contains the OmniOpt2 installation and the
<tt>runs</tt> folder. The current working directory of the process and the directory of
the <tt>omniopt</tt> script are both checked.
</div>

<div class="caveat tip">
Marker files are listed in <tt>.gitignore</tt>, so accidentally committing them is not a
problem. They are meant to be temporary control files.
</div>

<div class="caveat warning">
Sharing still requires network access to the OmniOpt2-Share server. If the server is not
reachable, the activation message is printed without a URL and the periodic updater keeps
retrying.
</div>

## See also

- [Sharing results with OmniOpt2-Share](tutorials?tutorial=oo_share)
- [Folder structure of OmniOpt2 runs](tutorials?tutorial=folder_structure)
- [Environment variables](tutorials?tutorial=environment_variables)
