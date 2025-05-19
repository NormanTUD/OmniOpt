# <span class="tutorial_icon invert_in_dark_mode">🌍</span> OmniOpt2-Share

<!-- What is OmniOpt2-Share and how to use it? -->

<!-- Category: Plotting and Sharing Results -->

<div id="toc"></div>

## What is OmniOpt2-Share?

OmniOpt2-Share allows you to Share your results with others, online. You can simply submit a job by

```bash
./omniopt_share runs/my_experiment/0
```

The program will upload the
job to our server, and allow give you a link to it which is valid for 30 days.

## `--help`

```run_php
	$file_path = "../omniopt_share";
	echo extract_help_params_from_bash($file_path);
```

## Run locally in Docker

It is possible to run OmniOpt-Share locally, via Docker.

```bash
cd .gui
bash docker.sh --local-port 1234 --shares-path /tmp/shares_stuff
echo "http://localhost:1234/" > ~/.oo_base_url
```

From there on, you will push OmniOpt2-Share to your local machine, reachable under `localhost:1234`.

### Download all exports

If you want to download a lot of exports, you can first run the local docker installation. Submit every job you want to the local OmniOpt2 installation (see the previous point).

You can then download all exports as single HTML files by simply doing this:


```bash
cd .gui
bash download_local_exports --export_dir /home/YourUsername/test/randomtest_98580
```

The `download_local_exports` script has some options to filter which jobs you want to export:

```run_php
	$file_path = "download_local_exports";
	echo extract_help_params_from_bash($file_path);
```

Run Parameters *without* equal signs, like this:

```bash
bash download_local_exports --export_dir /home/YourUsername/test/randomtest_98580 --user YourUsername --experiment MY_EXPERIMENT_NAME
```

## Download exported shares via curl

If you want to automatically download exported shares, ie. over curl, you can do the following:

```bash
URL="https://imageseg.scads.de/omniax/share?user_id=norman&sort=time_desc&experiment_name=__main__tests__BOTORCH_MODULAR___nogridsearch_nr_results_2&sort=time_desc&run_nr=0&sort=time_desc"
curl $URL | awk '/<!-- export.html -->/{p=!p; next} p' | perl -MHTML::Entities -pe 'decode_entities($_)' > name_of_your_exported_file.html
```

## Notes on Privacy
<div class="caveat tip">
You can chose a random name to which OmniOpt2-Share should call you. But remember: the data you upload
is publically available for 30 days.
</div>
