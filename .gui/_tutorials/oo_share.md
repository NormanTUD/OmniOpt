# OmniOpt2-Share

<!-- What is OmniOpt2-Share and how to use it? -->

<div id="toc"></div>

## What is OmniOpt2-Share?

OmniOpt2-Share allows you to Share your results with others, online. You can simply submit a job by

```bash
./omniopt_share runs/my_experiment/0
```

The program will upload the
job to our server, and allow give you a link to it which is valid for 30 days.

## Notes on Privacy

You can chose a random name to which OmniOpt2-Share should call you. But remember: the data you upload
is publically available for 30 days.

## `--help`

```run_php
	$file_path = "../omniopt_share";
	echo extract_help_params_from_bash($file_path);
```

## Run locally

It is possible to run OmniOpt-Share locally, via Docker.

```
cd .gui
docker-compose up --build
echo "http://localhost:8080/" > ~/.oo_base_url
```

From there on, you will push OmniOpt2-Share to your local machine, reachable under `localhost:8080`.
