FROM debian:bookworm

RUN apt-get update
RUN apt-get install -y python3 python3-pip python3-venv xterm git

COPY . /var/opt/omniopt/
RUN bash /var/opt/omniopt/.shellscript_functions
#RUN bash /var/opt/omniopt/omniopt --tests --num_parallel_jobs=1 --max_eval=1 --worker_timeout=1 --run_program "" --experiment_name ""


#CMD ["apache2-foreground"]
