#!/usr/bin/perl

#SBATCH --signal=B:USR1@120

use strict;
use warnings FATAL => 'all';

use POSIX qw(strftime);
use Sys::Hostname;
use File::Path qw(make_path);
use Data::Dumper;
use Term::ANSIColor;
use Hash::Util qw(lock_keys);
use autodie;
use File::Basename;
use IO::Socket::INET;
use Carp;
use Cwd;
use File::Copy;
use Memoize;

sub std_print (@);

memoize 'get_log_path_date_folder';
memoize 'get_project_folder';

my $indentation_char = "─";
my $indentation = 0;

use lib './perllib';
require OmniOptLogo;
use Env::Modify;

our $debug_log_file = '';
$debug_log_file = get_log_file_name();

sub std_print (@) {
        foreach my $msg (@_) {
                print $msg;
                if($debug_log_file) {
                        open my $fh, '>>', $debug_log_file;
                        print $fh $msg;
                        close $fh;
                }
        }
}

std_print color('underline')."Log file: $debug_log_file".color('reset')."\n";

show_logo();

use constant {
        DEFAULT_MIN_RAND_GENERATOR => 2_048,
        DEFAULT_MAX_RAND_GENERATOR => 65_500,
        MIN_RECOMMENDED_MEM_PER_CPU => 1_024,
        MAX_WORKER_WITHOUT_WARNING => 160
};

my $ran_anything = 0;
my ($master_ip, $master_port) = (undef, undef);
my $received_exit_signal = 0;

sub debug (@);
sub debug_sub (@);
sub error (@);
sub no_suicide_error (@);
sub debug_system ($);
sub warning (@);
sub dryrun (@);
sub message (@);
sub ok (@);
sub ok_debug (@);

sub usr_signal {
        debug 'usr_signal()';
        $received_exit_signal = 1;
        shutdown_script();
}

sub shutdown_script () {
        debug 'shutdown_script()';
        if($ran_anything) {
                get_dmesg('end');
                backup_mongo_db();
                end_mongo_db();
        }
}

my %default_values = (
        mempercpu => 2_000,
        number_of_allocated_gpus => 0,
        worker => 1,
        sleep_nvidia_smi => 10,
        sleep_db_up => 30
);

lock_keys(%default_values);

my %options = (
        project => undef,
        projectpath => undef,
        logpathdate => undef,
        worker => $default_values{worker},
        debug => 0,
        slurmid => $ENV{SLURM_JOB_ID},
        originalslurmid => $ENV{SLURM_JOB_ID},
        dryrun => 0,
        sleepafterfmin => 10,
        mempercpu => $default_values{mempercpu},
        dryrunmsgs => 1,
        warnings => 1,
        messages => 1,
        sanitycheck => 1,
        number_of_allocated_gpus => $default_values{number_of_allocated_gpus},
        run_nvidia_smi => 1,
        keep_db_up => 1,
        sleep_nvidia_smi => $default_values{sleep_nvidia_smi},
        sleep_db_up => $default_values{sleep_db_up},
        debug_srun => 0,
        projectdir => get_working_directory().'/projects/',
        install_despite_dryrun => 0
);

lock_keys(%options);

my $python = "python3";

my $python_module = "PYTHONPATH=\$PYTHONPATH:".get_working_directory()."/script/ ";

my $this_cwd = get_working_directory();
my $script_folder = $this_cwd.'/script';
my $tools_folder = $this_cwd.'/tools';
my $test_folder = $this_cwd.'/test';

my %script_paths = (
        testscript => "$test_folder/tests.py",
        worker => "$script_folder/worker.py",
        endmongodb => "$script_folder/endmongodb.py",
        fmin => "$script_folder/fmin.py",
        mongodb => "$script_folder/mongodb.py",
        backupmongodb => "$script_folder/backupmongodb.py",
        show_sys_path => "$script_folder/show_sys_path.py",
        loggpu => "$tools_folder/loggpu.sh"
);

lock_keys(%script_paths);

get_environment_variables();
analyze_args(@ARGV);

debug_sub 'Checking paths of script_paths';
foreach my $name (keys %script_paths) {
        my $path = $script_paths{$name};

        if(-e $path) {
                debug "$name: $path -> exists";
        } else {
                error "$name: $path -> does not exist!";
        }
}

if(!$options{projectpath}) {
        $options{projectpath} = get_project_folder($options{project});
        if($options{projectpath} !~ m#/$#) {
                $options{projectpath} .= "/";
        }
}
$options{logpathdate} = get_log_path_date_folder($options{project});

debug_sub 'Outside of any function';
debug 'Locking keys of %options';
debug_options();
get_dmesg_start();

END {
        debug_sub 'END-BLOCK';
        if($ran_anything) {
                if($options{debug}) {
                        get_dmesg('end');
                }
        }
        shutdown_script();
}

$SIG{USR1} = $SIG{USR2} = \&usr_signal;

debug split(/\R/, Dumper \%ENV);

main();

sub main {
        $indentation++;
        debug_sub 'main()';

        cancel_message();
        sanity_check();

        my %ini_config = read_ini_file("$options{projectpath}/config.ini");
        if(!$options{debug} && exists $ini_config{DEBUG} && exists $ini_config{DEBUG}{debug} && $ini_config{DEBUG}{debug}) {
                $options{debug} = 1;
                debug "Enabled debug because it was enabled in the config.ini";
        }

        create_and_delete_random_files_in_subfolders();

        set_python_path();

        environment_sanity_check();

        create_paths();
        install_needed_packages();
        show_sys_path() if $options{debug};

        if(!defined(mongodb_already_started())) {
                start_mongo_db_fork();
                start_fmin_fork();
        }

        if(start_worker_fork()) {
                if(!$options{dryrun}) {
                        sleep 10;
                }

                run_nvidia_smi_periodically();
                keep_db_up();
                wait_for_unfinished_jobs();
        } else {
                warning "It seems that no job could be started!";
        }

        $indentation--;
}
sub environment_sanity_check {
        $indentation++;
        my $errors = 0;
        debug_sub "environment_sanity_check()";

        check_needed_programs();

        if(!program_installed($python)) {
                error "$python not found!";
                $errors++;
        }

        debug_loaded_modules();

        $indentation--;
        return $errors;
}

sub show_sys_path {
        $indentation++;
        debug_sub "show_sys_path()";
        system("whereis $python");
        system("$python $script_paths{show_sys_path}");
        $indentation--;
}

sub mongodb_already_started {
        $indentation++;
        debug_sub "mongodb_already_started()";
        my $ret = undef;
        if(program_installed("squeue")) {
                my $squeue_output = qx(squeue -u \$USER);
                my $ipfiles_dir = $options{projectdir}.'/ipfiles/';
                if(-d $ipfiles_dir) {
                        my %possible_jobs = ();
                        while (my $file = <$ipfiles_dir/*>) {
                                if($file =~ m#mongodbserverip-(\d+)$#) {
                                        my $jobid = $1;
                                        $possible_jobs{$jobid} = 1;
                                }
                        }

                        foreach my $slurm_id (keys %possible_jobs) {
                                if($squeue_output =~ m#^\s*$slurm_id\s+#) {
                                        message "The running slurm job `$slurm_id` was detected as compatible. Using it instead of starting a new DB if server is up under that address.";
                                        $ret = $slurm_id;
                                }
                        }

                        if(defined $ret) {
                                my $unchecked_master_ip = read_file_chomp("$ipfiles_dir/mongodbserverip-$ret");
                                my $unchecked_master_port = read_file_chomp("$ipfiles_dir/mongodbportfile-$ret");

                                if(is_ipv4($unchecked_master_ip) && $unchecked_master_port =~ m#^\d+$#) {
                                        if(server_port_is_open($unchecked_master_ip, $unchecked_master_port)) {
                                                warning "No server running on $unchecked_master_ip:$unchecked_master_port";
                                        } else {
                                                ok "Using $unchecked_master_ip:$unchecked_master_port";
                                                ($master_ip, $master_port) = ($unchecked_master_ip, $unchecked_master_port);
                                                $options{slurmid} = $ret;
                                        }
                                }
                        }
                }
        } else {
                warning "squeue is not installed";
        }
        $indentation--;
        return $ret;
}

sub read_file_chomp {
        $indentation++;
        my $filename = shift;
        debug "read_file_chomp($filename)";
        my $contents = read_file($filename);
        chomp $contents;
        $indentation--;
        return $contents;
}

sub read_file {
        $indentation++;
        my $filename = shift;
        debug "read_file($filename)";
        my $contents = '';

        open my $fh, '<', $filename or warn $!;
        while (<$fh>) {
                $contents .= $_;
        }

        $indentation--;
        return $contents;
}

sub keep_db_up {
        $indentation++;
        debug_sub "keep_db_up()";
        if(!$options{keep_db_up}) {
                debug "Not checking of DB is up because of --keep_db_up=0";
                $indentation--;
                return 1;
        }

        if($options{dryrun}) {
                dryrun "No need to check for DB because this is only a dry-run.";
                $indentation--;
                return 0;
        }

        debug "Forking for keep_db_up()";
        my $pid = fork();
        error "ERROR Forking for keep_db_up: $!" if not defined $pid;
        if (not $pid) {
                debug "Inside fork";

                while (!$received_exit_signal) {
                        if (server_port_is_open($master_ip, $master_port)) {
                                message "DB does not seem to be up at $master_ip:$master_port. Trying to restart it...";
                                sleep 10;
                                start_mongo_db_fork();
                        } else {
                                ok_debug "The server seems to be up at $master_ip:$master_port";
                        }
                        debug "Sleeping for $options{sleep_db_up} (set value via `... sbatch.pl --sleep_db_up=n ...`) seconds before ".
                        "executing checking again if the DB is up";
                        sleep $options{sleep_db_up};
                }
                exit(0);
        }
        $indentation--;
}

sub run_nvidia_smi_periodically {
        $indentation++;
        debug_sub "run_nvidia_smi_periodically()";
        if(!$options{run_nvidia_smi}) {
                debug "Not running nvidia-smi because of --run_nvidia_smi=0";
                $indentation--;
                return 1;
        }

        if($options{dryrun}) {
                debug "Not running nvidia-smi because of --dryrun";
                $indentation--;
                return 1;
        }

        if(exists $ENV{'SLURM_JOB_NODELIST'}) {
                debug "Environment variable `SLURM_JOB_NODELIST` exists ($ENV{SLURM_JOB_NODELIST}), therefore, I am running nvidia-smi periodically";

                debug "Forking for run_nvidia_smi_periodically()";
                my $pid = fork();
                error "ERROR Forking for nvidia-smi: $!" if not defined $pid;
                if (not $pid) {
                        debug "Inside fork";
                        my $slurm_nodes = $ENV{'SLURM_JOB_NODELIST'};
                        my @server = get_servers_from_SLURM_NODES($slurm_nodes);

                        while (!$received_exit_signal) {
                                foreach my $this_server (@server) {
                                        my $nvidialogpath = "$options{logpathdate}/nvidia-$this_server/";
                                        if(!-d $nvidialogpath) {
                                                debug "$nvidialogpath does not exist yet, creating it";
                                                debug_system("mkdir -p $nvidialogpath") unless -d $nvidialogpath;
                                        }

                                        my $nvidialogfile = "$nvidialogpath/gpu_usage.csv";
                                        if(!-e $nvidialogfile) {
                                                debug_system("touch $nvidialogfile");
                                        }

                                        my $ipfiles_dir = $options{projectdir}.'/'.$options{project}.'/ipfiles/';
                                        my $command = qq#bash $script_paths{loggpu} "$nvidialogfile" "$ipfiles_dir" #.$ENV{SLURM_JOB_ID};
                                        my $sshcommand = "ssh -o LogLevel=ERROR $this_server '$command'";
                                        my $return_code = debug_system($sshcommand);
                                        if($return_code) {
                                                warning "$sshcommand seems to have failed! Exit-Code: $return_code";
                                        }
                                }

                                debug "Sleeping for $options{sleep_nvidia_smi} (set value via `... sbatch.pl --sleep_nvidia_smi=n ...`) seconds before ".
                                "executing nvidia-smi on each server again";
                                sleep $options{sleep_nvidia_smi};
                        }
                        exit(0);
                }
        } else {
                message "\$ENV{SLURM_JOB_NODELIST} not defined, are you sure you are in a Slurm-Job? Not running nvidia-smi."
        }
        $indentation--;
}

sub cancel_message {
        if(defined $options{originalslurmid}) {
                message "If you want to cancel this job, please use\n".
                "\t\t\tscancel --signal=USR1 --batch $options{originalslurmid}\n".
                "\t\tThis way, the database can be shut down correctly.";
        }
}


sub set_python_path {
        $indentation++;
        debug_sub 'set_python_path()';

        my $add_python_path = dirname(__FILE__);

        modify_system('PYTHONPATH='.$add_python_path.'/script/:$PYTHONPATH');

        $indentation--;
}

sub load_needed_modules {
        $indentation++;
        debug_sub 'load_needed_modules()';

        #modify_system("ml --force purge");
        my $lmod_path = $ENV{LMOD_CMD};
        modify_system("eval \$($lmod_path sh --force purge)");

        my $arch = (POSIX::uname)[4];

        my @modules = ();

        if($arch =~ m#ppc64le#) {
                push @modules, (
                        'modenv/ml',
                        'MongoDB/4.0.3',
                        'Python/3.7.4-GCCcore-8.3.0',
                        'Hyperopt/0.2.2-fosscuda-2019b-Python-3.7.4',
                );
        } else {
                push @modules, (
                        'modenv/scs5',
                        'MongoDB/4.0.3',
                        'Hyperopt/0.2.2-fosscuda-2019b-Python-3.7.4',
                        'Python/3.7.4-GCCcore-8.3.0',
                );
        }

        modules_load(@modules);
        $indentation--;
}

sub start_fmin_fork {
        $indentation++;
        debug_sub 'start_fmin_fork()';
        fork_and_dont_wait(start_fmin());
        wait_for_fmin();
        $indentation--;
}

sub create_and_delete_random_files_in_subfolders {
=head
        This weird construct creates and then immidiately deletes a random file in each
        subfolder. This is sometimes neccessary to update the file-system-cache on nodes,
        so that all files appear properly. (Yeah, kind of hacky, but it works...)
=cut
        $indentation++;
        debug "create_and_delete_random_files_in_subfolders()";
        opendir(my $dh, $this_cwd);
        my @files = readdir($dh);
        closedir($dh);

        foreach my $folder (sort { $a cmp $b } @files) {
                next if $folder =~ /^\.\.$/;
                next if !-d $folder;

                create_and_delete_random_file($folder);
        }
        $indentation--;
}

sub create_and_delete_random_file {
        $indentation++;
        my $folder = shift;
        debug "create_and_delete_random_file($folder)";

        my $file = "$folder/".rand();
        while (-e $file) {
                $file = "$folder/".rand();
        }

        open my $fh, '>', $file;

        if($?) {
                warning "ERROR: $? -> $!";
        } else {
                print $fh "";
                unlink($file);
        }

        close $fh;
        $indentation--;
}


sub start_worker_fork {
        $indentation++;
        debug_sub 'start_worker_fork()';
        my $start_worker_command = create_worker_start_command();
        debug $start_worker_command;
        my $ret_code = run_srun($start_worker_command);
        $indentation--;
        return $ret_code;
}

sub get_dmesg_start {
        $indentation++;
        debug_sub 'get_dmesg_start()';
        if($options{debug}) {
                get_dmesg('start');
        }
        $indentation--;
}

sub write_master_ip_and_port {
        $indentation++;
        debug_sub 'write_master_ip_and_port()';
        write_master_ip();
        get_open_ports();
        $indentation--;
}

sub wait_for_unfinished_jobs {
        $indentation++;
        debug_sub "wait_for_unfinished_jobs()";
        debug 'Waiting for started subjobs to exit';
        wait;
        ok_debug 'Done waiting for started subjobs';
        $indentation--;
}

sub wait_for_fmin {
        $indentation++;
        if(!$options{dryrun}) {
                debug "Waiting for $options{sleepafterfmin} seconds after starting fmin";
                sleep $options{sleepafterfmin};
        } else {
                debug "Would sleep for $options{sleepafterfmin} seconds, but not doing it because of --dryrun";
        }
        $indentation--;
}

sub create_worker_start_command {
        $indentation++;
        debug_sub 'create_worker_start_command()';
        my $slurmparameter = '';
        my $slurm_or_none = 'none';
        if($options{slurmid}) {
                $slurmparameter = " --slurmid=$options{slurmid} ";
                $slurm_or_none = $options{slurmid};
        }

        my $command = "$python_module$python $script_paths{worker} --max=$options{worker} --project=$options{project} --projectdir=$options{projectdir} $slurmparameter 2>&1 | tee -a $options{logpathdate}/log-start-worker.log";

        if(!$options{dryrun}) {
                my $exit_code_first_try = debug_system($command);
                if($exit_code_first_try != 0) {
                        warning "$command failed with $exit_code_first_try. Trying again.";
                        my $exit_code_second_try = debug_system($command);
                        if($exit_code_second_try != 0) {
                            warning "$command failed with $exit_code_first_try. Exiting now.";
                            exit 2;
                        }
                }
        } else {
                dryrun "Would run $command now, but not doing it because of --dryrun";
        }

        my $path = "$options{projectdir}/$options{project}/ipfiles/startworker-$slurm_or_none";
        my $worker_start_command = "bash $path | tee -a $options{logpathdate}/log-worker.log";

        if(!-e $path) {
                if($options{dryrun}) {
                        warning "$path could not be found, but I'm ignoring this because I'm in a dry-run";
                } else {
                        error "$path could not be found!";
                        exit(1);
                }
        }
        $worker_start_command =~ s#/{2,}#/#g;

        $indentation--;
        return $worker_start_command;
}

sub db_is_up {
        $indentation++;
        debug_sub "db_is_up()";
        if(!$master_ip || !$master_port) {
                $indentation--;
                return 0;
        } else {
                if (server_port_is_open($master_ip, $master_port)) {
                        $indentation--;
                        return 0;
                } else {
                        $indentation--;
                        return 1;
                }
        }
}

sub run_srun {
        $indentation++;
        my $run = shift;
        debug_sub "run_srun($run)";

        my $gpu_options = '';
        my $bash_vars = '';
        if(defined $options{number_of_allocated_gpus} && $options{number_of_allocated_gpus}) {
                $gpu_options .= ' --gres=gpu:1 ';
                $gpu_options .= ' --gres-flags=enforce-binding ';
                $gpu_options .= ' --gpus-per-task=1 ';
                $gpu_options =~ s#\s+# #g;
                debug "Set \$gpu_options to `$gpu_options`";
        }

        if($options{worker} <= 0) {
                warning "Don't start any workers because the worker is smaller than or equal to 0 (worker setting is $options{worker})";
                $indentation--;
                return 0;
        }

        if($options{debug_srun}) {
                $gpu_options .= ' -vvvvvvvvvvvvvvvvvvvvvvvv ';
                $gpu_options =~ s#\s+# #g;
        }

#### New Code

        #my $command = qq#srun -n1 --export=ALL,CUDA_VISIBLE_DEVICES,CUDA_DEVICE_ORDER=PCI_BUS_ID --exclusive --ntasks-per-core=1 $gpu_options --mpi=none --mem-per-cpu=$options{mempercpu} --no-kill bash -c "$bash_vars$run"#;
        my $command = qq#srun -n1 --export=ALL,CUDA_VISIBLE_DEVICES --exclusive --ntasks-per-core=1 $gpu_options --mpi=none --mem-per-cpu=$options{mempercpu} --no-kill bash -c "$bash_vars$run"#;
        $command =~ s#\s{2,}# #g;
        $command =~ s#/{2,}#/#g;

        debug $command;

        if(!$options{dryrun}) {
                foreach my $worker_nr (1 .. $options{worker}) {
                        debug "Waiting 5 seconds before starting new srun";
                        sleep 5;
                        debug "Starting $worker_nr of $options{worker}";
                        fork_and_dont_wait($command);
                }
        } else {
                dryrun "Not running\n>>> $command\nbecause of --dryrun";
        }
        $indentation--;
        return 1;
}

sub fork_and_dont_wait {
        $indentation++;
        my $command = shift;
        debug_sub "fork_and_dont_wait($command)";

        if(!$options{dryrun}) {
                my $pid = fork();
                my $errno = $!;
                error 'SOMETHING WENT HORRIBLY WRONG WHILE FORKING!' if not defined $pid;
                if (defined $pid && $pid == -1) {
                        error '!!! fork() FAILED AND RETURNED -1 AS PID IN fork_and_dont_wait. ERRNO: '.$errno.' !!!' if not defined $pid;
                } else {
                        if(!defined $pid) {
                                error 'SOMETHING WENT HORRIBLY WRONG WHILE FORKING!';
                        } elsif (not $pid) {
                                debug "In child process $$";
                                exec($command) or error "Something went wrong executing '$command': $!";;
                        } else {
                                debug 'In parent process (fork_and_dont_wait)';
                        }
                }
        } else {
                dryrun "Not really forking_and_not_waiting for\n$command\nbecause of --dryrun";
        }
        $indentation--;
}

sub fork_sub_call {
        $indentation++;
        my ($subref, @args) = @_;

        if(ref $subref ne 'CODE') {
                error 'Wrong var-type. Should be subref, nothing else.';
        } else {
                my $pid = fork();
                my $errno = $!;
                if (defined $pid && $pid == -1) {
                        error '!!! fork() FAILED AND RETURNED -1 AS PID IN fork_sub_call(). ERRNO: '.$errno.' !!!' if not defined $pid;
                } else {
                        if (!defined $pid) {
                                error 'SOMETHING WENT WRONG WHILE FORKING!';
                        } elsif ($pid != 0) {
                                $indentation--;
                                return &{$subref}->(@args);
                        } else {
                                debug "In parent process (fork_sub_call)";
                        }
                }
        }
        wait();
        $indentation--;
}

sub start_fmin {
        $indentation++;
        debug_sub 'start_fmin()';
        my $slurmparameter = '';
        if($options{slurmid}) {
                $slurmparameter = " --slurmid=$options{slurmid} ";
        }

        my $command = qq#$python_module$python $script_paths{fmin} --max=$options{worker} --project=$options{project} --projectdir=$options{projectdir} $slurmparameter 2>&1 | tee $options{logpathdate}/log-fmin.log#;

        debug $command;
        $indentation--;
        return $command;
}

sub remove_mongodb_lock_file {
        $indentation++;
        debug_sub "remove_mongodb_lock_file()";

        my $mongodb_dir = $options{projectdir}.'/'.$options{project}.'/mongodb';
        if(-d $mongodb_dir) {
                my $lock_file = "$mongodb_dir/mongod.lock";
                if(-e $lock_file) {
                        debug "Unlinking $lock_file";
                        if($options{dryrun}) {
                                dryrun "Not really removing lockfile";
                        } else {
                                unlink $lock_file;
                        }
                } else {
                        ok "$lock_file does not exist";
                }
        } else {
                warning "The dir $mongodb_dir does not exist";
        }

        $indentation--;
}

sub start_mongo_db_fork {
        $indentation++;
        debug_sub 'start_mongo_db_fork()';

        remove_mongodb_lock_file();

        write_master_ip_and_port();
        my $slurmparameter = '';
        if($options{slurmid}) {
                $slurmparameter = " --slurmid=$options{slurmid} ";
        }
        my $command = qq#$python_module$python $script_paths{mongodb} --max=$options{worker} --project=$options{project} --projectdir=$options{projectdir} $slurmparameter 2>&1 | tee $options{logpathdate}/log-mongodb.log#;
        debug $command;
        $indentation-- if $options{dryrun};
        return dryrun 'Not starting MongoDB because of --dryrun' if $options{dryrun};
        if(my $exit_code = debug_system($command)) {
                # TODO: mongod --repair --dbpath mongodb --storageEngine wiredTiger
                warning "ERROR: MongoDB could not start. Exit-Code: $exit_code.";

                my $mongodb_dir = "$options{projectpath}/mongodb";

                my $i = 0;
                my $new_dir = $mongodb_dir."_".$i;

                while (-d $new_dir) {
                        $new_dir = $mongodb_dir."_".$i;
                        $i++;
                }

                warning "Trying to move the directory $mongodb_dir to another folder ($new_dir)";

                move($mongodb_dir, $new_dir) or warning "WARNING: Cannot move $mongodb_dir to $new_dir, error: $!";

                debug_system("mkdir -p $mongodb_dir");

                debug "Trying again to start MongoDB (last try)";

                if(my $exit_code = debug_system($command)) {
                        error "ERROR: MongoDB could not start. Exit-Code: $exit_code.";
                }
        } else {
                $ran_anything = 1;
                backup_mongo_db();
        }

        if($master_ip) {
                if($master_port) {
                        if(server_port_is_open($master_ip, $master_port)) {
                                warning "The port $master_port on $master_ip is open. This probably means that MongoDB did not start correctly!";
                        }
                } else {
                        warning "Master port not defined!";
                }
        } else {
                warning "Master IP not defined!";
        }
        $indentation--;
}

sub install_needed_packages {
        $indentation++;
        debug_sub 'install_needed_packages()';
        if($options{dryrun}) {
                debug "Not installing needed packages because of --dryrun";
                $indentation--;
                return 1;
        }

        check_networkx_version();
        my @packages = (
                'dill',
                'pymongo',
                'psutil',
                'future'
        );

        if(is_ml()) {
                warning "No pip3 on ML! Not installing needed packages. Make sure you got them installed by yourself somehow!";
        } else {
                foreach my $package (@packages) {
                        debug "Trying to install $package if needed";
                        install_via_pip3($package);
                }
        }
        $indentation--;
}

sub check_networkx_version {
        $indentation++;
        debug_sub 'check_networkx_version()';

        if($options{dryrun}) {
                $indentation--;
                dryrun 'Not running check_networkx_version() because of --dryrun';
                return 1;
        }

        my $get_version_command = q#pip3 show networkx 2>&1 | grep "Version:" | sed -e 's/Version: //'#;

        my $version = debug_qx($get_version_command);
        if($version eq '1.11') {
                debug_sub 'networkxversion ok (1.11)';
        } else {
                warning 'The version of networkx is not the one supported by HyperOpt. Using 1.11.';
                install_via_pip3('networkx==1.11');
        }
        $indentation--;
}

sub install_via_pip3 {
        $indentation++;
        my $package = shift;
        debug_sub "install_via_pip3($package)";

        $indentation-- if $options{dryrun};
        if(!$options{install_despite_dryrun}) {
                return dryrun "Not installing `$package` via pip3, because of --dryrun" if $options{dryrun};
        }

        if(program_installed('pip3')) {
                my $command = "pip3 install --user $package 2>/dev/null";
                debug_system($command);
        } else {
                warning 'pip3 is not in the $PATH';
        }
        $indentation--;
}

sub create_paths {
        $indentation++;
        debug_sub 'create_paths()';
        my @folders = (
                "debuglogs",
                "$options{projectpath}",
                "$options{projectpath}/logs",
                "$options{projectpath}/slurmlogs",
                "$options{projectpath}/mongodb",
                "$options{projectpath}/ipfiles",
                "$options{logpathdate}"
        );

        foreach my $folder (@folders) {
                debug "create_paths() -> $folder";
                if(-d $folder) {
                        ok_debug "$folder already exists. Doing nothing.";
                } else {
                        if(!$options{dryrun}) {
                                make_path($folder) or error $!;
                        } else {
                                dryrun "Not really creating $folder because of --dryrun";
                        }
                }
        }
        $indentation--;
}

sub get_dmesg {
        $indentation++;
        my $pos = shift;
        debug_sub "get_dmesg($pos)";
        if($options{debug}) {
                if($options{originalslurmid}) {
                        if($options{project}) {
                                get_dmesg_general($pos);
                        } else {
                                error 'No project set';
                        }
                } else {
                        warning 'No Slurm-ID, not getting dmesg';
                }
        }
        $indentation--;
}

sub sanity_check {
        $indentation++;
        debug_sub 'sanity_check()';

        if(!$options{sanitycheck}) {
                message 'Disabled sanity check';
        } else {
                if(!$options{originalslurmid}) {
                        warning 'No Slurm-ID!';
                }

                foreach my $needed_script (sort { $a cmp $b } keys %script_paths) {
                        debug "Checking for $needed_script in $script_paths{$needed_script}...";
                        if(-e $script_paths{$needed_script}) {
                                ok_debug "$script_paths{$needed_script} found!";
                        } else {
                                error "$script_paths{$needed_script} NOT found!";
                        }
                }

                debug 'Check for project';
                if(!$options{project}) {
                        error 'No project set!';
                } else {
                        ok_debug "Project set to $options{project}";
                }

                debug 'Check for project path';
                if(!-d $options{projectpath}) {
                        error "$options{projectpath} not found!";
                } else {
                        ok_debug "$options{projectpath} exists";
                }

                debug 'Check for project config.ini';
                if(!-e "$options{projectpath}/config.ini") {
                        error "$options{projectpath}/config.ini not found!";
                } else {
                        ok_debug "$options{projectpath}/config.ini exists";
                }

                debug 'Check if worker number is defined';
                if(!defined $options{worker}) {
                        error 'Worker not defined!';
                } else {
                        ok_debug "Worker defined: $options{worker}";
                }

                debug 'Check if at least two workers were selected';
                if($options{worker} < 2) {
                        warning 'Less than 2 workers';
                } else {
                        ok_debug "Number of workers is greater than 2";
                }

                debug 'Check if at least 1GB of RAM was selected';
                if($options{mempercpu} < MIN_RECOMMENDED_MEM_PER_CPU) {
                        warning 'Less than 1GB of RAM per CPU. This might lead into trouble.';
                } else {
                        ok_debug "There is more than ".MIN_RECOMMENDED_MEM_PER_CPU()."MB Ram";
                }

                debug "Warn if at more than ".MAX_WORKER_WITHOUT_WARNING()." workers";
                if($options{worker} >= MAX_WORKER_WITHOUT_WARNING) {
                        warning "There might be problems with more than ".MAX_WORKER_WITHOUT_WARNING()." workers, e.g. I've experienced missing STDOUT from subprocesses and therefore corrupt optimizations.";
                } else {
                        ok_debug "There are less than ".MAX_WORKER_WITHOUT_WARNING()." workers allocated";
                }
        }
        $indentation--;
}

sub _help {
        debug_sub '_help()';

        my $cli_code = color('yellow bold');
        my $reset = color('reset');

        print <<"EOF";
=== HELP ===

This is the main script for OmniOpt, a mostly-automated hyperparameter optimizer. This script is
thought to be ran with sbatch, like this:

You will need to define the following shell-variables:

    $cli_code export \$PROJECTNAME=name_of_project_folder$reset
    $cli_code export \$MEMPERCPU=mem_in_megabytes$reset
    $cli_code export \$NUMBEROFWORKERS=number_of_workers_plus_3$reset

    $cli_code sbatch -J '\$PROJECTNAME' --mem-per-cpu='\$MEMPERCPU' --ntasks='\$NUMBEROFWORKERS' --time="00:10:00" sbatch.pl$reset

This will automatically start (ntasks - 3) workers that will consume \$MEM_PER_CPU mb per worker for
10 Minutes working on the project \$PROJECTNAME. There are no further options needed!

When you run

    $cli_code sbatch -J '\$PROJECTNAME' --mem-per-cpu='\$MEMPERCPU' --ntasks='\$NUMBEROFWORKERS' --mincpus=\$NUMBEROFWORKERS --gres=gpu:\$NUMBEROFWORKERS --time="00:10:00" sbatch.pl$reset

Will do the same, but every worker will have one GPU available to use for scripts.

Any parameters of sbatch.pl are optional. It will try to get the data from the sbatch-environment.

But if you have another jobname than the projectname, for example, you can use

    $cli_code ... sbatch.pl --project=\$PROJECTNAME$reset

This will override any Slurm-environment-variables.

Parameters:
    --help                                  This help
    --worker=n                              Number of workers (usually automatically deducted from --ntasks)
    --mempercpu=1234                        Defines how much memory every worker should get. Usually this is not needed,
                                            because this script get's it from the sbatch command's --mem-per-cpu
    --nosanitycheck                         Disables the sanity checks (not recommended!)
    --project=projectname                   Name of the project you want to execute
    --keep_db_up=[01]                       Check if the DB is up every \$sleep_db_up seconds and restart it if it's not
    --sleep_db_up=n                         Check if DB is up every n seconds and restart if it's not (default: $default_values{sleep_db_up})
    --run_nvidia_smi=[01]                   Run (1) or don't run (0) nvidia-smi periodically to get information about the GPU usage of the workers
    --sleep_nvidia_smi=n                    Sleep n seconds after each try to get nvidia-smi-gpu-infos (default: $default_values{sleep_nvidia_smi} seconds)
    --debug_srun                            Enables debug-output for srun
    --projectdir=/path/to/projects/         This allows you to change the project directory to anything outside of this script path (helpful,
                                            so that the git is kept small), only use absolute paths here!

Debug and output parameters:
    --nomsgs                                Disables messages
    --debug                                 Enables lots and lots of debug outputs
    --dryrun                                Run this script without any side effects (i.e. not creating files, not starting workers etc.)
    --nowarnings                            Disables the outputting of warnings (not recommended!)
    --nodryrunmsgs                          Disables the outputting of --dryrun-messages
    --run_tests                             Runs a bunch of tests and exits without doing anything else
    --run_full_tests                        Run testsuite and also run a testjob (takes longer, but is more safety to ensure stability)

System parameters:
    --only_install_modules                  Only install the neccessary modules
EOF
}

sub get_project_folder {
        $indentation++;
        my $projectname = shift;

        debug_sub "get_project_folder($projectname)";
        $indentation--;
        my $projectdir = $options{projectdir};
        foreach (@ARGV) {
                ### HACKY SOLUTION
                if(m#--projectdir=(.*)#) {
                        $projectdir = $1;
                        if($projectdir !~ m#^/#) {
                                $projectdir = get_working_directory()."/$projectdir";
                        }
                        $options{projectdir} = $projectdir;
                }
        }
        my $str = "$projectdir/$projectname/";
        debug "Chosen projectdir -> $str";
        $options{projectpath} = $str;

        $str =~ s#/{2,}#/#g;
        return $str;
}

sub get_log_path_date_folder {
        $indentation++;
        my $project = shift;
        debug_sub "get_log_path_date_folder($project)";
        my $date = strftime '%Y-%m-%d_%H-%M-%S', localtime;
        $indentation--;
        return get_project_folder($project)."/logs/$date"
}

sub analyze_args {
        my @args = @_;

        my $run_tests = 0;
        my $run_full_tests = 0;

        foreach my $arg (@args) {
                if($arg =~ m#^--project=(.*)$#) {
                        if(defined $options{project}) {
                                message 'The --project overrides the --jobname of sbatch';
                        } else {
                                $options{project} = $1;
                                $options{logpathdate} = get_log_path_date_folder($1);
                        }
                } elsif ($arg =~ m#^--worker=(\d+)$#) {
                        $options{worker} = $1;
                } elsif ($arg eq '--dryrun') {
                        $options{dryrun} = 1;
                } elsif ($arg eq '--nomsgs') {
                        $options{messages} = 0;
                } elsif ($arg eq '--nowarnings') {
                        $options{warnings} = 0;
                } elsif ($arg eq '--nosanitycheck') {
                        $options{sanitycheck} = 0;
                } elsif ($arg eq '--nodryrunmsgs') {
                        $options{dryrunmsgs} = 0;
                } elsif ($arg eq '--debug') {
                        $options{debug} = 1;
                } elsif ($arg =~ m#^--run_nvidia_smi=([01])$#) {
                        $options{run_nvidia_smi} = $1;
                } elsif ($arg =~ m#^--sleep_nvidia_smi=(\d+)$#) {
                        $options{sleep_nvidia_smi} = $1;
                } elsif ($arg =~ m#^--debug_srun$#) {
                        $options{debug_srun} = 1;
                } elsif ($arg =~ m#^--keep_db_up=(\d+)$#) {
                        $options{keep_db_up} = $1;
                } elsif ($arg =~ m#^--sleep_db_up=(\d+)$#) {
                        $options{sleep_db_up} = $1;
                } elsif ($arg =~ m#^--mempercpu=(\d+)$#) {
                        $options{mempercpu} = $1;
                } elsif ($arg =~ m#^--install_despite_dryrun$#) {
                        $options{install_despite_dryrun} = 1;
                } elsif ($arg =~ m#^--projectdir=(.+)$#) {
                        my $projectdir = $1;
                        if($projectdir !~ m#^/#) {
                                $projectdir = get_working_directory()."/$projectdir";
                        }
                        $options{projectdir} = $projectdir;
                } elsif ($arg =~ m#^--only_install_modules$#) {
                        install_needed_packages();
                        exit(0);
                } elsif ($arg =~ m#^--run_tests$#) {
                        $run_tests = 1;
                } elsif ($arg =~ m#^--run_full_tests$#) {
                        $run_tests = 1;
                        $run_full_tests = 1;
                } elsif ($arg eq '--help') {
                        _help();
                        exit(0);
                } else {
                        std_print color("red underline")."Unknown command line switch: $arg".color("reset")."\n";
                        _help();
                        exit(1);
                }
        }

        load_needed_modules();

        if($run_tests) {
                exit(run_tests($run_full_tests));
        }

        error "No project defined. Use sbatch -J \$PROJECTNAME or --project=\$PROJECTNAME to define a project!" unless $options{project};
}

sub debug_qx ($) {
        $indentation++;
        my $command = shift;
        debug_sub "debug_qx($command)";

        $indentation--;
        return qx($command);
}

sub debug_system ($) {
        $indentation++;
        my $command = shift;
        debug_sub "debug_system($command)";

        system($command);
        my $error_code = $?;
        my $exit_code = $error_code >> 8;
        my $signal_code = $error_code & 127;
        debug "EXIT-Code: $exit_code, Signal: $signal_code";
        $indentation--;
        return $exit_code;
}

sub is_ml {
        $indentation++;
        debug_sub "is_ml()";
        if(hostname() =~ m#taurusml#) {
                debug "is_ml() -> 1";
                $indentation--;
                return 1;
        } else {
                debug "is_ml() -> 0";
                $indentation--;
                return 0;
        }
}

sub backup_mongo_db {
        $indentation++;
        debug_sub 'backup_mongo_db()';

        if($options{dryrun}) {
                dryrun 'Not backing up MongoDB because of --dryrun';
                $indentation--;
                return 1;
        }

        if(is_ml()) {
                $indentation--;
                return;
        }

        if($options{project}) {
                if(!$options{slurmid}) {
                        warning 'No Slurm-ID!';
                }
        } else {
                warning 'No project!';
        }
        $indentation--;
}

sub end_mongo_db {
        $indentation++;
        debug_sub 'end_mongo_db()';

        $indentation-- if $options{dryrun};
        return dryrun 'Not ending MongoDB because of --dryrun' if $options{dryrun};

        if($options{project}) {
                if($options{slurmid}) {
                        my $command = "$python_module$python $script_paths{endmongodb} --project=$options{project} --slurmid=$options{slurmid} --projectdir=$options{projectdir}";
                        debug_system $command;
                } else {
                        warning 'No Slurm-ID!';
                }
        } else {
                warning 'No project!';
        }
        $indentation--;
}

sub _get_project {
        $indentation++;
        debug_sub '_get_project()';
        my $tmp_project = _get_environment_variable('SLURM_JOB_NAME');
        if($tmp_project) {
                if($tmp_project && -e get_project_folder($tmp_project)) {
                        $options{project} = $tmp_project;
                        $options{projectpath} = get_project_folder($tmp_project);
                        $options{logpathdate} = get_log_path_date_folder($tmp_project);
                } else {
                        warning 'Invalid project name or the project folder does not exist';
                }
        } else {
                warning 'No project name defined in Slurm-Job';
        }
        $indentation--;
}

sub _get_number_of_workers {
        $indentation++;
        debug_sub '_get_number_of_workers()';
        my $ntasks = _get_environment_variable('SLURM_NTASKS');
        if($ntasks) {
                if($ntasks && $ntasks =~ m#^\d+$# && $ntasks >= 4) {
                        $indentation--;
                        return $ntasks;
                } else {
                        warning 'Invalid NTASKS ('.$ntasks.'). Should be number and at least 4!';
                }
        } else {
                warning 'ntasks not defined in Slurm-Job';
        }

        $indentation--;
        return $default_values{worker};
}

sub _get_mem_per_cpu {
        $indentation++;
        debug_sub '_get_mem_per_cpu()';
        my $mempercpu = _get_environment_variable('SLURM_MEM_PER_CPU');
        if($mempercpu) {
                if($mempercpu && $mempercpu =~ m#^\d+$# && $mempercpu >= 1) {
                        return $mempercpu;
                } else {
                        warning 'Invalid MEM_PER_CPU. Should be number and at least 1!';
                }
        } else {
                warning 'mempercpu not defined in Slurm-Job';
        }

        $indentation--;
        return $default_values{mempercpu};
}

sub _get_gpu_info {
        $indentation++;
        debug_sub '_get_gpu_info()';
        my $gpu_device_ordinal = _get_environment_variable('GPU_DEVICE_ORDINAL');
        if(defined $gpu_device_ordinal && $gpu_device_ordinal !~ /dev/i) {
                my @gpus = split(/,/, $gpu_device_ordinal);
                $indentation--;
                return scalar @gpus;
        } else {
                message 'No specific number of GPUs allocated, using the default number: '.$default_values{number_of_allocated_gpus};
                $indentation--;
                return $default_values{number_of_allocated_gpus};
        }
}

sub get_environment_variables {
        $indentation++;
        debug_sub 'get_environment_variables()';

        _get_project();
        $options{worker} = _get_number_of_workers();
        $options{mempercpu} = _get_mem_per_cpu();
        $options{number_of_allocated_gpus} = _get_gpu_info();

        $indentation--;
        return 1;
}

sub _get_environment_variable {
        $indentation++;
        my $name = shift;
        debug_sub "_get_environment_variable($name)";
        $indentation--;
        return $ENV{$name};
}

sub write_master_ip {
        $indentation++;
        debug_sub 'write_master_ip()';

        my $ip = get_local_ip_address();

        my $dirname = $options{projectdir}.'/'.$options{project};
        error "$dirname could not be found!" unless -d $dirname;

        $dirname = "$dirname/ipfiles/";
        mkdir $dirname unless -d $dirname;

        my $slurmidornone = defined $options{slurmid} ? $options{slurmid} : 'NONE';

        my $file = "$dirname/mongodbserverip-$slurmidornone";
        debug "Writing IP `$ip` to $file";

        if (!-e $file) {
                if(is_ipv4($ip)) {
                        if(!$options{dryrun}) {
                                open my $fh, '>', $file or croak $!;
                                print $fh $ip;
                                close $fh;
                        } else {
                                dryrun "Not writing $ip to $file because of dryrun";
                        }
                } else {
                        warning "ERROR: $ip is not a valid IPv4-address!";
                }
        } else {
                debug "$file already exists!";
        }

        $master_ip = $ip;
        $indentation--;
        return $ip;
}

sub modules_load {
        $indentation++;
        my @modules = @_;
        debug_sub 'modules_load('.join(', ', @modules).')';
        foreach my $mod (@modules) {
                module_load($mod);
        }

        $indentation--;
        return 1;
}

sub modify_system {
        my $command = shift;
        $indentation++;
        debug_sub("modify_system($command)");
        $indentation--;
        my $return_code = Env::Modify::system($command);

        my $exit_code = $return_code >> 8;
        my $exit_signal = $return_code & 127;

        if($exit_code != 0) {
                warning "Exit-Code: $exit_code\n";
        }

        if($exit_signal != 0) {
                warning "Got signal: $exit_signal\n";
        }

        return $exit_code;
}

sub module_load {
        $indentation++;
        my $toload = shift;
        debug_sub "module_load($toload)";

        if($toload) {
                if($options{dryrun}) {
                        dryrun "Not loading module $toload because of --dryrun";
                } else {
                        if(exists($ENV{LMOD_CMD})) {
                                my $lmod_path = $ENV{LMOD_CMD};
                                my $command = "eval \$($lmod_path sh load $toload)";
                                debug $command;
                                local $Env::Modify::CMDOPT{startup} = 1;
                                modify_system($command);
                        } else {
                                warning "LMOD_CMD not found in ENV";
                        }
                }
        } else {
                warning 'Empty module_load!';
        }
        $indentation--;
        return 1;
}

sub debug_options {
        $indentation++;
        debug_sub 'debug_options()';
        debug 'This script file: '.get_working_directory()."/".__FILE__;
        debug split(/\R/, Dumper \%options);
        $indentation--;
        return 1;
}

sub program_installed {
        $indentation++;
        my $program = shift;
        debug_sub "program_installed($program)";
        my $exists = 0;
        my $ret = modify_system(qq#which $program > /dev/null 2> /dev/null#);

        if($ret == 0) {
                debug "$program already installed";
                $exists = 1;
        } else {
                warning "$program does not seem to be installed. Please install it!";
        }
        $indentation--;

        return $exists;
}

# https://stackoverflow.com/questions/330458/how-can-i-determine-the-local-machines-ip-addresses-from-perl
sub get_local_ip_address {
        $indentation++;
        debug_sub 'get_local_ip_address()';
        my $socket = IO::Socket::INET->new(
                Proto       => 'udp',
                PeerAddr    => '198.41.0.4', # a.root-servers.net
                PeerPort    => '53', # DNS
        );

        # A side-effect of making a socket connection is that our IP address
        # is available from the 'sockhost' method
        my $local_ip_address = $socket->sockhost;

        $indentation--;
        return $local_ip_address;
}

sub is_ipv4 {
        $indentation++;
        my $ip = shift;
        debug_sub "is_ipv4($ip)";
        $indentation--;
        if ($ip =~ /^((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)$/) {
                return 1;
        } else {
                return 0;
        }
}

sub server_port_is_open {
        $indentation++;
        my $server = shift;
        my $port = shift;

        debug_sub "server_port_is_open($server, $port)";

        local $| = 1;

        # versucht ein TCP-Socket auf dem Server mit dem Port zu öffnen; wenn das geht, ist der Port nicht offen (return 0)

        my $socket = IO::Socket::INET->new(
                PeerHost => $server,
                PeerPort => $port,
                Proto => 'tcp'
        );

        $indentation--;
        if($socket) {
                return 0;
        } else {
                return 1;
        }
}

sub get_servers_from_SLURM_NODES {
        $indentation++;
        my $string = shift;
        debug_sub "get_servers_from_SLURM_NODES($string)";

        my @server = map { chomp; $_ } qx#scontrol show hostname $string#;

        if(!@server) {
                while ($string =~ m#(.*?)\[(.*?)\](?:,|\R|$)#gi) {
                        my $servercluster = $1;
                        my $servernumbers = $2;
                        foreach my $thisservernumber (split(/,/, $servernumbers)) {
                                if($servernumbers !~ /-/) {
                                        push @server, "$servercluster$thisservernumber";
                                }
                        }

                        if($servernumbers =~ m#(\d+)-(\d+)#) {
                                push @server, map { "$servercluster$_" } $1 .. $2;
                        }
                }
        }

        $indentation--;
        if(@server) {
                return @server;
        } else {
                return ('127.0.0.1');
        }
}

sub get_random_number {
        $indentation++;
        my $minimum = shift // DEFAULT_MIN_RAND_GENERATOR;
        my $maximum = shift // DEFAULT_MAX_RAND_GENERATOR;
        debug_sub "get_random_number($minimum, $maximum)";
        my $x = $minimum + int(rand($maximum - $minimum));
        debug "random_number -> $x";
        $indentation--;
        return $x;
}

sub test_port_on_multiple_servers {
        $indentation++;
        my ($port, @servers) = @_;
        debug_sub "test_port_on_multiple_servers($port, (".join(', ', @servers).'))';
        my $is_open_everywhere = 1;
        THISFOREACH: foreach my $server (@servers) {
                if(!server_port_is_open($server, $port)) {
                        $is_open_everywhere = 0;
                        debug "Port $port was not open on server $server";
                        last THISFOREACH;
                }
        }
        if($is_open_everywhere) {
                ok_debug "Port $port is open everywhere!";
        } else {
                warning "Port $port is NOT open everywhere";
        }
        $indentation--;

        return $is_open_everywhere;
}

sub get_open_ports {
        $indentation++;
        debug_sub 'get_open_ports()';

        my $slurm_nodes = '127.0.0.1';
        if(exists $ENV{'SLURM_JOB_NODELIST'}) {
                $slurm_nodes = $ENV{'SLURM_JOB_NODELIST'};
        }

        debug "Slurm-Nodes: $slurm_nodes";

        my @server = get_servers_from_SLURM_NODES($slurm_nodes);
        my $port = get_random_number();
        while (!test_port_on_multiple_servers($port, @server)) {
                $port = get_random_number();
        }

        ok_debug "Port: $port";

        my $slurmidornone = defined $options{slurmid} ? $options{slurmid} : 'NONE';

        my $slurm_id = $ENV{'SLURM_JOB_ID'};
        my $file = $options{projectdir}."/$options{project}/ipfiles/mongodbportfile-$slurmidornone";
        debug "Writing open port $port to file $file";
        if(!$options{dryrun}) {
                open my $fh, '>', $file or croak $!;
                print $fh "$port\n";
                close $fh or croak $!;
        } else {
                dryrun "Not writing $port to $file because of dryrun";
        }

        $master_port = $port;
        $indentation--;
        return $port;
}

sub get_dmesg_general {
        $indentation++;
        my $type = shift // 'start';
        debug_sub "get_dmesg_general($type)";
        my $slurm_nodes = '127.0.0.1';
        if(exists $ENV{'SLURM_JOB_NODELIST'}) {
                debug "Environment variable `SLURM_JOB_NODELIST` exists ($ENV{SLURM_JOB_NODELIST})";
                $slurm_nodes = $ENV{'SLURM_JOB_NODELIST'};
        }

        debug "Slurm-Nodes: $slurm_nodes";

        my @server = get_servers_from_SLURM_NODES($slurm_nodes);
        my $slurmidornone = defined $options{slurmid} ? $options{slurmid} : 'NONE';

        foreach my $server (@server) {
                my $logpath = '/var/log/';
                my @logfiles = ('dmesg', 'nhc.log');
                foreach my $this_logfile (@logfiles) {
                        my $logfile = "$logpath$this_logfile";
                        my $log_folder = "$options{logpathdate}/$slurmidornone/$type/$this_logfile";
                        system "mkdir -p $log_folder";
                        my $log_file_name = "$server-$this_logfile.$type.log";
                        my $log_file_path = "$log_folder/$log_file_name";

                        my $command = "scp -o LogLevel=ERROR $server:/$logfile $log_file_path";
                        debug $command;
                        qx($command);
                }
        }

        $indentation--;
        return 1;
}

sub run_tests {
        my $run_full_tests = shift // 0;
        $indentation++;

        my @failed_tests = ();

        run_bash_test("bash test/test_packages.sh", \@failed_tests);

        {
                for my $i (0 .. 10) {
                        my $rand_nr = get_random_number();
                        if(!($rand_nr >= DEFAULT_MIN_RAND_GENERATOR && $rand_nr <= DEFAULT_MAX_RAND_GENERATOR)) {
                                no_suicide_error "get_random_number() test failed";
                                push @failed_tests, "get_random_number()";
                        } else {
                                ok "get_random_number() test ok";
                        }
                }
        }

        {
                for my $i (0 .. 10) {
                        my ($min, $max) = (10, 20);
                        my $rand_nr = get_random_number($min, $max);
                        if(!($rand_nr >= $min && $rand_nr <= $max)) {
                                no_suicide_error "get_random_number($min, $max) test failed (nr. $i)";
                                push @failed_tests, "get_random_number()";
                        } else {
                                ok "get_random_number($min, $max) test ok (nr. $i)";
                        }
                }
        }

        {
                my $test_ip_true = "120.255.1.102";
                if(!is_ipv4($test_ip_true)) {
                        no_suicide_error "is_ipv4($test_ip_true) test failed";
                        push @failed_tests, "is_ipv4($test_ip_true)";
                } else {
                        ok "is_ipv4($test_ip_true) test ok";
                }
        }

        {
                my $test_ip_false = "120.2255.1.102";
                if(is_ipv4($test_ip_false)) {
                        no_suicide_error "is_ipv4($test_ip_false) test failed";
                        push @failed_tests, "is_ipv4($test_ip_false)";
                } else {
                        ok "is_ipv4($test_ip_false) test ok";
                }
        }

        {
                my $test_ip_true = get_local_ip_address();
                if(!is_ipv4($test_ip_true)) {
                        no_suicide_error "is_ipv4($test_ip_true) test failed";
                        push @failed_tests, "is_ipv4($test_ip_true)";
                } else {
                        ok "is_ipv4($test_ip_true) test ok";
                }
        }

        {
                my $test_program_installed = "ls";
                if(!program_installed($test_program_installed) eq '/bin/ls') {
                        no_suicide_error "program_installed($test_program_installed) test failed";
                        push @failed_tests, "program_installed($test_program_installed)";
                } else {
                        ok "program_installed($test_program_installed) test ok";
                }
        }

        {
                my $original_warning = $options{warnings};
                $options{warnings} = 0;
                my $test_program_installed = "askjdasldsfljdgndgjkndfglkdf";
                if(!defined(program_installed($test_program_installed))) {
                        no_suicide_error "program_installed($test_program_installed) test failed";
                        push @failed_tests, "program_installed($test_program_installed)";
                } else {
                        ok "program_installed($test_program_installed) test ok";
                }
                $options{warnings} = $original_warning;
        }

        {
                my $slurm_nodes = 'taurusi2012';
                if(!get_servers_from_SLURM_NODES($slurm_nodes) eq 'taurusi2012') {
                        no_suicide_error "Error parsing $slurm_nodes";
                        push @failed_tests, "get_servers_from_SLURM_NODES($slurm_nodes)";
                } else {
                        ok "get_servers_from_SLURM_NODES($slurm_nodes) ok";
                }
        }

        {
                my $config_file = "test/projects/DONOTDELETE_testcase/config.ini";
                my %ini = read_ini_file($config_file);
                my $string = join(',', sort { $a cmp $b  } keys %ini);
                if($string eq 'DATA,DEBUG,DIMENSIONS,MONGODB') {
                        ok "read_ini_file ok";
                } else {
                        no_suicide_error "read_ini_file failed";
                        push @failed_tests, "read_ini_file($config_file)";
                }
        }

        if (!$ENV{SKIPPYTHONTESTS}){
                run_bash_test("$python_module$python $script_paths{testscript}", \@failed_tests);
        }

        {
                if(check_needed_programs() == 0) {
                        ok "Every needed program is installed";
                } else {
                        no_suicide_error "Not every needed program is installed";
                        push @failed_tests, "check_needed_programs()";
                }
        }

        {
                my $has_error = environment_sanity_check();
                if($has_error) {
                        push @failed_tests, "environment_sanity_check()";
                }
        }

        if ($run_full_tests) {
                run_bash_test("bash test/run_cpu_test.sh --nocountdown", \@failed_tests);
                run_bash_test("bash test/run_cpu_test_2.sh --nocountdown", \@failed_tests);
                run_bash_test("bash test/run_gpu_test.sh --nocountdown", \@failed_tests);
                run_bash_test("bash test/run_gpu_test_alpha.sh --nocountdown", \@failed_tests);
                run_bash_test("bash test/test_plot.sh", \@failed_tests);
                run_bash_test("bash test/test_plot_2.sh", \@failed_tests);
                run_bash_test("bash test/test_export.sh cpu_test", \@failed_tests);
                run_bash_test("bash test/test_export.sh cpu_test2", \@failed_tests);
                run_bash_test("bash test/test_wallclock_time.sh", \@failed_tests);
                run_bash_test("bash test/test_gpu_plot.sh", \@failed_tests);
        }

        $indentation--;

        if(@failed_tests == 0) {
                ok "All tests successful!";
        } else {
                no_suicide_error "Not all tests successful!";
                foreach my $failed_test (@failed_tests) {
                        no_suicide_error "FAILED TEST -> $failed_test";
                }
                exit scalar @failed_tests;
        }
}

sub run_bash_test {
        my $command = shift;
        my $failed_tests = shift;
        my $errors = 0;
        modify_system($command);
        my $error_code = $?;
        my $exit_code = $error_code >> 8;
        my $sig_code = $error_code & 127;
        if($exit_code) {
                no_suicide_error "The script `$command` did not exit with exit-code 0, but instead $exit_code (SIG-Code: $sig_code)";
                $errors++;
                push @$failed_tests, $command;
        }   
        return $errors;
}

sub get_working_directory {
        $indentation++;
        debug_sub "get_working_directory()";
        my $cwd = '';
        if(exists $ENV{SLURM_JOB_ID}) {
                if(program_installed("scontrol")) {
                        my $command = qq#scontrol show job $ENV{SLURM_JOB_ID} | egrep "^\\s*WorkDir=" | sed -e 's/^\\s*WorkDir=//'#;
                        $cwd = debug_qx($command);
                        chomp $cwd;
                        debug "Found CWD: $cwd";
                }
        }

        if (!-d $cwd) {
                $cwd = getcwd();
        }

        if(!defined $cwd) {
                error "WARNING: CWD Seems to be empty!"
        }

        if(!-d $cwd) {
                error "WARNING: CWD ($cwd) could not be found!!!";
        }

        $indentation--;
        return $cwd;
}

sub run_ssh_command_on_all_nodes {
        $indentation++;
        my $command = shift;
        debug_sub "run_ssh_command_on_all_nodes($command)";

        if(exists $ENV{SLURM_JOB_ID} && exists $ENV{SLURM_JOB_NODELIST}) {
                my @server = get_servers_from_SLURM_NODES($ENV{SLURM_JOB_NODELIST});

                foreach my $this_server (@server) {
                        debug_system("ssh -o LogLevel=ERROR $this_server '$command'");
                }
        } else {
                debug "Not in a Slurm-Job, therefore, no allocated nodes.";
        }
        $indentation--;
}

sub check_needed_programs {
        $indentation++;
        debug_sub "check_needed_programs()";

        my $errors = 0;

        my @needed_programs = (
                $python,
                "ssh",
                "bash",
                "srun",
                "pip3",
                "squeue",
                "mongod"
        );

        if(is_ml()) {
                @needed_programs = (
                        $python,
                        "ssh",
                        "bash",
                        "srun",
                        "squeue",
                        "mongod"
                );
        }

        foreach my $program (@needed_programs) {
                if(program_installed($program)) {
                        ok_debug "$program seems to be installed correctly.";
                } else {
                        warning "$program does NOT seem to be installed correctly!";
                        $errors++;
                }
        }
        $indentation--;
        return $errors;
}

sub get_log_file_name {
        $indentation++;
        my $debug_log_folder = get_working_directory()."/debuglogs/";
        my $j = 0;
        my $i = "$debug_log_folder$j";
        while (-e "$debug_log_folder$j") {
                $j++;
                $i = "$debug_log_folder$j";
        }
        if(!-d $debug_log_folder) {
                debug_system("mkdir $debug_log_folder");
        }
        debug_system("touch $i");
        $indentation--;
        return $i;
}

#### Debug-Outputs

sub stderr_debug_log ($$) {
        my $show = shift;
        my $string = shift;

        if($debug_log_file) {
                open my $fh, '>>', "$debug_log_file";
                print $fh $string;
                close $fh;
        }
        return unless $show;
        warn $string;
}

sub dryrun (@) {
        my @msgs = @_;
        if($options{dryrun}) {
                my $begin = "DRYRUN:\t\t";
                my $spaces = "├".$indentation_char x $indentation." ";
                if(@msgs) {
                        foreach my $msg (@msgs) {
                                $msg = "$begin$spaces".join("\n$begin$spaces", split(/\R/, $msg));
                                stderr_debug_log $options{dryrunmsgs}, color('magenta').$msg.color('reset')."\n";
                        }
                }

                return 1;
        }
}

sub ok_debug (@) {
        my $spaces = "├".$indentation_char x $indentation." ";
        foreach (@_) {
                stderr_debug_log $options{debug}, color('green')."OK_DEBUG:\t$spaces$_".color('reset')."\n";
        }
}

sub ok (@) {
        foreach (@_) {
                stderr_debug_log 1, color('green')."OK:\t\t$_".color('reset')."\n";
        }
}

sub message (@) {
        my $spaces = "├".$indentation_char x $indentation." ";
        foreach (@_) {
                stderr_debug_log $options{messages}, color('cyan')."MESSAGE: \t$spaces$_".color('reset')."\n";
        }
}

sub warning (@) {
        return if !$options{warnings};
        my @warnings = @_;
        my $sub_name = '';
        my $spaces = "├".$indentation_char x $indentation." ";
        foreach my $wrn (@warnings) {
                stderr_debug_log $options{warnings}, color('yellow')."WARNING$sub_name:\t$spaces$wrn".color('reset')."\n";
        }
}

sub debug (@) {
        foreach (@_) {
                my $spaces = "├".$indentation_char x $indentation." ";
                stderr_debug_log $options{debug}, color('cyan')."DEBUG:\t\t$spaces$_".color('reset')."\n";
        }
}

sub debug_sub (@) {
        my $spaces = "├─".((($indentation - 1) < 0) ? "" : $indentation_char x ($indentation - 1)." ");
        foreach (@_) {
                stderr_debug_log $options{debug}, color('blue')."DEBUG_SUB:\t$spaces".color("underline")."$_".color('reset')."\n";
        }
}

sub no_suicide_error (@) {
        foreach (@_) {
                stderr_debug_log 1, color('red')."ERROR:\t\t$_.".color('reset')."\n";
        }
}

sub error (@) {
        foreach (@_) {
                stderr_debug_log 1, color('red')."ERROR:\t\t$_.".color('reset')."\n";
        }
        gentle_suicide();
        exit 1;
}

sub read_ini_file {
        my $file = shift;
        $indentation++;
        debug_sub "read_ini_file($file)";
        my $conf;
        open (my $INI, $file) || error "Can't open ini-file `$file`: $!\n";
        my $section = '';
        while (<$INI>) {
                chomp;
                if (/^\s*\[\s*(.+?)\s*\]\s*$/) {
                        $section = $1;
                }

                if ( /^\s*([^=]+?)\s*=\s*(.*?)\s*$/ ) {
                        $conf->{$section}->{$1} = $2;
                }
        }
        close ($INI);
        $indentation--;
        return %{$conf};
}

sub gentle_suicide {
        $indentation++;
        debug_sub "gentle_suicide()";
        if($options{originalslurmid}) {
                debug_qx(qq#scancel --signal=USR1 --batch $options{originalslurmid}#);
        }
        $indentation--;
}

sub debug_loaded_modules {
        $indentation++;
        debug_sub "debug_loaded_modules()";
        if($options{debug}) {
                my $command = q#for i in $(seq -f "%03g" $_ModuleTable_Sz_); do eval "echo \$_ModuleTable${i}_"; done | base64 --decode | sed -e 's/,/,\n/g'#;
                modify_system($command);
                warn "\n";
        }
        $indentation--;
}
