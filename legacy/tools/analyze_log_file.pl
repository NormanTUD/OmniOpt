#!/usr/bin/perl

# todo: parameter bester jobs herausfinden (nach loss), cnt ausgeben in csv!!!!!!!!!

use strict;
use warnings;
use Term::ANSIColor;
use Data::Dumper;
use Time::HiRes qw( time );

my $start = time();

my %options = (
    slurmid => undef,
    csv => 0,
    skipcancelled => 0,
    debug => 0,
    precision => 2,
    i => 0,
    nodb => 0
);

sub debug (@) {
    my @msg = @_;

    if($options{debug}) {
        foreach (@msg) {
            warn color("WHITE ON_YELLOW").$_.color("reset")."\n";
        }
    }
}

analyze_params(@ARGV);

sub _help {
        my $exit_code = shift;
        print <<EOF;
Analyzes slurm log files.

Parametesr:

        --slurmid=12345                 Checks slurm-12345.out
        --precision=5                   Sets the precision for printing out results
        --debug                         Enables debug
        --i=123                         Starts at 123
        --skipcancelled                 Skips cancelled jobs
        --redocache                     Redos the cache
        --nodb                          Disables checking for results in the DB
        --csv                           Enables CSV output
EOF
        exit($exit_code);
}

sub analyze_params {
    my @params = @_;

    foreach (@params) {
        if(/^--slurmid=(\d+)$/) {
            $options{slurmid} = $1;
        } elsif(/^--precision=(\d+)$/) {
            $options{precision} = $1;
        } elsif(/^--i=(\d+)$/) {
            $options{i} = $1;
        } elsif(/^--debug$/) {
            $options{debug} = 1;
        } elsif (/^--skipcancelled$/) {
            $options{skipcancelled} = 1;
        } elsif (/^--redocache$/) {
            $options{redocache} = 1;
        } elsif (/^--nodb$/) {
            $options{nodb} = 1;
        } elsif (/^--csv$/) {
            $options{csv} = 1;
        } else {
            warn "Unknown parameter `$_`\n";
            _help(1);
        }
    }

    if($options{slurmid}) {
        if(!is_valid_number($options{slurmid})) {
            die("$options{slurmid} is not a slurm-id-number");
        }
    } else {
        die("Cannot continue without slurm id");
    }

    debug Dumper \%options;
}

main();

sub remove_carriage_returns {
        my $logfile = shift;
        my $command = 'bash tools/remove_carriage_returns.sh '.$logfile;
        debug $command;
        qx($command)
}

sub main {
    debug "main";
    my $logfile = 'slurm-'.$options{slurmid}.'.out';
    remove_carriage_returns($logfile);
    debug "Checking for $logfile";

    if(-e $logfile) {
        # number of workers herausfinden
        debug "Found $logfile";

        my %data = (
                projectname => get_project_name_from_logfile($logfile),
                slurmid => $options{slurmid}
        );

        ($data{file}, $data{number_of_jobs}) = read_file($logfile, 'trials with best loss');
        debug "Found $data{number_of_jobs} executions";

        if($data{number_of_jobs}) {
            ($data{job_runtime_slurm}, $data{job_runtime_human_readable}) = get_job_runtime($options{slurmid});
            debug "Job apparently took $data{job_runtime_slurm} seconds (sacct-data)";

            $data{number_of_workers} = get_number_of_workers($logfile);
            $data{slurm_errors} = get_had_slurm_error($logfile);

            my %jobdata = (
                dbavg => -1,
                dbreal => -1
            );
            %jobdata = get_avg_and_full_runtime($data{projectname});

            $data{dbreal} = convert_seconds_to_hhmmss($jobdata{dbreal});
            $data{dbavg} = convert_seconds_to_hhmmss($jobdata{dbavg});
            $data{maxdim} = $jobdata{maxdim};
            $data{results} = $jobdata{results};

            $data{jobs_per_second} = undef;
            $data{jobs_per_real_second} = undef;
            if(defined $data{job_runtime_slurm}) {
                $data{jobs_per_second} = sprintf("%.$options{precision}f", $data{number_of_jobs} / $data{job_runtime_slurm});
            } else {
                debug "job_runtime_slurm is not defined!";
            }

            if(defined $jobdata{dbreal}) {
                $data{jobs_per_real_second} = sprintf("%.$options{precision}f", $data{number_of_jobs} / $jobdata{dbreal});
            } else {
                debug "dbreal is not defined!";
            }

            if(!defined $data{jobs_per_real_second}) {
                $data{jobs_per_real_second} = undef;
            }

            if(!defined $data{jobs_per_second}) {
                $data{onejobeveryseconds} = "NaN";
            } else {
                $data{onejobeveryseconds} = sprintf("%.$options{precision}f", 1 / $data{jobs_per_second});
            }
            if(!defined $data{jobs_per_real_second}) {
                $data{onejobeverysecondsreal} = "NaN";
            } else {
                $data{onejobeverysecondsreal} = sprintf("%.$options{precision}f", 1 / $data{jobs_per_real_second});
            }

            #if($options{csv}) {
                my $csv_string = get_csv_string(%data);
                print $csv_string;
            #} else {
            #   print create_string(%data)."\n";
            #}
        } else {
            die "No jobs done in `$logfile`";
        }
    } else {
        die "$logfile not found";
    }
}

sub get_csv_string {
    my %data = @_;
    my @keys = qw(
            projectname
            jobs_per_second
            number_of_workers
            jobs_per_real_second
            dbreal
            job_runtime_slurm
            number_of_jobs
            slurmid
            slurm_errors
    );

    my @items = ();
    foreach (@keys) {
            push @items, $data{$_};
    }

    foreach (sort { $a cmp $b || $a <=> $b } keys %{$data{results}{result}}) {
        push @keys, $_;
        push @items, $data{results}{result}{$_};
    }

    foreach (sort { $a cmp $b || $a <=> $b } keys %{$data{results}{parameters}}) {
        push @keys, $_;
        push @items, $data{results}{parameters}{$_};
    }

    my $seperator = ";";

    if($options{i} == 0) {
        print join($seperator, @keys)."\n";
    }

    @items = map { if (!defined $_) { "NaN" } else { $_ } } @items;
    return join($seperator, @items)."\n";
}

sub read_file {
    my $filename = shift;
    my $re = shift;

    my $contents = '';
    my $number_of_lines = 0;
    open my $fh, '<', $filename or die $!;
    while (<$fh>) {
        if($options{skipcancelled}) {
            exit if /CANCELLED/;
        }
        if(($re && /$re/) || !$re) {
            $contents .= $_;
            $number_of_lines++;
        }
    }
    close $fh;

    if(wantarray()) {
        return ($contents, $number_of_lines);
    } else {
        return $contents;
    }
}

sub get_job_runtime {
    my $job_id = shift;
    debug "get_job_runtime($job_id)";

    my $command = qq#sacct -j $job_id --format=JobID,Elapsed | egrep "^$job_id\\s+"#;
    my $job_runtime_human_readable = run_command($command);

    if($job_runtime_human_readable =~ m#^\d+\s*(?<whole>(?:(?<days>\d)-)?(?<hours>\d+):(?<minutes>\d+):(?<seconds>\d+))\s*$#) {
        my $seconds = $+{seconds};
        $seconds += 60 * $+{minutes};
        $seconds += 3600 * $+{hours};
        if(exists $+{days}) {
            $seconds += 86400 * $+{days};
        }
        debug "Converted $+{whole} to $seconds seconds";
        if(wantarray()) {
            return ($seconds, $+{whole});
        } else {
            return $seconds;
        }
    } else {
        die "Could not get runtime from string `$job_runtime_human_readable`";
    }
}

sub is_valid_number {
    my $number = shift;

    if($number =~ m#^\d+$#) {
        return 1;
    }

    return 0;
}

sub run_command {
    my $command = shift;
    debug $command;
    my $output = qx($command);
    chomp $output;
    return $output;
}

sub get_had_slurm_error {
        my $logfile = shift;

        my $contents = read_file($logfile);
        $contents = remove_colors_from_bash_output($contents);

        if($contents =~ m#Job step creation temporarily disabled#gi) {
                return 1;
        } else {
                return 0;
        }
}

sub get_number_of_workers {
        my $logfile = shift;
        debug "get_number_of_workers($logfile)";

        my $contents = read_file($logfile, 'slurm_script');
        $contents = remove_colors_from_bash_output($contents);

        $contents =~ s/\r\n//g;

        #if($contents =~ m#slurm_script\s.*?(\d+)$#) {
        if($contents =~ m#slurm_script\s*.*?(\d+).*?$#) {
            my $number_of_workers = $1;
            return $number_of_workers;
        } else {
            die "Cannot determine the number of workers for `$logfile`!";
        }
}

sub remove_colors_from_bash_output {
        my $code = shift;

        $code =~ s/\x1b\[[0-9;]*m//g;
        $code =~ s/\x1b\[[0-9;]*[mG]//g;
        $code =~ s/\x1b\[[0-9;]*[mGKH]//g;
        $code =~ s/\x1b\[[0-9;]*[a-zA-Z]//g;

        return $code;
}

sub get_avg_and_full_runtime {
    my $project = shift;

    my %res = (
        dbavg => undef,
        dbreal => undef
    );


    my $projectdir = "./projects/$project";

    my $lockfile = "$projectdir/mongodb/mongod.lock";

    unlink $lockfile;

    if(-d $projectdir) {
        my $output = '';
        if(!$options{nodb}) {
            my $redo_cache = '';
            if ($options{redocache}) {
                $redo_cache = ' 0 ';
            }
            my $command = "bash load_cached.sh$redo_cache python3 script/analyzedb.py --project=$project 2>&1";
            debug $command;
            $output = qx($command);
        }

        debug $output;



        if($output =~ m#avg:\s*(\d+)#) {
            $res{dbavg} = $1;
        }

        if($output =~ m#wholetime:\s*(\d+)#) {
            $res{dbreal} = $1;
        }

        if($output =~ m#maxdim:\s*(\d+)#) {
            $res{maxdim} = $1;
        }

        while ($output =~ m#(x_\d*):\s*([\.\d]+)#g) {
            $res{"results"}{"parameters"}{$1} = sprintf("%.".$options{precision}."f", $2);
        }

        while ($output =~ m#(cnt|result|loss|q):\s*([\.\d]+)#gi) {
            $res{"results"}{"result"}{$1} = sprintf("%.".$options{precision}."f", $2);
        }
    } else {
        warn "$projectdir not found!!!\n";
    }

    return %res;
}

sub get_project_name_from_logfile {
        my $logfile = shift;

        my $name = undef;
        my $content = read_file($logfile, 'slurm_script');
        if($content =~ m#slurm_script\s+(.*?)\s#gi) {
                $name = $1;
        }
        return $name;
}

sub create_string {
    my %data = @_;
    my @data_names = (
        {
            key => "slurmid",
            name => "Slurm-ID",
            show => 1
        },
        {
            key => "projectname",
            name => "Project",
            show => 1
        },
        {
            key => "number_of_workers",
            name => "Workers",
            show => 1
        },
        {
            key => "job_runtime_human_readable",
            name => "Runtime (Slurm)",
            show => 1
        },
        {
            key => "dbreal",
            name => "Runtime (DB)",
            show => 1
        },
        {
            key => "number_of_jobs", 
            name => "Jobs done",
            show => 1
        },
        {
            key => "dbavg",
            name => "Avg. runtime/J (DB)",
            show => 1
        },
        {
            key => "jobs_per_second",
            name => "J/s (Slurm)",
            show => 1
        },

        {
            key => "onejobeveryseconds",
            name => "Job done every s (Slurm)",
            show => 1
        },
        {
            key => "jobs_per_real_second",
            name => "J/s (DB)",
            show => 1
        },
        {
            key => "onejobeverysecondsreal",
            name => "Job done every s (DB)",
            show => 1
        },


        {
            key => "file",
            name => "Filename",
            show => 0
        },
        {
            key => "job_runtime_slurm",
            name => "Job runtime (Slurm)",
            show => 0
        }
    );

    my @string_types = ();

    foreach my $this (@data_names) {
        my $this_key = $this->{key};
        my $this_name = $this->{name};
        my $this_show = $this->{show};
        my $this_data = $data{$this_key};

        if($this_show) {
            if (exists $data{$this_key}) {
                if($this_data !~ /^-/) {
                    push @string_types, "$this_name: ".color("blue").$this_data.color("reset");
                } else {
                    debug "Negative value of $this_key ($this_name -> $this_data), not showing it";
                }
            } else {
                debug "$this_key does not exist!";
            }
        } else {
                debug "Not showing $this_key, because it's `show`-key is 0";
        }
    }

    my $string = join(" | ", @string_types)."\n";
    return $string;
}

sub convert_seconds_to_hhmmss {
    return undef if !defined($_[0]);
    my $hourz = int($_[0]/3600);
    my $leftover = $_[0] % 3600;
    my $minz = int($leftover/60);
    my $secz = int($leftover % 60);

    return sprintf("%02d:%02d:%02d", $hourz, $minz, $secz);
}

my $end = time();

debug sprintf("perl analyze_log_file.pl, runtime: %.2fs", $end - $start);
