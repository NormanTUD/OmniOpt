#!/usr/bin/perl

$| = 1;

use strict;
use warnings;
use Data::Dumper;

use lib './perllib';
use Env::Modify;

use File::Basename;
use Cwd 'abs_path';
my $basedirname = dirname(abs_path($0));

use Term::ANSIColor;


sub myqx ($;$);
sub debug ($);

sub p ($;$) {
        my $percent = shift;
        my $status = shift // "";
        if($ENV{DISPLAYGAUGE}) {
                warn "PERCENTGAUGE: $percent\n";
                if($status) {
                        warn "GAUGESTATUS: $status\n";
                }
        }
}

my %options = (
        debug => 0,
        project => '',
        int => 0,
        seperator => ';',
        projectdir => './projects',
        maxvalue => undef,
        maxtime => undef,
        keepmongodb => 0,
        nopip => 0,
        dontloadmodules => 0,
        SHOWFAILEDJOBSINPLOT => exists($ENV{SHOWFAILEDJOBSINPLOT}) ? $ENV{SHOWFAILEDJOBSINPLOT} : 0
);

analyze_args(@ARGV);

main();

sub main {
        my $projectfolder = get_project_folder($options{project});

        p 1, "Loading modules...";
        modules_load(
                "modenv/scs5",
                "MongoDB/4.0.3",
                "Hyperopt/0.2.2-fosscuda-2019b-Python-3.7.4",
                "Python/3.7.4-GCCcore-8.3.0",
                "matplotlib/3.1.1-foss-2019b-Python-3.7.4"
        );
        p 10, "Modules loaded";


        modify_system("export DONTCHECKMONGODBSTART=1");
        modify_system("export SHOWFAILEDJOBSINPLOT=$options{SHOWFAILEDJOBSINPLOT}");
        modify_system('PYTHONPATH=$HOME/.local/lib/python3.7/site-packages:/software/haswell/Python/3.7.4-GCCcore-8.3.0/lib/python3.7/site-packages:$PYTHONPATH');
        print "If the module past is missing, install the package future via pip3 install --user future\n";
        print "If the module pymongo is missing, install the package future via pip3 install --user pymongo\n";



        if(!$options{nopip}) {
                myqx("pip3 install --user psutil");
        }

        my $command = "python3 -u script/plot3.py --projectdir=".$options{projectdir}.' --project='.$options{project};
        $command .= " --maxvalue=".$options{maxvalue} if $options{maxvalue};
        $command .= " --maxtime=".$options{maxtime} if $options{maxtime};
        $command .= " --int" if $options{int};
        $command .= " --parameter=".$options{parameter} if $options{parameter};
        $command .= " --switchaxes" if $options{switchaxes};

        p 15, "Getting SLURM-ID";

        my $slurm_id_project = get_id_of_project();

        p 30, "Got Slurm-ID";

        if(exists $ENV{mongodbmachine} && exists $ENV{mongodbport} && length $ENV{mongodbmachine} >= 2 && length $ENV{mongodbport} >= 2) {
                my ($dbip, $dbport) = ($ENV{mongodbmachine}, $ENV{mongodbport});

                warn "Found the following ip:port in \$ENV: $dbip:$dbport\n";

                $command .= " --mongodbmachine=$dbip --mongodbport=$dbport";
                p 40, "Starting Pythonscript (1)";
                print myqx($command, 1);
        } elsif (defined $slurm_id_project) {
                my $ipfilefolder = "$projectfolder/ipfiles";
                if(-d $ipfilefolder) {
                        my ($dbip, $dbport) = map { read_file("$ipfilefolder/$_") } ("mongodbserverip-$slurm_id_project", "mongodbportfile-$slurm_id_project");
                        chomp $dbip;
                        chomp $dbport;

                        warn "Found the following ip:port: $dbip:$dbport\n";
                        $command .= " --mongodbmachine=$dbip --mongodbport=$dbport";
                        p 40, "Starting Pythonscript (2)";
                        print myqx($command);
                } else {
                        die "$ipfilefolder not found";
                }
        } else {
                warn "The slurm job for the project `$options{project}` could not be found. Is it running and has the same slurm name? If not, a new database instance will be started and this warning can be ignored.\n";

                my $lockfile = "$projectfolder/mongodb/mongod.lock";

                if(!-e $lockfile) {
                        print myqx($command);
                        if(!$options{keepmongodb}) {
                                p 99, "Ending MongoDB";
                                myqx("python3 script/endmongodb.py --projectdir=".$options{projectdir}.' --project='.$options{project});
                                p 100, "Ended MongoDB";
                        }
                } else {
                        die "There is a lockfile `$lockfile`, that means the server was either not shut down correctly or it is running in another job. Better not doing anything automatically.\n";
                }
        }
        p 100, "Ended script";
}

sub read_file {
        my $file = shift;

        my $contents = '';
        open my $fh, '<', $file or die "Error opening file $file: $!";

        while (<$fh>) {
                $contents .= $_;
        }

        close $fh;

        return $contents;
}

sub get_job_work_dir {
        my $slurmid = shift;
        my $workdir = '';
        $workdir = qx(scontrol show job $slurmid | grep WorkDir | sed -e 's/.*=//');
        chomp $workdir;
        return $workdir;
}

sub get_id_of_project {
        my %running_jobs = get_running_jobs();

        foreach my $name (keys %running_jobs) {
                debug "$name -> $options{project}?";
                if($name eq $options{project} || $name eq qq#'$options{project}'#) {
                        my $job_work_dir = get_job_work_dir($running_jobs{$name});
                        if ($basedirname =~ m#$job_work_dir/#) {
                                return $running_jobs{$name};
                        }
                }
        }

        return undef;
}

sub get_running_jobs {
        my $command = 'sacct --format="JobID,State,JobName%100"';

        my @jobs = map { chomp $_; $_; } myqx $command;

        my %running_jobs = ();

        foreach (@jobs) {
                if(m#^(\d+)\s+RUNNING\s{4,}(.*?)\s*$#) {
                        my $id = $1;
                        my $name = $2;

                        $running_jobs{$name} = $id;
                }
        }

        p 20, "Got running jobs";

        return %running_jobs;
}

sub get_signal_name {
        my $signal = shift;

        if($signal =~ m#^\d+$#) {
                my $name = scalar qx(kill -l $signal);
                chomp $name;
                return $name;
        } else {
                return "unknown";
        }
}

sub myqx ($;$) {
        my $command = shift;
        my $die_on_error = shift // 0;

        debug "command: $command";

        if(wantarray()) {
                my @res = qx($command);
                my $error_code = $?;
                my $exit_code = $error_code >> 8;
                my $signal_code = $error_code & 127;
                warn color("red")."Exited with $exit_code".color("reset")."\n" if $exit_code;
                warn color("red")."Program exited, got signal $signal_code (".get_signal_name($signal_code).")".color("reset")."\n" if $signal_code;
                exit $error_code if($die_on_error && $error_code != 0);
                return @res;
        } else {
                my $res = qx($command);
                my $error_code = $?;
                my $exit_code = $error_code >> 8;
                my $signal_code = $error_code & 127;
                warn color("red")."Exited with $exit_code".color("reset")."\n" if $exit_code;
                warn color("red")."Program exited, got signal $signal_code (".get_signal_name($signal_code).")".color("reset")."\n" if $signal_code;
                exit $error_code if($die_on_error && $error_code != 0);
                return $res;
        }
}

sub help {
        my $exit = shift // 0;
        print <<EOF;
tools/plot.pl : Plot database of OmniOpt run

Parameters:

--debug                                 Enables debug mode
--project=NAME                          Sets the name of the project to be plotted
--switchaxes                            Switches X and Y axes in plot
--maxvalue=FLOAT                        Sets a maximal value to get from the DB
--maxtime=INT                           Sets a maximal time (in unix epoch time) to get from the DB
--projectdir=/path/                     Path of the projects
--int                                   Round all non-ints to ints
--keepmongodb                           Do not end MongoDB after plotting
--dontloadmodules                       Disables loading of the modules (e.g. for manual, faster loading)
--nopip                                 Disables psutil installation
--help                                  This help

Needed modules when --dontloadmodules is specified:

ml modenv/scs5
ml MongoDB/4.0.3
ml Hyperopt/0.2.2-fosscuda-2019b-Python-3.7.4
ml Python/3.7.4-GCCcore-8.3.0
ml matplotlib/3.1.1-foss-2019b-Python-3.7.4

EOF
        if($exit) {
                exit 0;
        }
}

sub analyze_args {
        my @args = @_;

        foreach (@args) {
                if(m#^--debug$#) {
                        $options{debug} = 1;
                } elsif (m#^--help$#) {
                        help(1);
                } elsif (m#^--keepmongodb$#) {
                        $options{keepmongodb} = 1;
                } elsif (m#^--nopip$#) {
                        $options{nopip} = 1;
                } elsif (m#^--dontloadmodules$#) {
                        $options{dontloadmodules} = 1;
                } elsif (m#^--project=(.*)$#) {
                        $options{project} = $1;
                } elsif (m#^--switchaxes$#) {
                        $options{switchaxes} = 1;
                } elsif (m#^--maxtime=(\d+)$#) {
                        $options{maxtime} = $1;
                } elsif (m#^--maxvalue=(.+)$#) {
                        $options{maxvalue} = $1;
                } elsif (m#^--parameter=(.+)$#) {
                        $options{parameter} = $1;
                } elsif (m#^--projectdir=(.+)$#) {
                        $options{projectdir} = $1;
                } elsif (m#^--int$#) {
                        $options{int} = 1;
                } else {
                        warn "Unknown parameter $_";
                        help(1);
                }
        }

        if($options{project}) {
                my $proj_folder = get_project_folder($options{project});
                if(!-d $proj_folder) {
                        die "Project folder `$proj_folder` not found";
                }
        } else {
                die "Unknown project";
        }
}

sub get_project_folder {
        my $project = shift;


        return "$options{projectdir}/$project/";
}

sub debug ($) {
        my $arg = shift;
        if($options{debug}) {
                warn "$arg\n";
        }
}

sub modules_load {
        my @modules = @_;
        if($options{dontloadmodules}) {
                return 1;
        }
        foreach my $mod (@modules) {
                module_load($mod);
        }

        return 1;
}

sub modify_system {
        my $command = shift;
        debug "modify_system($command)";
        return Env::Modify::system($command);
}

sub module_load {
        my $toload = shift;

        if($options{dontloadmodules}) {
                return 1;
        }

        if($toload) {
                my $lmod_path = $ENV{LMOD_CMD};
                my $command = "eval \$($lmod_path sh load $toload)";
                debug $command;
                local $Env::Modify::CMDOPT{startup} = 1;
                modify_system($command);
        } else {
                warn 'Empty module_load!';
        }
        return 1;
}
